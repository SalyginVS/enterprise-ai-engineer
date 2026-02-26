"""
EXP-010: Валидация аргументов + System Prompt + Fallback
Цель: закрыть баги из EXP-009 перед переходом к CustomerSupportAgent.
Модель: qwen3:14b
"""

import requests
import json
from pydantic import BaseModel, field_validator, ValidationError

OLLAMA_URL = "http://192.168.0.128:11434/api/chat"
MODEL = "qwen3:14b"
MAX_ITERATIONS = 10

# ─────────────────────────────────────────────
# 1. Pydantic-схемы для валидации аргументов
# ─────────────────────────────────────────────
# Это первое появление Pydantic в нашем коде.
# В M1 она станет основой для всей валидации агента.
# Здесь используем её точечно: только для аргументов инструментов,
# которые модель могла галлюцинировать.

VALID_CURRENCIES = {"RUB", "USD", "EUR", "UAH"}

class InvoiceStatusArgs(BaseModel):
    invoice_id: str

    @field_validator("invoice_id")
    @classmethod
    def validate_format(cls, v):
        if not v.startswith("INV-"):
            raise ValueError(f"Некорректный формат номера счёта: '{v}'. Ожидается INV-YYYY-NNN")
        return v

class ConvertCurrencyArgs(BaseModel):
    amount: float
    from_currency: str
    to_currency: str

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError(f"Сумма должна быть положительной, получено: {v}")
        return v

    @field_validator("from_currency", "to_currency")
    @classmethod
    def validate_currency(cls, v):
        upper = v.upper()
        if upper not in VALID_CURRENCIES:
            raise ValueError(f"Неизвестная валюта: '{v}'. Допустимые: {VALID_CURRENCIES}")
        return upper

# ─────────────────────────────────────────────
# 2. Описания инструментов (без изменений)
# ─────────────────────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_invoice_status",
            "description": "Возвращает статус счёта: оплачен, ожидает оплаты, просрочен. Также возвращает сумму и валюту.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "Номер счёта в формате INV-YYYY-NNN, например INV-2024-001"
                    }
                },
                "required": ["invoice_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": (
                "Конвертирует сумму из одной валюты в другую. "
                "ВАЖНО: используй только реальные значения из предыдущих результатов, "
                "не придумывай сумму или валюту самостоятельно."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Сумма для конвертации — берётся из результата get_invoice_status"
                    },
                    "from_currency": {
                        "type": "string",
                        "description": "Исходная валюта из результата инструмента: UAH, RUB, USD или EUR"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "Целевая валюта: UAH, RUB, USD или EUR"
                    }
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    }
]

# ─────────────────────────────────────────────
# 3. System prompt
# ─────────────────────────────────────────────
# Явные инструкции снижают недетерминизм модели.
# Ключевое правило: не вызывай второй инструмент до получения
# результата первого, если второй зависит от первого.

SYSTEM_PROMPT = """Ты — ассистент по работе со счетами и финансовыми данными.

Правила работы с инструментами:
1. Всегда используй инструменты для получения актуальных данных — не придумывай статусы и суммы.
2. Если нужно конвертировать сумму из счёта — сначала получи данные счёта, затем используй реальную сумму и валюту из результата.
3. Никогда не вызывай convert_currency с угаданными значениями — только с теми, что получены из get_invoice_status.
4. Всегда формулируй финальный текстовый ответ пользователю после получения всех данных."""

# ─────────────────────────────────────────────
# 4. Mock-реализации
# ─────────────────────────────────────────────

def get_invoice_status(invoice_id: str) -> dict:
    statuses = {
        "INV-2024-001": {"status": "Оплачен",        "amount": 15400.00, "currency": "UAH", "date": "2024-01-15"},
        "INV-2024-002": {"status": "Ожидает оплаты", "amount": 8750.50,  "currency": "UAH", "date": "2024-01-20"},
        "INV-2024-003": {"status": "Просрочен",       "amount": 32100.00, "currency": "UAH", "date": "2023-12-01"},
    }
    return statuses.get(invoice_id, {"status": "Не найден", "invoice_id": invoice_id})

def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    rates_to_usd = {"RUB": 0.011, "USD": 1.0, "EUR": 1.08, "UAH": 0.024}
    rates_from_usd = {"RUB": 91.0, "USD": 1.0, "EUR": 0.926, "UAH": 41.5}
    amount_in_usd = amount * rates_to_usd[from_currency]
    result = amount_in_usd * rates_from_usd[to_currency]
    return {
        "original_amount": amount,
        "from_currency": from_currency,
        "converted_amount": round(result, 2),
        "to_currency": to_currency,
        "rate_used": f"1 {from_currency} = {round(rates_to_usd[from_currency] * rates_from_usd[to_currency], 4)} {to_currency}"
    }

# ─────────────────────────────────────────────
# 5. Валидирующий роутер инструментов
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, raw_args: dict) -> str:
    """
    Выполняет инструмент ПОСЛЕ валидации аргументов.
    Если аргументы некорректны — возвращает описание ошибки
    вместо того чтобы упасть с исключением или выполнить
    некорректную операцию.
    """
    try:
        if tool_name == "get_invoice_status":
            args = InvoiceStatusArgs(**raw_args)
            result = get_invoice_status(args.invoice_id)

        elif tool_name == "convert_currency":
            args = ConvertCurrencyArgs(**raw_args)
            result = convert_currency(args.amount, args.from_currency, args.to_currency)

        else:
            result = {"error": f"Неизвестный инструмент: {tool_name}"}

    except ValidationError as e:
        # Аргументы от модели не прошли валидацию
        errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
        result = {
            "error": "Некорректные аргументы инструмента",
            "details": errors,
            "received_args": raw_args
        }
        print(f"  ⚠ ВАЛИДАЦИЯ ПРОВАЛЕНА: {errors}")

    return json.dumps(result, ensure_ascii=False)

# ─────────────────────────────────────────────
# 6. Agent loop с fallback на пустой ответ
# ─────────────────────────────────────────────

def run_agent(user_question: str) -> str:
    messages = [{"role": "user", "content": user_question}]

    print(f"\n{'='*60}")
    print(f"Вопрос: {user_question}")
    print(f"{'='*60}")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n[Итерация {iteration}]")

        payload = {
            "model": MODEL,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "tools": tools,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        assistant_message = response.json()["message"]
        tool_calls = assistant_message.get("tool_calls")

        if not tool_calls:
            content = assistant_message.get("content", "").strip()

            # Fallback: пустой ответ — просим модель сформулировать явно
            if not content:
                print(f"  ⚠ Пустой ответ — запрашиваем fallback")
                messages.append({"role": "assistant", "content": ""})
                messages.append({
                    "role": "user",
                    "content": "Пожалуйста, сформулируй итоговый ответ на основе полученных данных."
                })
                continue

            print(f"  → Финальный ответ (итераций: {iteration})")
            return content

        # Обрабатываем tool calls
        print(f"  → Запрошено инструментов: {len(tool_calls)}")
        messages.append({"role": "assistant", **assistant_message})

        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"]["arguments"]

            print(f"  → Вызов: {func_name}({func_args})")
            tool_result = execute_tool(func_name, func_args)
            print(f"  → Результат: {tool_result}")

            messages.append({"role": "tool", "content": tool_result})

    return f"[ОШИБКА] Агент превысил лимит итераций ({MAX_ITERATIONS})."

# ─────────────────────────────────────────────
# 7. Тест-кейсы — включая проблемный из EXP-009
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        # Кейс из EXP-009 который давал пустой ответ и галлюцинацию
        "Проверь счёт INV-2024-003 и скажи, сколько это в USD.",

        # Намеренно некорректный номер счёта — проверяем валидацию
        "Какой статус у счёта 12345?",

        # Стандартный кейс для регрессии
        "Переведи 5000 UAH в EUR.",
    ]

    for question in test_questions:
        answer = run_agent(question)
        print(f"\nОтвет агента:\n{answer}\n")