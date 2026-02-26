"""
EXP-009: Два инструмента + Agent Loop
Цель: освоить цикл агента — модель сама решает,
      какие инструменты вызвать и в каком порядке.
Модель: qwen3:14b
"""

import requests
import json

OLLAMA_URL = "http://192.168.0.128:11434/api/chat"
MODEL = "qwen3:14b"
MAX_ITERATIONS = 10  # защита от бесконечного цикла (preview М3)

# ─────────────────────────────────────────────
# 1. Два инструмента
# ─────────────────────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_invoice_status",
            "description": "Возвращает статус счёта: оплачен, ожидает оплаты, просрочен. Также возвращает сумму и дату.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "Номер счёта, например INV-2024-001"
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
            "description": "Конвертирует сумму из одной валюты в другую по текущему курсу.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Сумма для конвертации"
                    },
                    "from_currency": {
                        "type": "string",
                        "description": "Исходная валюта: RUB, USD, EUR, UAH"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "Целевая валюта: RUB, USD, EUR, UAH"
                    }
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    }
]

# ─────────────────────────────────────────────
# 2. Mock-реализации инструментов
# ─────────────────────────────────────────────

def get_invoice_status(invoice_id: str) -> dict:
    statuses = {
        "INV-2024-001": {"status": "Оплачен",          "amount": 15400.00, "currency": "UAH", "date": "2024-01-15"},
        "INV-2024-002": {"status": "Ожидает оплаты",   "amount": 8750.50,  "currency": "UAH", "date": "2024-01-20"},
        "INV-2024-003": {"status": "Просрочен",         "amount": 32100.00, "currency": "UAH", "date": "2023-12-01"},
    }
    return statuses.get(invoice_id, {"status": "Не найден", "invoice_id": invoice_id})

def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    # Фиксированные курсы для учебного примера
    rates_to_usd = {"RUB": 0.011, "USD": 1.0, "EUR": 1.08, "UAH": 0.024}
    rates_from_usd = {"RUB": 91.0, "USD": 1.0, "EUR": 0.926, "UAH": 41.5}

    from_c = from_currency.upper()
    to_c = to_currency.upper()

    if from_c not in rates_to_usd or to_c not in rates_from_usd:
        return {"error": f"Неизвестная валюта: {from_currency} или {to_currency}"}

    amount_in_usd = amount * rates_to_usd[from_c]
    result = amount_in_usd * rates_from_usd[to_c]

    return {
        "original_amount": amount,
        "from_currency": from_c,
        "converted_amount": round(result, 2),
        "to_currency": to_c,
        "rate_used": f"1 {from_c} = {round(rates_to_usd[from_c] * rates_from_usd[to_c], 4)} {to_c}"
    }

# ─────────────────────────────────────────────
# 3. Роутер инструментов
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name == "get_invoice_status":
        result = get_invoice_status(**tool_args)
    elif tool_name == "convert_currency":
        result = convert_currency(**tool_args)
    else:
        result = {"error": f"Неизвестный инструмент: {tool_name}"}
    return json.dumps(result, ensure_ascii=False)

# ─────────────────────────────────────────────
# 4. Agent Loop
# ─────────────────────────────────────────────

def run_agent(user_question: str) -> str:
    """
    Цикл агента: крутится, пока модель запрашивает инструменты.
    Завершается, когда модель возвращает текстовый ответ без tool_calls.
    """
    messages = [{"role": "user", "content": user_question}]

    print(f"\n{'='*60}")
    print(f"Вопрос: {user_question}")
    print(f"{'='*60}")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n[Итерация {iteration}]")

        payload = {
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        assistant_message = response.json()["message"]

        tool_calls = assistant_message.get("tool_calls")

        if not tool_calls:
            # Модель дала финальный ответ — выходим из цикла
            print(f"  → Финальный ответ (итераций потребовалось: {iteration})")
            return assistant_message.get("content", "")

        # Модель запросила один или несколько инструментов
        print(f"  → Запрошено инструментов: {len(tool_calls)}")
        messages.append({"role": "assistant", **assistant_message})

        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"]["arguments"]

            print(f"  → Вызов: {func_name}({func_args})")
            tool_result = execute_tool(func_name, func_args)
            print(f"  → Результат: {tool_result}")

            messages.append({
                "role": "tool",
                "content": tool_result
            })

    # Сюда попадаем только если превышен MAX_ITERATIONS
    return f"[ОШИБКА] Агент превысил лимит итераций ({MAX_ITERATIONS}). Требуется ручная проверка."

# ─────────────────────────────────────────────
# 5. Тест-кейсы
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        # Кейс 1: один инструмент
        "Какой статус у счёта INV-2024-002?",

        # Кейс 2: цепочка — сначала узнать сумму, потом конвертировать
        "Проверь счёт INV-2024-003 и скажи, сколько это в USD.",

        # Кейс 3: только конвертация, без счёта
        "Переведи 5000 UAH в EUR.",
    ]

    for question in test_questions:
        answer = run_agent(question)
        print(f"\nОтвет агента:\n{answer}\n")