"""
EXP-011: CustomerSupportAgent — задание 0.4
Цель: первый полноценный агент с инкапсулированной логикой,
      валидацией аргументов и всеми паттернами из EXP-008..010.
Модель: qwen3:14b
"""

import requests
import json
import time
from datetime import datetime
from pydantic import BaseModel, field_validator, ValidationError

OLLAMA_URL = "http://192.168.0.128:11434/api/chat"
MODEL = "qwen3:14b"
MAX_ITERATIONS = 10

# ─────────────────────────────────────────────
# 1. Pydantic-схемы для валидации аргументов
# ─────────────────────────────────────────────

class SearchFAQArgs(BaseModel):
    topic: str

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v):
        if len(v.strip()) < 2:
            raise ValueError("Тема слишком короткая")
        return v.strip().lower()

class GetOrderStatusArgs(BaseModel):
    order_id: str

    @field_validator("order_id")
    @classmethod
    def validate_order_id(cls, v):
        if not v.startswith("ORD-"):
            raise ValueError(
                f"Некорректный формат: '{v}'. Ожидается ORD-NNN"
            )
        return v

class ClassifyRequestArgs(BaseModel):
    text: str
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Текст слишком короткий для классификации")
        return v.strip()

# ─────────────────────────────────────────────
# 2. CustomerSupportAgent
# ─────────────────────────────────────────────

class CustomerSupportAgent:
    """
    Агент поддержки клиентов.
    Умеет: отвечать на FAQ, проверять заказы, классифицировать обращения.
    
    Инкапсулирует: tools, system prompt, agent loop, валидацию,
    mock-базы знаний, простую телеметрию (счётчики вызовов).
    """

    def __init__(self):
        self.model = MODEL
        self.ollama_url = OLLAMA_URL

        # Простая телеметрия — preview паттерна из M1/M2
        self.stats = {
            "total_requests": 0,
            "tool_calls": 0,
            "validation_errors": 0,
            "fallbacks_used": 0,
            "total_iterations": 0,
        }

        # Mock-база FAQ
        self._faq = {
            "доставка": (
                "Стандартная доставка занимает 3-5 рабочих дней. "
                "Экспресс-доставка — 1-2 дня. "
                "Курьерская доставка доступна в пределах города."
            ),
            "возврат": (
                "Возврат товара возможен в течение 30 дней с момента покупки. "
                "Товар должен быть в оригинальной упаковке. "
                "Для оформления возврата обратитесь на support@company.com."
            ),
            "оплата": (
                "Принимаем оплату картами Visa/MasterCard, "
                "банковским переводом и наличными при получении."
            ),
            "гарантия": (
                "Гарантия на все товары — 12 месяцев. "
                "Гарантийный ремонт выполняется бесплатно."
            ),
            "контакты": (
                "Служба поддержки: support@company.com, "
                "тел. +380 44 123-45-67, "
                "режим работы: пн-пт 9:00-18:00."
            ),
        }

        # Mock-база заказов
        self._orders = {
            "ORD-001": {
                "status": "Доставлен",
                "items": ["Ноутбук Lenovo ThinkPad"],
                "total": 45000.00,
                "currency": "UAH",
                "date": "2024-01-10",
                "tracking": "NP20240110001UA",
            },
            "ORD-002": {
                "status": "В пути",
                "items": ["Мышь Logitech MX", "Клавиатура механическая"],
                "total": 3850.00,
                "currency": "UAH",
                "date": "2024-01-18",
                "tracking": "NP20240118002UA",
            },
            "ORD-003": {
                "status": "Обрабатывается",
                "items": ["Монитор Dell 27\""],
                "total": 18900.00,
                "currency": "UAH",
                "date": "2024-01-22",
                "tracking": None,
            },
        }

    # ─────────────────────────────────────────
    # 3. Реализации инструментов
    # ─────────────────────────────────────────

    def _search_faq(self, topic: str) -> dict:
        """Поиск по FAQ — простое совпадение по ключевым словам"""
        for key, answer in self._faq.items():
            if key in topic or topic in key:
                return {"found": True, "topic": key, "answer": answer}
        
        # Частичное совпадение
        for key, answer in self._faq.items():
            if any(word in key for word in topic.split()):
                return {"found": True, "topic": key, "answer": answer}
        
        return {
            "found": False,
            "message": f"Информация по теме '{topic}' не найдена в FAQ.",
            "available_topics": list(self._faq.keys()),
        }

    def _get_order_status(self, order_id: str) -> dict:
        """Поиск заказа по номеру"""
        if order_id in self._orders:
            return {"found": True, **self._orders[order_id]}
        return {
            "found": False,
            "message": f"Заказ {order_id} не найден.",
        }

    def _classify_request(self, text: str) -> dict:
        """Классификация типа обращения"""
        text_lower = text.lower()
        
        complaint_keywords = ["жалоба", "плохо", "ужасно", "недоволен", "проблема", "сломан"]
        question_keywords  = ["как", "когда", "где", "сколько", "что такое", "можно ли"]
        request_keywords   = ["верните", "замените", "исправьте", "оформите", "отмените"]
        
        if any(kw in text_lower for kw in complaint_keywords):
            category = "complaint"
        elif any(kw in text_lower for kw in request_keywords):
            category = "request"
        elif any(kw in text_lower for kw in question_keywords):
            category = "question"
        else:
            category = "other"
        
        return {
            "category": category,
            "confidence": "high" if category != "other" else "low",
        }

    # ─────────────────────────────────────────
    # 4. Описания инструментов для модели
    # ─────────────────────────────────────────

    @property
    def _tools(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_faq",
                    "description": (
                        "Ищет ответ на вопрос в базе FAQ по теме. "
                        "Используй для вопросов о доставке, возврате, "
                        "оплате, гарантии, контактах."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Тема вопроса одним словом: доставка, возврат, оплата, гарантия, контакты"
                            }
                        },
                        "required": ["topic"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_order_status",
                    "description": "Возвращает статус заказа по его номеру.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "Номер заказа в формате ORD-NNN, например ORD-001"
                            }
                        },
                        "required": ["order_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "classify_request",
                    "description": (
                        "Классифицирует тип обращения клиента: "
                        "complaint (жалоба), question (вопрос), "
                        "request (запрос действия), other."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Текст обращения клиента для классификации"
                            }
                        },
                        "required": ["text"]
                    }
                }
            }
        ]

    @property
    def _system_prompt(self) -> str:
        return """Ты — вежливый и профессиональный ассистент службы поддержки клиентов.

Твои возможности:
- Отвечать на вопросы из FAQ (доставка, возврат, оплата, гарантия, контакты)
- Проверять статус заказов по номеру
- Классифицировать тип обращения

Правила:
1. Всегда используй инструменты для получения актуальных данных — не придумывай информацию.
2. Если клиент спрашивает о заказе — используй get_order_status с его номером.
3. Если клиент задаёт общий вопрос — используй search_faq.
4. Если нужно понять тип обращения — используй classify_request.
5. Отвечай на языке клиента (русский или украинский).
6. Всегда формулируй завершённый текстовый ответ после получения данных."""

    # ─────────────────────────────────────────
    # 5. Валидирующий роутер
    # ─────────────────────────────────────────

    def _execute_tool(self, tool_name: str, raw_args: dict) -> str:
        self.stats["tool_calls"] += 1
        
        try:
            if tool_name == "search_faq":
                args = SearchFAQArgs(**raw_args)
                result = self._search_faq(args.topic)
            elif tool_name == "get_order_status":
                args = GetOrderStatusArgs(**raw_args)
                result = self._get_order_status(args.order_id)
            elif tool_name == "classify_request":
                args = ClassifyRequestArgs(**raw_args)
                result = self._classify_request(args.text)
            else:
                result = {"error": f"Неизвестный инструмент: {tool_name}"}

        except ValidationError as e:
            self.stats["validation_errors"] += 1
            errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
            print(f"  ⚠ ВАЛИДАЦИЯ: {errors}")
            result = {
                "error": "Некорректные аргументы",
                "details": errors,
            }

        return json.dumps(result, ensure_ascii=False)

    # ─────────────────────────────────────────
    # 6. Публичный метод — точка входа
    # ─────────────────────────────────────────

    def handle(self, user_message: str) -> str:
        """
        Обрабатывает обращение клиента.
        Возвращает текстовый ответ.
        """
        self.stats["total_requests"] += 1
        start_time = time.time()

        messages = [{"role": "user", "content": user_message}]

        for iteration in range(1, MAX_ITERATIONS + 1):
            self.stats["total_iterations"] += 1

            payload = {
                "model": self.model,
                "system": self._system_prompt,
                "messages": messages,
                "tools": self._tools,
                "stream": False,
            }

            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            assistant_message = response.json()["message"]
            tool_calls = assistant_message.get("tool_calls")

            if not tool_calls:
                content = assistant_message.get("content", "").strip()

                if not content:
                    self.stats["fallbacks_used"] += 1
                    messages.append({"role": "assistant", "content": ""})
                    messages.append({
                        "role": "user",
                        "content": "Пожалуйста, сформулируй итоговый ответ клиенту."
                    })
                    continue

                elapsed = round(time.time() - start_time, 2)
                print(f"  ✓ Завершено за {iteration} итераций, {elapsed}с")
                return content

            messages.append({"role": "assistant", **assistant_message})

            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]
                print(f"  [tool] {func_name}({func_args})")
                tool_result = self._execute_tool(func_name, func_args)
                messages.append({"role": "tool", "content": tool_result})

        self.stats["total_requests"] -= 1  # не считаем как успешный
        return "[ОШИБКА] Агент не смог сформировать ответ. Обратитесь к оператору."

    def print_stats(self):
        """Выводит статистику работы агента"""
        print("\n" + "="*40)
        print("СТАТИСТИКА АГЕНТА")
        print("="*40)
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        if self.stats["total_requests"] > 0:
            avg = round(
                self.stats["total_iterations"] / self.stats["total_requests"], 1
            )
            print(f"  avg_iterations_per_request: {avg}")


# ─────────────────────────────────────────────
# 7. Тест-кейсы
# ─────────────────────────────────────────────

if __name__ == "__main__":
    agent = CustomerSupportAgent()

    test_cases = [
        # FAQ
        "Какова ваша политика возврата товара?",
        "Как долго идёт доставка?",
        # Заказы
        "Где мой заказ ORD-002?",
        "Скажи статус заказа ORD-003",
        # Несуществующий заказ
        "Что с заказом ORD-999?",
        # Жалоба
        "Я недоволен — товар сломан при доставке!",
        # Неизвестная тема FAQ
        "Расскажи про программу лояльности",
    ]

    for message in test_cases:
        print(f"\n{'='*60}")
        print(f"Клиент: {message}")
        print(f"{'-'*60}")
        answer = agent.handle(message)
        print(f"Агент: {answer}")

    agent.print_stats()