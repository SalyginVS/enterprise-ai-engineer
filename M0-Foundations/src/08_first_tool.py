"""
EXP-008: Первый Tool Call через Ollama API
Цель: понять механизм tool calling на минимальном примере.
Модель: qwen3:14b (универсал, поддерживает tools)
"""

import requests
import json

OLLAMA_URL = "http://192.168.0.128:11434/api/chat"
MODEL = "qwen3:14b"

# ─────────────────────────────────────────────
# 1. Определяем инструмент (JSON Schema)
# ─────────────────────────────────────────────
# Это НЕ Python-функция — это описание функции в формате,
# который понятен модели. Модель прочитает это описание
# и поймёт, когда и как её вызывать.

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_invoice_status",
            "description": "Возвращает статус обработки счёта по его номеру",
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
    }
]

# ─────────────────────────────────────────────
# 2. Реализация инструмента (заглушка / mock)
# ─────────────────────────────────────────────
# В реальной системе здесь был бы запрос к БД или ERP.
# Пока — словарь с тестовыми данными.

def get_invoice_status(invoice_id: str) -> dict:
    """Mock-реализация: имитирует обращение к системе учёта"""
    statuses = {
        "INV-2024-001": {"status": "Оплачен", "amount": 15400.00, "date": "2024-01-15"},
        "INV-2024-002": {"status": "Ожидает оплаты", "amount": 8750.50, "date": "2024-01-20"},
        "INV-2024-003": {"status": "Просрочен", "amount": 32100.00, "date": "2023-12-01"},
    }
    return statuses.get(invoice_id, {"status": "Не найден", "invoice_id": invoice_id})

# ─────────────────────────────────────────────
# 3. Роутер инструментов
# ─────────────────────────────────────────────
# Когда модель запрашивает вызов инструмента, мы должны
# сопоставить имя с реальной Python-функцией.

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Диспетчер: по имени вызывает нужную функцию"""
    if tool_name == "get_invoice_status":
        result = get_invoice_status(**tool_args)
        return json.dumps(result, ensure_ascii=False)
    else:
        return json.dumps({"error": f"Неизвестный инструмент: {tool_name}"})

# ─────────────────────────────────────────────
# 4. Вызов модели с инструментами
# ─────────────────────────────────────────────

def ask_with_tools(user_question: str) -> str:
    """
    Полный цикл tool calling:
    1) Отправляем вопрос + описание инструментов
    2) Модель отвечает либо текстом, либо запросом tool call
    3) Если запрошен tool call — выполняем, возвращаем результат
    4) Модель формирует финальный ответ
    """
    
    messages = [
        {"role": "user", "content": user_question}
    ]
    
    print(f"\n{'='*60}")
    print(f"Вопрос: {user_question}")
    print(f"{'='*60}")
    
    # Шаг 1: первый запрос к модели
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False
    }
    
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    
    assistant_message = data["message"]
    print(f"\n[Ответ модели (raw)]")
    print(f"  content: '{assistant_message.get('content', '')}'")
    print(f"  tool_calls: {assistant_message.get('tool_calls', 'нет')}")
    
    # Шаг 2: проверяем, запросила ли модель инструмент
    if not assistant_message.get("tool_calls"):
        # Модель ответила сразу, без tool call
        print("\n[Модель ответила без инструмента]")
        return assistant_message.get("content", "")
    
    # Шаг 3: обрабатываем tool call
    # Добавляем ответ ассистента в историю
    messages.append({"role": "assistant", **assistant_message})
    
    for tool_call in assistant_message["tool_calls"]:
        func_name = tool_call["function"]["name"]
        func_args = tool_call["function"]["arguments"]
        
        print(f"\n[Tool Call обнаружен]")
        print(f"  Инструмент: {func_name}")
        print(f"  Аргументы: {func_args}")
        
        # Выполняем инструмент
        tool_result = execute_tool(func_name, func_args)
        print(f"  Результат: {tool_result}")
        
        # Добавляем результат инструмента в историю сообщений
        messages.append({
            "role": "tool",
            "content": tool_result
        })
    
    # Шаг 4: финальный запрос — модель формирует ответ на основе данных инструмента
    payload["messages"] = messages
    
    final_response = requests.post(OLLAMA_URL, json=payload)
    final_response.raise_for_status()
    final_data = final_response.json()
    
    final_text = final_data["message"].get("content", "")
    print(f"\n[Финальный ответ модели]")
    return final_text


# ─────────────────────────────────────────────
# 5. Тесты
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "Какой статус у счёта INV-2024-001?",
        "Проверь счёт INV-2024-003 — он оплачен?",
        "Что происходит со счётом INV-2024-999?",
    ]
    
    for question in test_questions:
        answer = ask_with_tools(question)
        print(f"\nОтвет: {answer}")
        print()
# ```

# ### Что ожидать в выводе

# Для каждого вопроса ты должен увидеть примерно такой поток:
# ```
# ============================================================
# Вопрос: Какой статус у счёта INV-2024-001?
# ============================================================

# [Ответ модели (raw)]
#   content: ''
#   tool_calls: [{'function': {'name': 'get_invoice_status', 'arguments': {'invoice_id': 'INV-2024-001'}}}]

# [Tool Call обнаружен]
#   Инструмент: get_invoice_status
#   Аргументы: {'invoice_id': 'INV-2024-001'}
#   Результат: {"status": "Оплачен", "amount": 15400.0, "date": "2024-01-15"}

# [Финальный ответ модели]

# Ответ: Счёт INV-2024-001 был оплачен 15 января 2024 года на сумму 15 400 рублей.