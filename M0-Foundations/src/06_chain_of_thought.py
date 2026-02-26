# 06_chain_of_thought.py
# Паттерн: Chain-of-Thought (CoT) — цепочка рассуждений
# Идея: заставить модель думать пошагово ПЕРЕД финальным ответом
# Это повышает точность на сложных и неоднозначных случаях

from ollama import Client
import json
import time

client = Client(host="http://192.168.0.128:11434")

# CoT system prompt: модель сначала рассуждает, потом отвечает
SYSTEM_COT = """Ты — эксперт по классификации корпоративных расходов.

При классификации инвойса ОБЯЗАТЕЛЬНО:
1. Сначала выпиши ключевые признаки из текста
2. Рассмотри возможные категории и почему они подходят/не подходят
3. Сделай финальное решение

Категории: office_supplies, utilities, services, equipment, other

Отвечай строго в формате:
<thinking>
Признаки: ...
Анализ категорий: ...
Решение: ...
</thinking>
<answer>
{"category": "...", "confidence": 0.0, "reasoning": "..."}
</answer>"""

def classify_cot(invoice_text: str) -> dict:
    """Классификация с Chain-of-Thought рассуждением"""
    
    response = client.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": invoice_text}
        ],
        options={"temperature": 0.1}
    )
    
    raw = response.message.content.strip()
    
    # Извлекаем thinking и answer отдельно
    thinking = ""
    answer_json = {}
    
    if "<thinking>" in raw and "</thinking>" in raw:
        thinking = raw.split("<thinking>")[1].split("</thinking>")[0].strip()
    
    if "<answer>" in raw and "</answer>" in raw:
        answer_text = raw.split("<answer>")[1].split("</answer>")[0].strip()
        # Убираем markdown если есть
        if answer_text.startswith("```"):
            answer_text = answer_text.split("```")[1]
            if answer_text.startswith("json"):
                answer_text = answer_text[4:]
        answer_json = json.loads(answer_text.strip())
    
    return {"thinking": thinking, "result": answer_json}


# Тест на ДЕЙСТВИТЕЛЬНО сложных случаях
hard_cases = [
    {
        "text": """Рахунок від: IT Консалтинг Плюс
Послуги: Налаштування та встановлення серверного обладнання Dell PowerEdge R750
Включає: фізичне обладнання + монтаж + налаштування + 1 рік підтримки
Сума: 285,000 грн""",
        "note": "Сложный: оборудование + установка + поддержка в одном счёте"
    },
    {
        "text": """Invoice: Ergonomic office chairs x20 units
Herman Miller Aeron - premium ergonomic seating
Delivery and assembly included
Total: $28,000""",
        "note": "Мебель: equipment или office_supplies?"
    }
]

print("Chain-of-Thought — сложные пограничные случаи\n")

for case in hard_cases:
    print(f"{'='*55}")
    print(f"📋 {case['note']}")
    print(f"   Input: {case['text'][:80]}...")
    print()
    
    start = time.time()
    output = classify_cot(case["text"])
    elapsed = time.time() - start
    
    print(f"💭 Рассуждение модели:")
    # Печатаем thinking с отступом
    for line in output["thinking"].split("\n"):
        if line.strip():
            print(f"   {line.strip()}")
    
    print()
    result = output["result"]
    print(f"✅ Финальный ответ:")
    print(f"   Категория:   {result.get('category')}")
    print(f"   Уверенность: {result.get('confidence')}")
    print(f"   Обоснование: {result.get('reasoning')}")
    print(f"   Время:       {elapsed:.1f}s")
    print()