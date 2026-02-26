# 05_few_shot.py
# Паттерн: Few-Shot Examples
# Идея: вместо длинного описания правил — показать модели примеры
# Один хороший пример заменяет страницу инструкций

from ollama import Client
import json
import time

client = Client(host="http://192.168.0.128:11434")

# Few-shot: system prompt содержит примеры входа/выхода
# Модель "понимает" паттерн и применяет его к новым данным
SYSTEM_FEW_SHOT = """Ты классифицируешь инвойсы. Изучи примеры и применяй тот же паттерн.

<example>
Input: "Office Depot - 50 reams of paper, 3 boxes of pens. Total: $280"
Output: {"category": "office_supplies", "confidence": 0.97, "reasoning": "paper and pens are office supplies"}
</example>

<example>
Input: "AWS Invoice - EC2 instances, S3 storage, data transfer. Total: $4,230"
Output: {"category": "services", "confidence": 0.95, "reasoning": "cloud infrastructure is IT services"}
</example>

<example>
Input: "Kyivenergo - electricity consumption 8,200 kWh. Total: 31,160 UAH"
Output: {"category": "utilities", "confidence": 0.99, "reasoning": "electricity is utilities"}
</example>

Отвечай ТОЛЬКО валидным JSON в том же формате что в примерах."""

def classify_few_shot(invoice_text: str) -> dict:
    response = client.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": SYSTEM_FEW_SHOT},
            {"role": "user", "content": invoice_text}
        ],
        options={"temperature": 0.1}
    )
    raw = response.message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# Тест на пограничных случаях — где классификация неочевидна
edge_cases = [
    {
        "text": "Microsoft 365 Business Premium - 50 licenses. Annual subscription. Total: $6,000",
        "expected": "services",
        "note": "Пограничный: софт = services или equipment?"
    },
    {
        "text": "Lenovo ThinkPad X1 Carbon x5 units. Total: $12,500",
        "expected": "equipment",
        "note": "Чёткий equipment"
    },
    {
        "text": "Курьерська доставка документів, вересень 2024. Сума: 3,400 грн",
        "expected": "services",
        "note": "Украинский, нестандартный кейс"
    }
]

print("Few-Shot классификатор — пограничные случаи\n")

for case in edge_cases:
    print(f"{'='*55}")
    print(f"📋 {case['note']}")
    print(f"   Input: {case['text'][:70]}")
    print(f"   Expected: {case['expected']}")
    
    start = time.time()
    try:
        result = classify_few_shot(case["text"])
        elapsed = time.time() - start
        
        match = "✅" if result.get("category") == case["expected"] else "⚠️ "
        print(f"   Got:      {result.get('category')} {match}")
        print(f"   Reason:   {result.get('reasoning')}")
        print(f"   Time:     {elapsed:.1f}s")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print(f"\n{'='*55}")