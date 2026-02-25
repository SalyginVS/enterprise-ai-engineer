# 04_system_prompt.py
# Паттерн: System Prompt + Structured JSON Output
# Это основа любого production агента — модель получает роль
# через system prompt и возвращает предсказуемый JSON

from ollama import Client
import json
import time

client = Client(host="http://192.168.0.128:11434")

# System prompt задаёт роль и правила ОДИН РАЗ
# Все последующие сообщения от user интерпретируются в этом контексте
SYSTEM_PROMPT = """Ты — классификатор инвойсов (счетов-фактур) для корпоративной бухгалтерии.

Твоя задача: проанализировать текст инвойса и вернуть классификацию.

Категории:
- office_supplies (канцелярия, бумага, ручки)
- utilities (электроэнергия, вода, интернет)  
- services (консалтинг, IT-услуги, обслуживание)
- equipment (оборудование, техника, мебель)
- other (всё остальное)

КРИТИЧНО: отвечай ТОЛЬКО валидным JSON без какого-либо текста до или после.
Формат ответа:
{
  "category": "одна из категорий выше",
  "confidence": 0.95,
  "vendor": "название поставщика или null",
  "amount": 1234.56 или null,
  "reasoning": "краткое обоснование классификации"
}"""

def classify_invoice(invoice_text: str) -> dict:
    """Классифицировать инвойс и вернуть структурированный результат"""
    
    response = client.chat(
        model="qwen3:8b",
        messages=[
            {"role": "user", "content": invoice_text}
        ],
        options={"temperature": 0.1}  # Низкая температура = более предсказуемый результат
    )
    
    # Пытаемся распарсить JSON
    raw_text = response.message.content.strip()
    
    # Убираем возможные markdown-блоки ```json ... ```
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    
    result = json.loads(raw_text)
    return result

# Тестовые инвойсы
test_invoices = [
    """СЧЁТ-ФАКТУРА №INV-2024-001
Поставщик: Офис-Центр ООО
Дата: 15.01.2024
Позиции:
- Бумага A4 (5 пачек) — 1250 грн
- Ручки шариковые (20 шт) — 380 грн
- Степлер — 245 грн
ИТОГО: 1875 грн""",

    """Invoice #SRV-789
Vendor: TechSupport Pro
Date: 2024-01-20
Services rendered: Monthly IT infrastructure maintenance
Server monitoring and administration — $2,400
Total: $2,400 USD""",

    """Рахунок №ЕЛ-2024-0123
Постачальник: ДТЕК Одеські мережі
Послуга: Електроенергія за грудень 2023
Спожито: 12,450 кВт·год
Сума до сплати: 47,823.45 грн"""
]

print("Тестирую классификатор инвойсов...\n")

for i, invoice in enumerate(test_invoices, 1):
    print(f"{'='*55}")
    print(f"Инвойс #{i}:")
    print(invoice[:80] + "...")
    print()
    
    start = time.time()
    try:
        result = classify_invoice(invoice)
        elapsed = time.time() - start
        
        print(f"✅ Результат:")
        print(f"   Категория:   {result.get('category')}")
        print(f"   Уверенность: {result.get('confidence')}")
        print(f"   Поставщик:   {result.get('vendor')}")
        print(f"   Сумма:       {result.get('amount')}")
        print(f"   Обоснование: {result.get('reasoning')}")
        print(f"   Время:       {elapsed:.1f}s")
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка парсинга JSON: {e}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

print(f"\n{'='*55}")