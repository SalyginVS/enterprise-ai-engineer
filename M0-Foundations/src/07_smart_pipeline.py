# 07_smart_pipeline.py
# Финальный скрипт M0: умный pipeline с автовыбором стратегии
# Логика: быстрая классификация → если confidence низкий → CoT
# Это прямой прототип того что будем строить в M1

from ollama import Client
import json
import time
from dataclasses import dataclass
from typing import Optional

client = Client(host="http://192.168.0.128:11434")

# ─── Конфигурация ───────────────────────────────────────
VALID_CATEGORIES = {
    "office_supplies", "utilities", "services", "equipment", "other"
}
CONFIDENCE_THRESHOLD = 0.85  # Ниже этого — escalate на CoT
MODEL = "qwen3:8b"

# ─── Промпты ────────────────────────────────────────────
SYSTEM_FAST = """Классифицируй инвойс. Отвечай ТОЛЬКО валидным JSON:
{"category": "office_supplies|utilities|services|equipment|other",
 "confidence": 0.0-1.0,
 "vendor": "название или null",
 "amount": число или null,
 "reasoning": "краткое обоснование"}"""

SYSTEM_COT = """Ты эксперт по классификации корпоративных расходов.
Категории: office_supplies, utilities, services, equipment, other

Рассуждай пошагово:
<thinking>
Признаки: ...
Анализ: ...
Решение: ...
</thinking>
<answer>
{"category": "...", "confidence": 0.0, "vendor": "...",
 "amount": null, "reasoning": "..."}
</answer>"""

# ─── Dataclass для результата ────────────────────────────
@dataclass
class ClassificationResult:
    category: str
    confidence: float
    vendor: Optional[str]
    amount: Optional[float]
    reasoning: str
    strategy_used: str      # "fast" или "cot"
    tokens_used: int
    elapsed: float
    valid: bool
    error: Optional[str] = None

# ─── Вспомогательные функции ────────────────────────────
def parse_json(text: str) -> dict:
    """Извлечь JSON из текста, убрав markdown-обёртку если есть"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

def validate_result(data: dict) -> tuple[bool, Optional[str]]:
    """Проверить что результат соответствует контракту"""
    if data.get("category") not in VALID_CATEGORIES:
        return False, f"Недопустимая категория: {data.get('category')}"
    conf = data.get("confidence", 0)
    if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
        return False, f"Недопустимый confidence: {conf}"
    return True, None

# ─── Стратегии классификации ─────────────────────────────
def classify_fast(invoice_text: str) -> tuple[dict, int]:
    """Быстрая классификация через System Prompt + JSON"""
    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_FAST},
            {"role": "user", "content": invoice_text}
        ],
        options={"temperature": 0.1}
    )
    tokens = (response.prompt_eval_count or 0) + (response.eval_count or 0)
    return parse_json(response.message.content), tokens

def classify_cot(invoice_text: str) -> tuple[dict, int]:
    """CoT классификация для сложных случаев"""
    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": invoice_text}
        ],
        options={"temperature": 0.1}
    )
    raw = response.message.content
    answer_text = raw.split("<answer>")[1].split("</answer>")[0].strip()
    tokens = (response.prompt_eval_count or 0) + (response.eval_count or 0)
    return parse_json(answer_text), tokens

# ─── Главный pipeline ────────────────────────────────────
def smart_classify(invoice_text: str) -> ClassificationResult:
    """
    Умная классификация с автоматическим выбором стратегии.
    
    Логика:
    1. Попробовать быструю классификацию
    2. Если confidence < threshold или категория невалидна → CoT
    3. Вернуть результат с метаданными
    """
    start = time.time()
    
    # Шаг 1: Быстрая классификация
    try:
        data, tokens = classify_fast(invoice_text)
        valid, error = validate_result(data)
        strategy = "fast"
        
        # Шаг 2: Escalation на CoT если нужно
        needs_escalation = (
            not valid or
            data.get("confidence", 0) < CONFIDENCE_THRESHOLD
        )
        
        if needs_escalation:
            reason = error or f"confidence {data.get('confidence')} < {CONFIDENCE_THRESHOLD}"
            print(f"   ⬆️  Escalating to CoT: {reason}")
            data, tokens = classify_cot(invoice_text)
            valid, error = validate_result(data)
            strategy = "cot"
        
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        # Если fast полностью сломался — сразу CoT
        print(f"   ⚠️  Fast failed ({e}), trying CoT...")
        try:
            data, tokens = classify_cot(invoice_text)
            valid, error = validate_result(data)
            strategy = "cot"
        except Exception as e2:
            elapsed = time.time() - start
            return ClassificationResult(
                category="other", confidence=0.0,
                vendor=None, amount=None,
                reasoning="Classification failed",
                strategy_used="failed", tokens_used=0,
                elapsed=elapsed, valid=False, error=str(e2)
            )
    
    elapsed = time.time() - start
    
    return ClassificationResult(
        category=data.get("category", "other"),
        confidence=data.get("confidence", 0.0),
        vendor=data.get("vendor"),
        amount=data.get("amount"),
        reasoning=data.get("reasoning", ""),
        strategy_used=strategy,
        tokens_used=tokens,
        elapsed=elapsed,
        valid=valid,
        error=error
    )

# ─── Тесты ──────────────────────────────────────────────
test_cases = [
    {
        "text": "Офис-Центр ООО — бумага A4, ручки, степлер. Итого: 1875 грн",
        "note": "Простой — ожидаем fast"
    },
    {
        "text": """IT Консалтинг Плюс: встановлення серверного обладнання
        Dell PowerEdge R750 + монтаж + 1 рік підтримки. 285,000 грн""",
        "note": "Сложный — ожидаем escalation"
    },
    {
        "text": "ДТЕК Одеські мережі — електроенергія грудень. 47,823 грн",
        "note": "Простой utilities"
    },
    {
        "text": "Vendor: XYZ Corp. Item: Miscellaneous supplies. Amount: $50",
        "note": "Низкий confidence — возможен escalation"
    }
]

print("Smart Pipeline — автовыбор стратегии классификации")
print("=" * 55)

total_tokens = 0
results_summary = []

for i, case in enumerate(test_cases, 1):
    print(f"\n[{i}/4] {case['note']}")
    print(f"Input: {case['text'][:65]}...")
    
    result = smart_classify(case["text"])
    total_tokens += result.tokens_used
    results_summary.append(result)
    
    status = "✅" if result.valid else "❌"
    print(f"{status} {result.category} "
          f"(conf: {result.confidence:.2f}, "
          f"strategy: {result.strategy_used}, "
          f"tokens: {result.tokens_used}, "
          f"time: {result.elapsed:.1f}s)")
    print(f"   {result.reasoning[:80]}")

# Итоговая статистика
print(f"\n{'='*55}")
print("📊 Итоговая статистика pipeline:")
fast_count = sum(1 for r in results_summary if r.strategy_used == "fast")
cot_count  = sum(1 for r in results_summary if r.strategy_used == "cot")
valid_count = sum(1 for r in results_summary if r.valid)
total_time = sum(r.elapsed for r in results_summary)

print(f"   Всего инвойсов:    {len(test_cases)}")
print(f"   Fast стратегия:    {fast_count}")
print(f"   CoT escalations:   {cot_count}")
print(f"   Валидных ответов:  {valid_count}/{len(test_cases)}")
print(f"   Всего токенов:     {total_tokens}")
print(f"   Общее время:       {total_time:.1f}s")
print(f"   Avg токенов/запрос:{total_tokens//len(test_cases)}")