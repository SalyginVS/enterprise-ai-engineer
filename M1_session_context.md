# Контекст для сессии M1 — Enterprise AI Agent Engineer
## Все необходимые файлы проекта в одном документе

---

<!-- ============================================================ -->
<!-- FILE: M1_Single_Agent_Engineering.md -->
<!-- ============================================================ -->


# М1: Single-Agent Engineering

**Длительность:** 3-4 недели (24-32 часа)  
**Бизнес-кейс:** Invoice Classifier для бэк-офиса

---

## Цели модуля

**Исходная точка:** Simple agent из М0  
**Переход:** От простого агента к production-ready single agent

**Ключевые навыки:**
- Production-ready agent architecture
- Structured outputs с валидацией (Pydantic)
- Comprehensive error handling
- Retry logic с exponential backoff
- Observability basics (structured logging, metrics)
- Agent state management

---

## Задание 1.1: Production Agent Architecture
**Timeboxing:** 6-8 часов

### Цель:
Спроектировать production-ready архитектуру для Invoice Classifier Agent.

### Компоненты:

**1. Structured Input/Output (Pydantic)**

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum
from datetime import date

class InvoiceCategory(str, Enum):
    OFFICE_SUPPLIES = "office_supplies"
    UTILITIES = "utilities"
    SERVICES = "services"
    EQUIPMENT = "equipment"
    OTHER = "other"

class ClassificationResult(BaseModel):
    """Structured output for classification"""
    category: InvoiceCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    extracted_vendor: Optional[str] = None
    extracted_amount: Optional[float] = None
    
    @validator('confidence')
    def confidence_reasonable(cls, v):
        if v < 0.5:
            raise ValueError("Confidence too low for production use")
        return v

class InvoiceInput(BaseModel):
    """Validated input"""
    invoice_text: str = Field(min_length=10)
    invoice_id: str
    metadata: Optional[dict] = None
```

**2. Agent Core Class**

```python
import anthropic
from typing import Optional
import logging

class InvoiceClassifierAgent:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
    
    def classify(self, invoice: InvoiceInput) -> ClassificationResult:
        """Classify invoice with retries and validation"""
        # Implementation in задание 1.2
        pass
```

### Deliverables:
- ✓ Pydantic models для input/output
- ✓ Agent class skeleton
- ✓ Validation logic
- ✓ Architecture diagram

---

## Задание 1.2: Error Handling & Retry Logic
**Timeboxing:** 6-8 часов

### Реализация:

```python
import time
from typing import Optional
import json

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)

class InvoiceClassifierAgent:
    # ... (previous code)
    
    def classify(self, invoice: InvoiceInput) -> ClassificationResult:
        """Classify with retry logic"""
        
        retry_config = RetryConfig()
        last_error = None
        
        for attempt in range(retry_config.max_retries):
            try:
                return self._classify_attempt(invoice)
            
            except anthropic.RateLimitError as e:
                self.logger.warning(f"Rate limit hit, attempt {attempt + 1}")
                last_error = e
                if attempt < retry_config.max_retries - 1:
                    delay = retry_config.get_delay(attempt)
                    time.sleep(delay)
            
            except anthropic.APIError as e:
                self.logger.error(f"API error: {e}")
                last_error = e
                if attempt < retry_config.max_retries - 1:
                    delay = retry_config.get_delay(attempt)
                    time.sleep(delay)
            
            except Exception as e:
                self.logger.exception(f"Unexpected error: {e}")
                raise
        
        # All retries exhausted
        raise Exception(f"Classification failed after {retry_config.max_retries} attempts") from last_error
    
    def _classify_attempt(self, invoice: InvoiceInput) -> ClassificationResult:
        """Single classification attempt"""
        
        prompt = self._build_prompt(invoice)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response
        result_text = response.content[0].text
        
        # Extract JSON from response
        result_data = self._extract_json(result_text)
        
        # Validate with Pydantic
        result = ClassificationResult(**result_data)
        
        self.logger.info(
            f"Classified invoice {invoice.invoice_id}: "
            f"{result.category} (confidence: {result.confidence})"
        )
        
        return result
    
    def _build_prompt(self, invoice: InvoiceInput) -> str:
        """Build classification prompt"""
        
        return f"""Classify this invoice into one of these categories:
- office_supplies
- utilities  
- services
- equipment
- other

Invoice text:
{invoice.invoice_text}

Respond ONLY with valid JSON in this format:
{{
  "category": "...",
  "confidence": 0.95,
  "reasoning": "...",
  "extracted_vendor": "...",
  "extracted_amount": 123.45
}}

Requirements:
- confidence must be 0.5-1.0
- reasoning must explain the classification
"""
    
    def _extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response"""
        # Handle potential markdown code blocks
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {text}")
            raise ValueError(f"Invalid JSON response") from e
```

### Deliverables:
- ✓ Retry logic с exponential backoff
- ✓ Error handling для API errors
- ✓ Validation с Pydantic
- ✓ JSON parsing robust
- ✓ Logging на each step

---

## Задание 1.3: Observability - Structured Logging & Metrics
**Timeboxing:** 6-8 часов

### Structured Logging:

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(self.StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            
            # Add extra fields
            if hasattr(record, 'invoice_id'):
                log_data['invoice_id'] = record.invoice_id
            if hasattr(record, 'category'):
                log_data['category'] = record.category
            if hasattr(record, 'confidence'):
                log_data['confidence'] = record.confidence
            
            return json.dumps(log_data)
    
    def info(self, message, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def error(self, message, **kwargs):
        self.logger.error(message, extra=kwargs)
```

### Metrics Collection:

```python
from collections import defaultdict
from typing import Dict
import time

class AgentMetrics:
    def __init__(self):
        self.classifications_total = 0
        self.classifications_by_category = defaultdict(int)
        self.errors_total = 0
        self.latencies = []
    
    def record_classification(
        self,
        category: str,
        confidence: float,
        latency_seconds: float
    ):
        self.classifications_total += 1
        self.classifications_by_category[category] += 1
        self.latencies.append(latency_seconds)
    
    def record_error(self, error_type: str):
        self.errors_total += 1
    
    def get_summary(self) -> Dict:
        return {
            "total_classifications": self.classifications_total,
            "by_category": dict(self.classifications_by_category),
            "errors": self.errors_total,
            "avg_latency": sum(self.latencies) / len(self.latencies) if self.latencies else 0,
            "p95_latency": self._percentile(self.latencies, 0.95) if self.latencies else 0,
        }
    
    def _percentile(self, data, p):
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p)
        return sorted_data[index] if sorted_data else 0
```

### Integration:

```python
class InvoiceClassifierAgent:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.logger = StructuredLogger(__name__)
        self.metrics = AgentMetrics()
    
    def classify(self, invoice: InvoiceInput) -> ClassificationResult:
        start_time = time.time()
        
        try:
            self.logger.info(
                "Starting classification",
                invoice_id=invoice.invoice_id
            )
            
            result = self._classify_with_retry(invoice)
            
            latency = time.time() - start_time
            self.metrics.record_classification(
                category=result.category,
                confidence=result.confidence,
                latency_seconds=latency
            )
            
            self.logger.info(
                "Classification successful",
                invoice_id=invoice.invoice_id,
                category=result.category,
                confidence=result.confidence,
                latency=latency
            )
            
            return result
        
        except Exception as e:
            self.metrics.record_error(type(e).__name__)
            self.logger.error(
                "Classification failed",
                invoice_id=invoice.invoice_id,
                error=str(e)
            )
            raise
```

### Deliverables:
- ✓ Structured logging implemented
- ✓ Metrics collection working
- ✓ Latency tracking (avg, p95)
- ✓ Category distribution tracking
- ✓ Error rate tracking

---

## Задание 1.4: Testing & Documentation
**Timeboxing:** 6-8 часов

### Unit Tests:

```python
import pytest
from unittest.mock import Mock, patch

def test_classification_success():
    agent = InvoiceClassifierAgent(api_key="test-key")
    
    invoice = InvoiceInput(
        invoice_id="INV-001",
        invoice_text="Office supplies from Staples: $150"
    )
    
    # Mock API response
    with patch.object(agent.client.messages, 'create') as mock_create:
        mock_create.return_value = Mock(
            content=[Mock(
                text='{"category": "office_supplies", "confidence": 0.95, "reasoning": "..."}'
            )]
        )
        
        result = agent.classify(invoice)
        
        assert result.category == InvoiceCategory.OFFICE_SUPPLIES
        assert result.confidence >= 0.5

def test_retry_on_rate_limit():
    agent = InvoiceClassifierAgent(api_key="test-key")
    
    invoice = InvoiceInput(
        invoice_id="INV-002",
        invoice_text="Test invoice"
    )
    
    with patch.object(agent.client.messages, 'create') as mock_create:
        # Fail twice, succeed third time
        mock_create.side_effect = [
            anthropic.RateLimitError("Rate limit"),
            anthropic.RateLimitError("Rate limit"),
            Mock(content=[Mock(text='{"category": "other", "confidence": 0.8, "reasoning": "test"}')])
        ]
        
        result = agent.classify(invoice)
        
        assert mock_create.call_count == 3
        assert result.category == InvoiceCategory.OTHER
```

### README:

```markdown
# Invoice Classifier Agent

Production-ready agent for classifying invoices.

## Features
- Structured input/output validation
- Automatic retry with exponential backoff
- Structured logging
- Metrics collection
- Comprehensive error handling

## Usage

```python
from invoice_classifier import InvoiceClassifierAgent, InvoiceInput

agent = InvoiceClassifierAgent(api_key="your-api-key")

invoice = InvoiceInput(
    invoice_id="INV-123",
    invoice_text="Your invoice text here"
)

result = agent.classify(invoice)
print(f"Category: {result.category}")
print(f"Confidence: {result.confidence}")
```

## Testing

```bash
pytest test_agent.py -v
```
```

### Deliverables:
- ✓ Unit tests (coverage >70%)
- ✓ Integration test (end-to-end)
- ✓ README с usage examples
- ✓ Code documented (docstrings)

---

## Критерии выхода М1

### Обязательный минимум:
- ✓ Agent architecture production-ready
- ✓ Structured outputs (Pydantic validation)
- ✓ Retry logic working (exponential backoff)
- ✓ Error handling comprehensive
- ✓ Structured logging implemented
- ✓ Basic metrics collection
- ✓ Tests (coverage >70%)
- ✓ Documentation complete

### Сильный уровень:
- ✓ Test coverage >85%
- ✓ Performance metrics (latency p95, p99)
- ✓ Advanced validation rules
- ✓ Circuit breaker pattern (опционально)
- ✓ Prometheus metrics export (опционально)

---

## Triumvirate Usage

**Perplexity:**
- "Python structured logging best practices 2026"
- "Pydantic validation patterns"
- "Exponential backoff retry strategies"

**Claude:**
- Architecture design review
- Code generation для boilerplate
- Prompt optimization

**Gemini:**
- Code review (error handling gaps?)
- Alternative retry strategies
- Testing strategy critique

---

**Следующий модуль:** M2_Observability_Safety.md
-e 

---

<!-- FILE: M2_Observability_Safety.md -->


# М2: Observability & Safety

**Длительность:** 3-4 недели (24-32 часа)  
**Бизнес-кейс:** Расширение Invoice Classifier из М1

---

## Цели модуля

**Исходная точка:** Production-ready single agent из М1  
**Переход:** От базового агента к observable & safe agent

**Ключевые навыки:**
- Distributed tracing (Jaeger)
- Metrics (Prometheus) & Dashboards (Grafana)
- Human-in-the-loop patterns
- Content filtering & safety guardrails
- Anomaly detection

---

## Задания (краткий обзор)

### 2.1: Distributed Tracing
- OpenTelemetry integration
- Trace ID propagation
- Jaeger setup
- Span instrumentation

### 2.2: Prometheus Metrics & Grafana
- Prometheus client library
- Custom metrics (counters, gauges, histograms)
- Grafana dashboards
- Alerts

### 2.3: Human-in-the-Loop
- Approval workflows
- Confidence thresholds
- Queue management
- Feedback collection

### 2.4: Content Filtering & Safety
- Input validation (jailbreak detection)
- Output filtering
- Toxicity detection
- PII detection basics

---

## Критерии выхода М2

### Обязательный минимум:
- ✓ Distributed tracing working (Jaeger)
- ✓ Prometheus metrics exported
- ✓ Grafana dashboard (минимум 5 panels)
- ✓ Human-in-the-loop для low confidence
- ✓ Basic content filtering implemented
- ✓ Tests для safety features

### Сильный уровень:
- ✓ Advanced anomaly detection
- ✓ Automated safety testing
- ✓ Multi-level approval workflows
- ✓ Comprehensive safety coverage

---

**Детали:** См. transcript `/mnt/transcripts/2026-02-01-08-47-54-enterprise-ai-agent-training-m0-m4-detailed.txt`

**Следующий модуль:** M3_Multi_Agent_Systems.md
-e 

---

<!-- FILE: README.md -->


# Enterprise AI Agent Engineer: 9-Month Training Program
## Version 1.1 (Updated after Gemini Review)

**Цель:** Подготовить Enterprise AI Agent Engineer за 6-9 месяцев из мотивированного инженера с 20+ лет ИТ-опыта (без глубокого ML/DS background).

**Длительность:** 9 месяцев (280-360 часов)  
**Нагрузка:** 8-12 часов/неделю  
**Формат:** Практический, project-based learning с обязательным red-teaming  
**Уровень выпускника:** Senior Enterprise AI Agent Engineer

---

## Структура программы

| Модуль | Название | Длительность | Фокус |
|--------|----------|--------------|-------|
| **M0** | Foundations & Mental Models | **1 неделя** | Quick boot-camp, LLM basics |
| **M1** | Single-Agent Engineering | 3-4 недели | Production-ready single agents |
| **M2** | Observability, Safety & Red-Teaming | 3-4 недели | Monitoring, **red-teaming**, fail-safe |
| **M3** | Multi-Agent Systems | 4-5 недель | Agent coordination, **timeouts** |
| **M4** | Orchestration & Workflows | 5-6 недель | State machines, **learning orchestrator** |
| **M5** | Enterprise Integration & Security | 5-6 недель | APIs, **PII anonymization**, **FinOps** |
| **M6** | Capstone Project | 6-8 недель | End-to-end + **AI Evals** |
| **M7** | Production Readiness & Ops | 3-4 недели | CI/CD, **semantic drift** procedures |

**Total:** 30-39 недель (7-9 месяцев)

---

## Ключевые обновления (v1.1)

### 🔴 Критичные добавления безопасности:

**1. Red-Teaming как обязательная практика (М2):**
- Инженер атакует собственного агента (prompt injection, HITL bypass, rate limits)
- Документирование успешных атак и mitigations
- Минимум 5-7 задокументированных векторов атак

**2. PII Anonymization Layer (М5):**
- Обязательный слой маскировки PII перед LLM
- PII не попадает в prompts, логи, traces
- Токенизация/псевдонимизация чувствительных данных

**3. Indirect Prompt Injection защита (М5):**
- Защита от атак через данные (PDF, базы)
- Input sanitization, allow-list команд
- Разделение данных/инструкций

### 🛡️ Production Safety Features:

**4. Fail-Safe на "мусор от модели" (М2):**
- Обработка invalid JSON, неожиданных структур
- Контролируемый выход без бесконечных retry
- Safe fallback states

**5. Global Timeouts & Max Loops (М3-М4):**
- Per-agent timeout и max steps
- Global workflow timeout
- Защита от "застревания" агентов

**6. AI Evals для Semantic Quality (М6):**
- Автоматизированная оценка семантического качества
- Сравнение версий (до/после изменений)
- DeepEval / RAGAS / LLM-as-a-judge

**7. Semantic Drift Procedures (М7):**
- Runbook для переоценки при обновлениях LLM
- Контролируемые изменения с eval verification
- Процедура rollback/accept/refine

### 💰 Business-Oriented Additions:

**8. FinOps & Unit-Экономика (М5):**
- Расчет стоимости обработки на запрос
- Сравнение моделей по цене/качеству
- Метрика "стоимость за успешную обработку"

**9. Local LLM Fallback (М5):**
- Автономность при потере облачного доступа
- vLLM / Ollama как fallback
- Graceful degradation качества

---

## Triumvirate Approach (Обновленные роли)

- **Perplexity** — Pattern Researcher + **FinOps Analyst**
  - Актуальные паттерны, фреймворки
  - **Тарифы LLM API, модели, цена/качество**
  
- **Claude** — System Builder & Mentor + **Architect**
  - Дизайн, код, обучение
  - **ADR (Architecture Decision Records) generation**
  - Обоснование декомпозиции и паттернов

- **Gemini** — Critic & Red Team + **Security Auditor**
  - Код-ревью, альтернативы
  - **Усиленный red-team (security, prompt injection)**
  - **AI Evals validation**
  - Мультимодальная критика (SCADA, схемы)

**Детали:** См. `Engineer_AI_Triumvirate_Constitution.md` и `Engineer_Triumvirate_Quick_Reference.md`

---

## Checkpoint Reviews

### Checkpoint #1 (М0-М2) — Week 11-12
- **Новый фокус:** Red-teaming skills demonstrated
- **Критерии:** Fail-safe mechanisms работают

### Checkpoint #2 (М3-М5) — Week 26-27
- **Новый фокус:** Timeouts/loops защита, PII anonymization, FinOps awareness
- **Критерии:** Security posture включает indirect prompt injection защиту

### Checkpoint #3 (Capstone) — Week 38-39
- **Новый фокус:** AI Evals implemented и демонстрированы
- **Критерии:** Semantic quality measurable

### Checkpoint #4 (Final Defense) — Week 42-43
- **Новый фокус:** Semantic drift procedures documented
- **Критерии:** Operational runbooks включают LLM update procedures

---

## Performance & Quality Targets

### M4-M7: Production Targets

**Performance:**
- End-to-End Latency (p95): <30s
- Throughput: >20 tasks/hour
- Success Rate: >90%

**Security (обновлено):**
- **PII Anonymization: 100% coverage (НОВОЕ)**
- **PII Detection Rate: >95%**
- **Red-Team Attacks Documented: ≥5 (НОВОЕ)**
- Security Score: Zero high/critical vulnerabilities
- Secrets Rotation: 100% automated

**AI Quality (НОВОЕ):**
- **Eval Pipeline: Automated**
- **Semantic Quality: Measurable & tracked**
- **Model Update Process: Documented & tested**

---

## Детальная документация

- `M0_Foundations.md` — **обновлено (1 неделя boot-camp)**
- `M1_Single_Agent_Engineering.md`
- `M2_Observability_Safety.md` — **обновлено (red-teaming, fail-safe)**
- `M3_Multi_Agent_Systems.md` — **обновлено (timeouts, max loops)**
- `M4_Orchestration_Workflows.md` — **обновлено (learning orchestrator)**
- `M5_Enterprise_Integration_Security.md` — **обновлено (PII, FinOps, local LLM)**
- `M6_Capstone_Project.md` — **обновлено (AI Evals)**
- `M7_Production_Readiness.md` — **обновлено (semantic drift)**
- `Checkpoint_Reviews.md` — **обновлено (новые критерии)**
- `Evaluation_Criteria.md` — **обновлено (security & AI quality)**

---

## Ключевые принципы программы (обновлено)

1. **Практический фокус:** Код, системы, симуляции
2. **Production-first mindset:** Все требования production-level
3. **Security-first:** Red-teaming, PII protection, injection защита
4. **Measurable targets:** Performance, security, **AI quality**
5. **Business awareness:** FinOps, unit-экономика, cost-quality
6. **Operational excellence:** Semantic drift procedures, model updates
7. **Triumvirate approach:** Three AI perspectives + FinOps + red-team
8. **High exit standards:** Качество важнее скорости

---

## Версия программы

**Version:** 1.1 (Updated after Gemini Review)  
**Date:** 2026-02-01  
**Status:** Готова к использованию

**Changelog:**

**v1.1 (2026-02-01):**
- **SECURITY:** Обязательный red-teaming (М2)
- **SECURITY:** PII Anonymization Layer (М5)
- **SECURITY:** Indirect Prompt Injection защита (М5)
- **SAFETY:** Fail-safe на invalid output (М2)
- **SAFETY:** Global timeouts & max loops (М3-М4)
- **QUALITY:** AI Evals обязательный блок (М6)
- **OPERATIONS:** Semantic Drift procedures (М7)
- **BUSINESS:** FinOps & unit-экономика (М5)
- **RESILIENCE:** Local LLM fallback (М5)
- **POSITIONING:** М0 как 1-week boot-camp
- **POSITIONING:** Кастомный оркестратор как learning tool (М4)
- **TRIUMVIRATE:** Обновлены роли (FinOps, red-team, ADR)

---

**Удачи в обучении! От теории к production-ready, secure, cost-aware AI systems за 9 месяцев.**
-e 

---

<!-- FILE: Evaluation_Criteria.md -->


# Evaluation Criteria & Grading System
## Version 1.1 (Updated after Gemini Review)

---

## Критерии по категориям (ОБНОВЛЕНО)

### Technical Skills (35%)

**Добавлено для Senior (85-94%):**
- **Red-teaming demonstrated** (≥5 attacks documented)
- **AI Evals pipeline implemented**
- Security-conscious design (PII, injection)

### Production Readiness (25%)

**Добавлено для Senior:**
- **Semantic drift runbook documented & tested**
- Model update procedures established

### Security & Compliance (20%)

**НОВЫЕ метрики:**
- **PII Anonymization: 100% coverage**
- **PII Detection Rate: >95%**
- **Red-Team Attacks: ≥5 documented & mitigated**
- **Indirect Injection: Protected**

**Senior (85-94%):**
- **All PII anonymized before LLM**
- **Red-team comprehensive** (all attacks mitigated)
- **Indirect injection защита** working

### AI Quality (НОВАЯ категория) - включено в Technical Skills

**Метрики:**
- **Eval Pipeline: Automated**
- **Semantic Quality: Measurable**
- **Before/After Comparison: Demonstrated**

**Senior (85-94%):**
- Eval pipeline runs automatically
- Semantic quality tracked
- Can demonstrate impact of changes

### Business Awareness (включено в evaluation)

**FinOps:**
- Cost per request calculated
- Model comparison by price/quality
- **Senior: Can justify model choice economically**

---

## Checkpoint-Specific Criteria (ОБНОВЛЕНО)

### Checkpoint #1 (М0-М2)

**Target:** Middle+ (75%+)  
**NEW Focus:**
- Red-teaming skills
- Fail-safe mechanisms

**PASS threshold:** 60%  
**+ Must have:**
- ✓ Red-team report (≥5 attacks)
- ✓ Fail-safe working

### Checkpoint #2 (М3-М5)

**Target:** Senior (85%+)  
**NEW Focus:**
- Timeout protection
- PII anonymization
- FinOps awareness

**PASS threshold:** 70%  
**+ Must have:**
- ✓ Timeouts/loops protection
- ✓ PII anonymized (100%)
- ✓ FinOps metrics

### Checkpoint #3 (Capstone)

**NEW Focus:**
- AI Evals demonstrated

**+ Must have:**
- ✓ Eval pipeline working
- ✓ Semantic quality measurable

### Checkpoint #4 (Final)

**NEW Focus:**
- Semantic drift procedures

**+ Must have:**
- ✓ Drift runbook documented
- ✓ Model update process tested

---

## Success Metrics (Program-Level) - UPDATED

**Quality Gates:**
- Checkpoint #1 PASS rate: >80%
- **+ Red-team completion rate: 100%**
- Checkpoint #2 PASS rate: >75%
- **+ PII anonymization coverage: 100%**
- Capstone PASS rate: >70%
- **+ AI Evals implementation: 100%**
- Final Defense Senior+ rate: >70%

---

**v1.1:** Security-first, AI quality measurable, business-aware evaluation
-e 

---

<!-- FILE: Паспорт_Лаборатории_на_старте_1.md -->


Паспорт стенда лаборатории на момент старта Кураса . Дата:21.02.2026

## 1. Общая информация

- Название: LLM‑лаборатория `llm` (home AI lab).  
- Назначение: локальный сервер для экспериментов с LLM (Large Language Model — большая языковая модель), мультимодальными моделями, RAG (Retrieval-Augmented Generation — генерация с дополнением поиском) и автоматизацией. [dev](https://dev.to/best_codes/qwen-3-benchmarks-comparisons-model-specifications-and-more-4hoa)
- Роль в инфраструктуре: стенд для пилотов и прототипов, потенциальный прообраз он‑прем (on‑premises — размещённый на собственной площадке) LLM‑сервера холдинга.

## 2. Аппаратная платформа

- Сервер: лабораторный ПК (desktop‑класс).  
- CPU (Central Processing Unit — центральный процессор): Intel Core i9‑9900KF (8C/16T), буст до 5 ГГц.
- RAM (Random Access Memory — оперативная память): 32 ГБ DDR4.
- GPU (Graphics Processing Unit — графический процессор): NVIDIA GeForce RTX 3090, 24 ГБ VRAM (Video RAM — видеопамять).
- Хранилище:  
  - SSD под систему и модели: ~1 ТБ, файловая система ext4.  
  - Свободное место под модели и данные: ~880 ГБ (порядок).
- Сетевое окружение:  
  - Локально: гигабитный Ethernet.  
  - План: доступ из внешней сети через MikroTik hAP ax2, статический внешний IP, проброс портов/Reverse Proxy (обратный прокси‑сервер).

## 3. Программная платформа

- ОС (операционная система): Ubuntu Server 24.04 LTS, headless (без графического интерфейса).  
- Файловая система: ext4 на корневом разделе `/`, размер ~937 ГБ, свободно ~879 ГБ (по результатам `lsblk`/`df`).  
- Драйвер NVIDIA: серия 59x, поддержка CUDA (Compute Unified Device Architecture — программная платформа NVIDIA для вычислений на GPU), VRAM 24 ГБ для ИИ‑нагрузок.
- Docker: установлен Docker Engine версии 28.x, поддержка `--gpus all` для контейнеров с доступом к GPU.
- Службы мониторинга (потенциал):  
  - `nvidia-smi` для мониторинга GPU.  
  - Системные логи через `journalctl`, `systemd`‑юнит Ollama.

## 4. Стек Ollama и модели

### Ollama

- Версия: 0.16.2 (современная ветка с поддержкой Qwen3, DeepSeek‑R1, vision‑моделей). [github](https://github.com/ollama/ollama/blob/main/docs/api.md?plain=1)
- Запуск: в виде systemd‑сервиса, API слушает `127.0.0.1:11434` (локальный HTTP‑endpoint для REST API). [ollama.readthedocs](https://ollama.readthedocs.io/en/api/)
- Основной протокол:  
  - `POST /api/generate` — потоковая (streaming) и нестриминговая генерация, параметры `model`, `prompt`, `images` и др. [github](https://github.com/ollama/ollama/blob/main/docs/api.md?plain=1)
  - `POST /api/chat` — чатовый режим с историей сообщений. [ollama.readthedocs](https://ollama.readthedocs.io/en/api/)

### Набор установленных моделей и их роли

| Роль                       | Модель (Ollama tag)      | Примерный размер | Тип | Назначение |
|---------------------------|--------------------------|------------------|-----|-----------|
| Быстрый универсал         | `qwen3:8b`               | ~5.2 ГБ          | Text | Ежедневный чат, быстрые ответы, лёгкий код. [ollama](https://ollama.com/library/qwen3) |
| Research / аналитика      | `qwen3:14b`              | ~9.3 ГБ          | Text | Глубокие объяснения, анализ документов, RAG‑контекст. [ollama](https://ollama.com/library/qwen3) |
| Основной кодер            | `deepseek-coder-v2:16b`  | ~8.9 ГБ          | Code | Генерация и рефакторинг кода, написание скриптов и сервисов. [github](https://github.com/ollama/ollama/blob/main/docs/api.md?plain=1) |
| Reasoning‑специалист      | `deepseek-r1:14b`        | ~9.0 ГБ          | Text+Thinking | Пошаговое рассуждение, сложные логические задачи, chain‑of‑thought. [github](https://github.com/ollama/ollama/blob/main/docs/api.md?plain=1) |
| Тяжёлый кодер             | `qwen3-coder:30b`        | ~18 ГБ           | Code | Максимальное качество по коду, отдельный запуск без конкурентов. [ollama](https://ollama.com/library/qwen3) |
| Мультимодальный ассистент | `qwen3-vl:8b`            | ~6.1 ГБ          | Vision+Text | Анализ изображений, OCR (Optical Character Recognition — оптическое распознавание текста), описания скриншотов. [ollama](https://ollama.com/library/qwen3-vl) |

Примечание: одновременно, исходя из VRAM 24 ГБ, разумно держать одну «тяжёлую» модель; одновременный запуск нескольких 14–30B ограничен памятью и общим теплопакетом GPU. [apatero](https://apatero.com/blog/ollama-qwen-3-vl-models-local-guide-2025)

## 5. Проверка работоспособности моделей

### 5.1. Базовая проверка текстовых моделей

- Команда:  
  - `ollama run qwen3:8b "Кратко представься и опиши свои сильные стороны в 2–3 предложениях."`  
  - Аналогичные тесты для `qwen3:14b`, `deepseek-coder-v2:16b`, `deepseek-r1:14b`, `qwen3-coder:30b`.  
- Критерии успеха:  
  - Модель отвечает без таймаута и ошибок, даёт осмысленный текст.  
  - Для DeepSeek‑R1 виден блок thinking (блок рассуждения) при включённой опции `think`. [github](https://github.com/ollama/ollama/blob/main/docs/api.md?plain=1)

### 5.2. Проверка мультимодальной модели `qwen3-vl:8b`

- Используемый API: `POST /api/generate` с параметром `images` — список base64‑строк без префикса. [ollama.readthedocs](https://ollama.readthedocs.io/en/api/)
- Подготовка:  

  ```bash
  base64 -w0 /home/vladimir/tests/test.png > /home/vladimir/tests/test.b64
  ```

- Вызов:

  ```bash
  curl -s http://127.0.0.1:11434/api/generate \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg img "$(cat /home/vladimir/tests/test.b64)" '{
      model: "qwen3-vl:8b",
      prompt: "Опиши, что изображено на картинке, в 2–3 предложениях.",
      images: [$img],
      stream: false
    }')" | jq .
  ```

- Полученный результат:  
  - `done: true`, `done_reason: "stop"`.  
  - Поле `response`:

    > «На картинке изображен смартфон с онлайн-тестом, где вариант B отмечен галочкой. Рядом расположены желтый карандаш, очки и секундомер, символизирующие учебный процесс и контроль времени.»

  - Поле `thinking` показывает детальный разбор визуальных элементов: надпись «Online Test», варианты A–D, выбран B, перечисление карандаша, очков, секундомера и проверку количества предложений. [ollama](https://ollama.com/blog/qwen3-vl)

- Интерпретация:  
  - Модель корректно получила изображение (через base64 в `images`).  
  - Распознала структуру интерфейса (смартфон с онлайн‑тестом), текст (варианты ответов), предметы вокруг (карандаш, очки, секундомер). [ollama](https://ollama.com/library/qwen3-vl)
  - Дала связное и точное описание — стенд мультимодального анализа считается полностью рабочим. [ollama](https://ollama.com/blog/qwen3-vl)

## 6. Типовая эксплуатация

### 6.1. Примеры запросов к API

- Текстовый запрос:

  ```bash
  curl -s http://127.0.0.1:11434/api/generate \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3:8b",
      "prompt": "Сформулируй политику использования LLM-сервера для внутреннего ИТ-отдела в 5 пунктах.",
      "stream": false
    }'
  ```

- Чатовый запрос:

  ```bash
  curl -s http://127.0.0.1:11434/api/chat \
    -H "Content-Type: application/json" \
    -d '{
      "model": "deepseek-r1:14b",
      "messages": [
        {"role": "system", "content": "Ты помощник по архитектуре корпоративных ИТ-систем."},
        {"role": "user", "content": "Предложи схему использования локального LLM-сервера для RAG-помощника Service Desk."}
      ],
      "stream": false
    }'
  ```
 [github](https://github.com/ollama/ollama/blob/main/docs/api.md?plain=1)

- Мультимодальный запрос (см. раздел 5.2) для картинок. [ollama.readthedocs](https://ollama.readthedocs.io/en/api/)

### 6.2. Операционные рекомендации

- Не запускать одновременно больше одной тяжёлой модели (14–30B) при активной нагрузке, чтобы избежать свопа VRAM/падения скорости. [apatero](https://apatero.com/blog/ollama-qwen-3-vl-models-local-guide-2025)
- Использовать квантованные (quantized — с пониженной разрядностью весов) варианты при необходимости уменьшить нагрузку на GPU (`qwen3:8b-q4_0`, и т.п.). [apatero](https://apatero.com/blog/ollama-qwen-3-vl-models-local-guide-2025)
- Периодически проверять `nvidia-smi`, `ollama ps`, `journalctl -u ollama` для контроля. [apatero](https://apatero.com/blog/ollama-qwen-3-vl-models-local-guide-2025)

## 7. Возможные дальнейшие шаги

- Добавить `qwen3-embedding` как специализированную модель для эмбеддингов (vector embeddings — векторные представления текста) в RAG‑контуре. [ollama](https://ollama.com/library/qwen3-embedding)
- Поверх этого сервера — поднять лёгкий API‑шлюз (FastAPI/Node.js), который:  
  - маршрутизирует запросы по ролям (чат, код, reasoning, vision);  
  - ведёт аудит (кто/когда/с чем обращался к моделям);  
  - ограничивает длину промптов и ответов.  
- Формализовать SLA (Service Level Agreement — соглашение об уровне сервиса) стенда: макс. время ответа, допустимая конкуренция запросов, правила обновления моделей.

