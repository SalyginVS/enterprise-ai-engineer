# 03_token_stats.py
# Цель: получить статистику токенов и скорость генерации
# Это основа для будущего FinOps и бенчмаркинга моделей

from ollama import Client
import time

client = Client(host="http://192.168.0.128:11434")

def call_with_stats(model: str, prompt: str):
    """Вызов с полной статистикой токенов"""
    
    print(f"Модель: {model}")
    print(f"Промпт: {prompt[:60]}...")
    print("-" * 50)
    
    start_time = time.time()
    
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    elapsed = time.time() - start_time
    
    # Текст ответа
    print(f"Ответ:\n{response.message.content}")
    print("-" * 50)
    
    # Статистика токенов из ответа
    # eval_count = количество сгенерированных токенов
    # eval_duration = время генерации в наносекундах
    # prompt_eval_count = количество токенов в промпте
    
    eval_count = response.eval_count or 0
    eval_duration_ns = response.eval_duration or 1  # наносекунды
    prompt_tokens = response.prompt_eval_count or 0
    
    # Переводим наносекунды в секунды (1 секунда = 1_000_000_000 нс)
    eval_duration_sec = eval_duration_ns / 1_000_000_000
    
    # Скорость генерации
    tokens_per_second = eval_count / eval_duration_sec if eval_duration_sec > 0 else 0
    
    print(f"📊 Статистика токенов:")
    print(f"  Токенов в промпте:    {prompt_tokens}")
    print(f"  Токенов сгенерировано: {eval_count}")
    print(f"  Итого токенов:        {prompt_tokens + eval_count}")
    print(f"  Скорость генерации:   {tokens_per_second:.1f} tok/s")
    print(f"  Время генерации:      {eval_duration_sec:.2f} сек")
    print(f"  Общее время:          {elapsed:.2f} сек")
    
    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": eval_count,
        "tokens_per_second": tokens_per_second,
        "elapsed": elapsed
    }


def compare_models():
    """Сравнение двух моделей на одном промпте"""
    
    prompt = "Объясни что такое AI Agent в 3 предложениях. Отвечай по-русски."
    
    models = ["qwen3:8b", "deepseek-r1:14b"]
    results = []
    
    for model in models:
        print(f"\n{'='*50}")
        stats = call_with_stats(model, prompt)
        results.append(stats)
        print()
    
    # Итоговое сравнение
    print(f"\n{'='*50}")
    print("📈 Сравнение моделей:")
    print(f"{'Модель':<25} {'tok/s':>8} {'Токенов':>10} {'Время':>8}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['model']:<25} "
            f"{r['tokens_per_second']:>7.1f} "
            f"{r['completion_tokens']:>10} "
            f"{r['elapsed']:>7.1f}s"
        )


if __name__ == "__main__":
    compare_models()