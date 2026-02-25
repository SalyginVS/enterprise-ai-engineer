# 02_ollama_library.py
# Цель: тот же вызов, но через официальную библиотеку ollama
# Сравни с предыдущим скриптом - код чище, но делает то же самое

from ollama import Client
import time

# Подключаемся к серверу лаборатории
client = Client(host="http://192.168.0.128:11434")

def call_with_library():
    """Вызов через официальную библиотеку ollama"""
    
    print("Отправляю запрос через ollama library...")
    start_time = time.time()
    
    response = client.chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": "Объясни что такое AI Agent в 2-3 предложениях. Отвечай по-русски."
            }
        ]
    )
    
    elapsed = time.time() - start_time
    
    # Доступ к ответу через объект
    print(f"\n--- Ответ модели ---")
    print(response.message.content)
    print(f"\n--- Статистика ---")
    print(f"Время ответа: {elapsed:.1f} сек")

if __name__ == "__main__":
    call_with_library()