# 01_raw_http.py
# Цель: увидеть "сырой" HTTP запрос к Ollama API
# Это нижний уровень - так работает любой клиент под капотом

import requests
import json
import time

# Адрес сервера лаборатории
OLLAMA_HOST = "http://192.168.0.128:11434"

def first_llm_call():
    """Первый вызов к LLM через сырой HTTP запрос"""
    
    print("Отправляю запрос к модели qwen3:8b...")
    start_time = time.time()
    
    # Формируем запрос
    payload = {
        "model": "qwen3:8b",
        "messages": [
            {
                "role": "user",
                "content": "Объясни что такое AI Agent в 2-3 предложениях. Отвечай по-русски."
            }
        ],
        "stream": False  # False = ждём полный ответ, не стрим
    }
    
    # Отправляем POST запрос
    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json=payload
    )
    
    elapsed = time.time() - start_time
    
    # Проверяем статус ответа
    if response.status_code == 200:
        data = response.json()
        answer = data["message"]["content"]
        
        print(f"\n--- Ответ модели ---")
        print(answer)
        print(f"\n--- Статистика ---")
        print(f"Время ответа: {elapsed:.1f} сек")
        print(f"HTTP статус: {response.status_code}")
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    first_llm_call()