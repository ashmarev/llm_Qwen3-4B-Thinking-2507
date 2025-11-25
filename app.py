from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Описание класса запроса
class ArticleRequest(BaseModel):
    prompt: str

# Инициализация FastAPI
app = FastAPI()

# Имя модели
MODEL_NAME = 'Qwen/Qwen3-4B-Thinking-2507'

# Загрузка токенайзера и модели
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {e}")

@app.get('/api')
async def get_pages():
    return [
        {"name": "Перечень страниц", "url": "/api"},
        {"name": "Проверка работоспособности", "url": "/health"},
        {"name": "Вызов LLM", "url": "/api/answer"}
    ]

@app.get('/health')
async def health_check():
    return {"status": "ok", "model_loaded": bool(model)}

@app.post('/api/answer')
async def generate_answer(request: ArticleRequest):
    # Проверка наличия промпта
    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    prompt = request.prompt
    messages = [{"role": "user", "content": prompt}]

    # Подготовка ввода для модели
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка подготовки ввода: {e}")

    # Генерация ответа
    try:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {e}")

    # Поиск токена </think> (ID 151668)
    THINK_TOKEN_ID = 151668
    try:
        # Ищем последний вхождение токена (с конца)
        index = len(output_ids) - output_ids[::-1].index(THINK_TOKEN_ID)
    except ValueError:
        index = 0  # Если токен не найден

    # Декодирование частей ответа
    try:
        thinking_content = tokenizer.decode(
            output_ids[:index],
            skip_special_tokens=True
        ).strip("\n")
        content = tokenizer.decode(
            output_ids[index:],
            skip_special_tokens=True
        ).strip("\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка декодирования: {e}")

    # Формирование ответа
    data = {
        "thinking_content": thinking_content,
        "content": content
    }
    return data

# Запуск сервера
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
