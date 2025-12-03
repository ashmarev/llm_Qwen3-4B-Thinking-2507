import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# Define query class
class ArticleRequest(BaseModel):
    model_family: str
    model_size: str
    model_type: str
    system_prompt: str
    prompt: str
    context: str

# Load the tokenizer and model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'Qwen/Qwen3-0.6B'
if model_family == 'qwen3':
    if model_size == '0.6B':
        model_name = 'Qwen/Qwen3-0.6B'
    elif model_size == '1.7B':    
        model_name = 'Qwen/Qwen3-1.7B'
    elif model_size == '4B':
        model_name = 'Qwen/Qwen3-4B'
    elif model_size == '4B-Instruct':
        model_name = 'Qwen/Qwen3-4B-Instruct-2507'
    elif model_size == '8B':
        model_name = 'Qwen/Qwen3-8B'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Answer generation function
def generate_llm_answer(request: ArticleRequest):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    start_ts = datetime.now().timestamp()

    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # Form system prompt
    if not request.system_prompt:
        system_prompt = '''Ты — умный и вежливый ассистент.
        Отвечай подробно, но без лишней воды.
        Если не знаешь ответа — скажи прямо.
        Говори на русском языке.'''
    else:
        system_prompt = request.system_prompt

    # Build chat messages
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Add context if provided
    if request.context:
        messages.append({
            "role": "user",
            "content": f"Контекст для ответа:\n\n{request.context}"
        })

    # Add user prompt
    messages.append({
        "role": "user",
        "content": request.prompt
    })

    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка шаблонизации: {e}")

    # Tokenize input
    try:
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка токенизации: {e}")

    # Generate response
    try:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768, # на слабых GPU начинаем с 1024
            do_sample=True,          # включение вероятностной генерации
            temperature=0.7,     # температура
            top_p=0.9,         # фильтрация маловероятных токенов
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {e}")

    # Parse thinking content (if used)
    THINK_TOKEN_ID = 151668
    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_TOKEN_ID)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], 
        skip_special_tokens=True
    ).strip("\n")
    
    content = tokenizer.decode(
        output_ids[index:],
        skip_special_tokens=True
    ).strip("\n")

    end_ts = datetime.now().timestamp()

    # Prepare response data
    data = {
        'duration_time': {
            'start': start_ts,
            'end': end_ts,
            'total_seconds': end_ts - start_ts
        },
        'device': device,
        'messages': messages,
        'thinking_content': thinking_content,
        'content': content
    }
    return data

# API
if __name__ == '__main__':
  # FastAPI app init
  app = FastAPI()
  
  @app.get('/api')
  async def get_pages():
    return [
      {"name": "Перечень страниц", "url": "/api"},
      {"name": "Проверка работоспособности", "url": "/health"},
      {"name": "Вызов LLM", "url": "/api/answer"},
    ]

  @app.get('/health')
  async def health_check():
    return {"status": "ok", "model_loaded": bool(model)}

  @app.post('/api/answer')
  async def generate_answer(request: ArticleRequest):
    result = generate_llm_answer(request)
    return result

  # launch
  import uvicorn
  uvicorn.run(app, host='0.0.0.0', port=5000)
