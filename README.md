# llm_Qwen3-4B-Thinking-2507
## Источник
https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
## Как развернуть
* Склонировать репозиторий
* Создать venv
* Активировать venv, установить зависимости
* python3 app.py
## Как вызывать
```
model_family = 'qwen3'
model_sizes = ['0.6B', '1.7B', '4B', '4B-Instruct', '8B']
system_prompt=''
prompt='Используй не менее 5 метафор из области физики и 3 — из искусства. Объём: 6 реплик'
context='Придумай диалог между учёным и художником о природе времени.'
url = "http://<ip>:5000/api/answer"
for model_size in model_sizes:
  res = requests.post(url, json={'model_family': 'qwen3', 'model_size': model_size, 'system_prompt': system_prompt, 'context': context, 'prompt':'prompt'})
  print(res.json())
```

