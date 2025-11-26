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
system_prompt=''
prompt='найди книги про котиков, магию и инопланетян'
context='уточнение показывай только книги авторов женщин'
url = "http://<ip>:5000/api/answer"
res = requests.post(url, json={'system_prompt': system_prompt, 'context': context, 'prompt':'prompt'})
print(res.json())
```

