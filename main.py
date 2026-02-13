import time 
from typing import List, Optional

from vdb import Vector_Data_Base
from LLM import LLMEngine
import LLM

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

import uvicorn


class Question(BaseModel):
    questtion: str

class Source(BaseModel):
    page: int
    score: float
    text: str 

class Answer(BaseModel):
    text: str
    source: List[Source]
    time: float

vector_db : Optional[Vector_Data_Base] = None
model: Optional[LLMEngine] = None


@asynccontextmanager
async def life_cycle_app(app: FastAPI):

    print("Запуск работы сервиса...")

    global vector_db, model

    model = LLMEngine()

    vector_db = Vector_Data_Base()
    vector_db.prepare_and_load_data(flag_rebuild=True)

    yield

    print("Остановка работы сервиса...")

    model = None
    vector_db = None


app = FastAPI(
    title='RAG SERVICE',
    description='API для подачи ответов на вопросы по документации к проекту Chimera',
    version='1.0.0',
    lifespan=life_cycle_app
)

@app.post('/ask', response_model=Answer)
async def ask_question(request: Question):
    if not vector_db and not model:
        raise HTTPException(status_code='503', detail='Сервис не доступен, нет подключения VectorDB или LLM')
    
    strat_time = time()

    res_search_data = vector_db.search(request.questtion, n_results=3)

    context = ''
    sources = []

    for items in res_search_data :
        context += f"Фрагемент: Страница {items['page']} | Вес: {items['score']} \n"
        context += f"{items['text']}\n\n"

        sources.append(Source(
            page = items['page'],
            score = items['score'],
            text=items['text']
        ))


    response = model.generate_response(quastion=request, context=context)

    full_time = round(time() - strat_time, 2)

    return Answer(
        text = response,
        source = sources,
        time = full_time)


@app.get('/health')
def health_chek():
    return {'status': ' ok', 'model': model is not None}




if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host='0.0.0.0',
        port=8000,
        reload=False
    )