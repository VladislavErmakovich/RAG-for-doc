import time 
from typing import List, Optional
import psutil
import os

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
    section: str
    score: float
    text: str 

class Stats(BaseModel):
    total_time: float
    generate_time: float
    token_cnt: int
    speed_tps: float
    ram_stat_gb: float

class Answer(BaseModel):
    text: str
    source: List[Source]
    stats: Stats

def get_ram_statistic():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 ** 3), 2)

vector_db : Optional[Vector_Data_Base] = None
model: Optional[LLMEngine] = None


@asynccontextmanager
async def life_cycle_app(app: FastAPI):

    print("Запуск работы сервиса...")

    ram_before = get_ram_statistic() # до загрузки
    print(f"RAM до загрузки: {ram_before} GB")

    global vector_db, model

    model = LLMEngine()

    vector_db = Vector_Data_Base()
    vector_db.prepare_and_load_data(flag_rebuild=True)

    ram_after = get_ram_statistic()
    print(f"RAM после загрузки: {ram_after} GB")

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
    
    start_total_time = time.time()

    res_search_data = vector_db.search(request.questtion, n_results=3)

    context = ''
    sources = []

    for items in res_search_data :
        context += f"Фрагемент: Страница {items['page']} | Вес: {items['score']} | Раздел {items['section']} \n"
        context += f"{items['text']}\n\n"

        sources.append(Source(
            page = items['page'],
            section= items['section'],
            score = items['score'],
            text=items['text']
        ))

    start_gen_time = time.time()
    response = model.generate_response(quastion=request, context=context)
    end_gen_time = time.time()

    gen_time = end_gen_time - start_gen_time

    tokens = response['completion_tokens']
    tps = round(tokens / gen_time, 2) if gen_time > 0 else 0.0

    total_time = end_gen_time - start_total_time
    #full_time = round(time.time() - strat_time, 2)

    stats = Stats(
        total_time=round(total_time, 2),
        generate_time= round(gen_time, 2),
        token_cnt= tokens,
        speed_tps= tps,
        ram_stat_gb=get_ram_statistic()
    )

    return Answer(
        text = response['answer'],
        source = sources,
        stats= stats)


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