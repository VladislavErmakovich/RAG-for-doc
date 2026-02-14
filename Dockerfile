FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \ 
    cmake \ 
    git \ 
    curl \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_OPENBLAS=ON" pip install --no-cache-dir -r requirements.txt

COPY . . 


RUN mkdir -p /app/model /app/data

EXPOSE 8000

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]