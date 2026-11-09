FROM python:3.11-slim

WORKDIR /code

ENV PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10 \
    PIP_NO_CACHE_DIR=1

# Sadece CPU surumu: Mac'teki konteynerde GPU yok, CUDA paketleri gereksiz
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY models ./models

EXPOSE 7001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7001"]
