FROM python:3.10-slim

RUN apt update && apt install -y curl procps

RUN pip install --upgrade pip

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=DEBUG

CMD ["python", "web_app.py"]