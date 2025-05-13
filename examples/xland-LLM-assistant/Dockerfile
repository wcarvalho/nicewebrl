FROM python:3.10-slim

RUN apt update && apt install -y curl procps git

RUN pip install --upgrade pip

WORKDIR /app

# Explicitly copy things in .gitignore
COPY ./config.py /app/

COPY ./google-cloud-key.json /app/

# Then copy everything else
COPY . /app

RUN pip install "nicewebrl[xland-assistant] @ git+https://github.com/wcarvalho/nicewebrl.git"

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=DEBUG

CMD ["python", "web_app_assistant.py"]