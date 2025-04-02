FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV USE_DOCKER=true
ENV PYTHONPATH=/app/src

CMD ["python", "src/app.py"]
