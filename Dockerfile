FROM python:3.11-slim

WORKDIR /app

RUN python3 -m venv .venv

COPY requirements.txt .

RUN . .venv/bin/activate && pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["/app/venv/bin/python3", "main.py"]
