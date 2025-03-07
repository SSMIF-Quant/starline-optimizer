FROM python:3.13-slim

# C compiler is required for convex solvers
RUN apt-get update && apt-get install gcc -y

WORKDIR /app

RUN python3 -m venv .venv

COPY . .

RUN . .venv/bin/activate
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python3", "main.py"]
