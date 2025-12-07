FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installation des deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code applicatif
COPY . /app/

# Entrée du job Cloud Run
CMD ["python", "-u", "mdtojson_runner.py"]
