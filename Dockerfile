FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN grep -v "^-e \." requirements.txt > requirements_docker.txt && \
    pip install --no-cache-dir -r requirements_docker.txt

COPY app.py setup.py ./
COPY src/ ./src/
COPY templates/ ./templates/
COPY artifact/model.pkl artifact/preprocessor.pkl ./artifact/

RUN pip install --no-cache-dir -e .

EXPOSE 5000

CMD ["python", "app.py"]