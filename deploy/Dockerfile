FROM python:3.10

WORKDIR /app

COPY deployrequirements.txt .

RUN pip install --no-cache-dir -r deployrequirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "deployapp:app", "--host", "0.0.0.0", "--port", "8000"]

