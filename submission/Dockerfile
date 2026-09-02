FROM python:3.12-slim

WORKDIR /app

COPY requirements(1).txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run_analysis.py"]
