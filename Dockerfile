FROM apache/airflow:3.1.7

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r /requirements.txt