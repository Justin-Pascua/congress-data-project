FROM python:3.13-slim-trixie

WORKDIR /app

RUN pip install --upgrade pip

# cpu only version of PyTorch to reduce size
RUN pip install \
    torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./api

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]