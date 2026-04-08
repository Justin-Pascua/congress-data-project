FROM python:3.13-slim-trixie

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip

# cpu only version of PyTorch to reduce size
RUN pip install \
    torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r /requirements.txt