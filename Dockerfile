FROM python:3.10-slim

WORKDIR /workspace

# Instalar dependencias de sistema requeridas por OpenCV y utilidades básicas
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalar JupyterLab y las dependencias de tu proyecto
RUN pip install --no-cache-dir jupyterlab ipywidgets && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
