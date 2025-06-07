# image python
FROM python:3.10-slim

# directorio de trabajo
WORKDIR /app

#copiar archivos
COPY . .

# instalar dependencias
RUN python -m pip install --upgrade pip && \
    python -m pip install -r \requirements.txt 

# Exponer el puerto 8501
EXPOSE 8501

# Comando para ejecutar la aplicación Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]