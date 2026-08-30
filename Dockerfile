FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    Pillow>=9.0.0 \
    scikit-learn>=1.3.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    python-multipart>=0.0.6

COPY src ./src
COPY checkpoints ./checkpoints

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]