FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi>=0.100.0 uvicorn[standard]>=0.23.0 python-multipart>=0.0.6
RUN python -c "import torch; \
    torch.hub.load('bryandlee/animegan2-pytorch:main', 'generator', pretrained='face_paint_512_v2', device='cpu'); \
    torch.hub.load('bryandlee/animegan2-pytorch:main', 'face2paint', size=512)"
COPY src ./src
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]