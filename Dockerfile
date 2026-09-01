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

# checkpoints/ is intentionally NOT copied here — it's gitignored and won't
# exist in a fresh clone, which would make this COPY fail the build.
# It's mounted as a volume instead (see docker-compose.yml), so anyone can
# drop their own best_model.pth in without rebuilding the image.

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]