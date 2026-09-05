import io
import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
import urllib.request

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_generator, run_inference, denormalize
from src.models.animegan import AnimeGANv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-anime-api")

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", str(PROJECT_ROOT / "checkpoints" / "epoch_022.pth")
)
MODEL_URL = "https://huggingface.co/dileepkumar5175/face_to_anime/resolve/main/best_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

state = {"cyclegan": None, "animegan": None}


def ensure_checkpoint():
    checkpoint_path = Path(CHECKPOINT_PATH)

    if checkpoint_path.exists():
        logger.info("Checkpoint already exists: %s", checkpoint_path)
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoint not found. Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, checkpoint_path)
    logger.info("Model downloaded to %s", checkpoint_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        checkpoint_path = Path(CHECKPOINT_PATH)
        ensure_checkpoint()
        logger.info("Loading CycleGAN generator from %s on %s", checkpoint_path, DEVICE)
        state["cyclegan"] = load_generator(str(checkpoint_path), DEVICE)
        logger.info("CycleGAN ready.")
    except Exception:
        logger.exception("Failed to load CycleGAN generator")
    try:
        logger.info("Loading AnimeGANv2 (Face Paint v2) on %s", DEVICE)
        state["animegan"] = AnimeGANv2(device=DEVICE)
        logger.info("AnimeGANv2 ready.")
    except Exception:
        logger.exception("Failed to load AnimeGANv2")

    yield

    state["cyclegan"] = None
    state["animegan"] = None


app = FastAPI(
    title="Face-Anime-AI",
    description="Face -> anime inference API (custom CycleGAN + pretrained AnimeGANv2)",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_model_loaded(model_name: str):
    if state.get(model_name) is None:
        raise HTTPException(
            status_code=503,
            detail=f"'{model_name}' model is not loaded. Check the server logs and restart the API.",
        )


@app.get("/health")
def health():
    return {
        "device": DEVICE,
        "cyclegan": "ok" if state["cyclegan"] is not None else "model_not_loaded",
        "animegan": "ok" if state["animegan"] is not None else "model_not_loaded",
        "checkpoint_path": CHECKPOINT_PATH,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = "cyclegan"):
    if model not in {"cyclegan", "animegan"}:
        raise HTTPException(
            status_code=400,
            detail="Unknown model. Use 'cyclegan' or 'animegan'.",
        )
    _ensure_model_loaded(model)

    if file.content_type not in {"image/jpeg", "image/png", "image/jpg", "image/webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. Upload a JPEG/PNG/WEBP image.",
        )

    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    try:
        if model == "animegan":
            output_image = state["animegan"].predict(image)
        else:
            fake_anime = run_inference(state["cyclegan"], image, DEVICE)
            output_tensor = denormalize(fake_anime).clamp(0, 1)
            output_image = transforms.ToPILImage()(output_tensor)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    buffer = io.BytesIO()
    output_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)