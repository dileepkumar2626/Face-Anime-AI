import io
import os
import sys
import logging
import urllib.request

from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_generator, run_inference, denormalize
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-anime-api")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = Path(
    os.environ.get(
        "CHECKPOINT_DIR",
        str(PROJECT_ROOT / "checkpoints")
    )
)
HF_BASE_URL = os.environ.get(
    "HF_BASE_URL",
    "https://huggingface.co/dileepkumar5175/face_to_anime/resolve/main"
)


MODELS = {
    "cyclegan_v1": {
        "name": "CycleGAN V1",
        "description": "Original CycleGAN face-to-anime model.",
        "filename": "best_model.pth",
        "url": f"{HF_BASE_URL}/best_model.pth",
    },

    "anime2_v2": {
        "name": "Anime2 V2",
        "description": "Updated Anime2 V2 face-to-anime model.",
        "filename": "anime2_v2.pth",
        "url": f"{HF_BASE_URL}/anime2_v2.pth",
    },
}
state = {
    "generators": {},
}
def ensure_checkpoint(model_id: str) -> Path:
    """
    Make sure the requested model checkpoint exists locally.

    Each model has its own filename, so Docker's persistent
    model_cache volume cannot accidentally replace one model
    with another.
    """

    if model_id not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_id}",
        )

    model_info = MODELS[model_id]

    checkpoint_path = CHECKPOINT_DIR / model_info["filename"]

    if checkpoint_path.exists():
        logger.info(
            "Checkpoint already exists for %s: %s",
            model_id,
            checkpoint_path,
        )
        return checkpoint_path

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Checkpoint for %s not found. Downloading from %s",
        model_id,
        model_info["url"],
    )

    try:
        urllib.request.urlretrieve(
            model_info["url"],
            checkpoint_path,
        )
    except Exception as exc:
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        logger.exception(
            "Failed to download model %s",
            model_id,
        )

        raise RuntimeError(
            f"Could not download model '{model_id}' "
            f"from {model_info['url']}"
        ) from exc

    logger.info(
        "Model %s downloaded to %s",
        model_id,
        checkpoint_path,
    )

    return checkpoint_path


# ============================================================
# Load model
# ============================================================

def get_generator(model_id: str):
    """
    Load a model only when it is requested.

    Models are cached in memory after the first request.
    """

    if model_id not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_id}",
        )

    # Already loaded
    if model_id in state["generators"]:
        return state["generators"][model_id]

    checkpoint_path = ensure_checkpoint(model_id)

    logger.info(
        "Loading model '%s' from %s on %s",
        model_id,
        checkpoint_path,
        DEVICE,
    )

    try:
        generator = load_generator(
            str(checkpoint_path),
            DEVICE,
            model_id,
        )
    except Exception as exc:
        logger.exception(
            "Failed to load model '%s'",
            model_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to load model '{model_id}': {exc}"
            ),
        ) from exc

    state["generators"][model_id] = generator

    logger.info(
        "Model '%s' loaded successfully.",
        model_id,
    )

    return generator
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Face-Anime-AI API starting on device: %s",
        DEVICE,
    )

    logger.info(
        "Available models: %s",
        ", ".join(MODELS.keys()),
    )
    yield

    logger.info("Shutting down Face-Anime-AI API.")

    state["generators"].clear()
app = FastAPI(
    title="Face-Anime-AI",
    description="Unpaired CycleGAN face-to-anime inference API.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://<your-server-ip-or-domain>:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
def health():
    """
    API health/status endpoint.
    """

    return {
        "status": "ok",
        "device": DEVICE,
        "available_models": list(MODELS.keys()),
        "loaded_models": list(state["generators"].keys()),
        "checkpoint_directory": str(CHECKPOINT_DIR),
    }


@app.get("/models")
def get_models():
    """
    Return all models available to the frontend.
    """

    return {
        "models": [
            {
                "id": model_id,
                "name": model_info["name"],
                "description": model_info["description"],
            }
            for model_id, model_info in MODELS.items()
        ]
    }


@app.get("/models/{model_id}")
def get_model(model_id: str):
    """
    Return information about one model.
    """

    if model_id not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_id}",
        )

    model_info = MODELS[model_id]

    checkpoint_path = CHECKPOINT_DIR / model_info["filename"]

    return {
        "id": model_id,
        "name": model_info["name"],
        "description": model_info["description"],
        "checkpoint": model_info["filename"],
        "downloaded": checkpoint_path.exists(),
        "loaded": model_id in state["generators"],
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("anime2_v2"),
):
    if model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model '{model}'. "
                f"Available models: {list(MODELS.keys())}"
            ),
        )
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type: {file.content_type}. "
                "Upload a JPEG/PNG/WEBP image."
            ),
        )
    generator = get_generator(model)
    raw_bytes = await file.read()

    try:
        image = Image.open(
            io.BytesIO(raw_bytes)
        ).convert("RGB")

    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read the uploaded file "
                "as an image."
            ),
        ) from exc
    try:

        logger.info(
            "Running inference with model '%s'",
            model,
        )

        fake_anime = run_inference(
            generator,
            image,
            DEVICE,
        )

    except Exception as exc:

        logger.exception(
            "Inference failed using model '%s'",
            model,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Inference failed with model "
                f"'{model}': {exc}"
            ),
        ) from exc
    output_tensor = (
        denormalize(fake_anime)
        .clamp(0, 1)
    )

    output_image = transforms.ToPILImage()(
        output_tensor
    )
    buffer = io.BytesIO()

    output_image.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/jpeg",
        headers={
            "X-Model": model,
            "X-Model-Name": MODELS[model]["name"],
        },
    )
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )