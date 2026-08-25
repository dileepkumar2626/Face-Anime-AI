import io
import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.inference import load_generator, run_inference, denormalize
from torchvision import transforms
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-anime-api")
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", str(PROJECT_ROOT / "checkpoints" / "best_model.pth")
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
state = {"generator": None}
@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        logger.warning(
            "Checkpoint not found at %s — /predict will fail until it exists "
            "or CHECKPOINT_PATH is set correctly.",
            checkpoint_path,
        )
    else:
        logger.info("Loading generator from %s on %s", checkpoint_path, DEVICE)
        state["generator"] = load_generator(str(checkpoint_path), DEVICE)
        logger.info("Generator ready.")
    yield
    state["generator"] = None
app = FastAPI(
    title="Face-Anime-AI",
    description="CycleGAN face -> anime inference API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def _ensure_generator_loaded():
    if state["generator"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Generator is not loaded. Check that CHECKPOINT_PATH "
                f"('{CHECKPOINT_PATH}') points to a valid checkpoint and restart the API."
            ),
        )
@app.get("/health")
def health():
    return {
        "status": "ok" if state["generator"] is not None else "model_not_loaded",
        "device": DEVICE,
        "checkpoint_path": CHECKPOINT_PATH,
    }
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    _ensure_generator_loaded()
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
        fake_anime = run_inference(state["generator"], image, DEVICE)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    output_tensor = denormalize(fake_anime).clamp(0, 1)
    output_image = transforms.ToPILImage()(output_tensor)
    buffer = io.BytesIO()
    output_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)