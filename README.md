# 🎨 Face → Anime AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CycleGAN-Image%20Translation-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
</p>

<p align="center">
  <strong>Turn a real human face into anime-style art — with your own trained CycleGAN or a pretrained AnimeGANv2 model.</strong>
</p>

<p align="center">
  A complete end-to-end deep learning application: dataset exploration, model training from scratch,
  face-aware inference, a dual-model REST API, a React frontend, and GPU-ready Docker deployment.
</p>

<p align="center">
  <a href="#-what-is-face--anime-ai">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-model">Model</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-training">Training</a> •
  <a href="#-inference">Inference</a> •
  <a href="#-fastapi">API</a> •
  <a href="#-docker">Docker</a>
</p>

---

## ✨ What is Face → Anime AI?

**Face → Anime AI** is an unpaired image-to-image translation system. At its core is a **CycleGAN**
trained from scratch on a real-face domain and an anime-face domain, with no paired examples required.
The API also ships a second, pretrained option — **AnimeGANv2 (face_paint_512_v2)** — so a user can
compare the custom model against a well-known reference model from a single interface.

```text
                    ┌─────────────────────┐
                    │      CycleGAN       │
                    │                     │
 Real Face ───────► │  G: Face → Anime    │ ───────► Anime
                    │                     │
 Anime ───────────► │  G: Anime → Face    │ ───────► Real Face
                    └─────────────────────┘
```

The project is built as a complete ML application, not just a training script:

```text
Dataset → EDA → Preprocessing → PyTorch Dataset/DataLoader
   → CycleGAN Training → Checkpointing → Face-Cropped Inference
   → FastAPI (dual model) → React → Docker + CUDA → GPU Deployment
```

---

## 🚀 Features

### 🤖 Machine Learning

- Custom **CycleGAN** trained from scratch (unpaired face ↔ anime translation)
- ResNet-based generator (9 residual blocks, reflection padding, instance norm)
- PatchGAN discriminator (4-layer conv stack)
- LSGAN adversarial objective, cycle-consistency loss (L1), identity loss (L1)
- **Fake-image replay buffer** (`ImagePool`, size 50) to stabilize discriminator training
- **Linear learning-rate decay** starting at the midpoint of training
- Second, pretrained **AnimeGANv2** model (`face_paint_512_v2`) available as an alternate style
- 256×256 training resolution, CUDA-accelerated training and inference

### 🧪 Data Pipeline

- Jupyter-based EDA (`notebooks/01_EDA.ipynb`)
- JSON-indexed train/test splits for both domains (`data/processed/*/*.json`)
- Configurable `torchvision` transform pipelines for train vs. test
- Custom PyTorch `Dataset` that pairs a real face with a randomly sampled anime image each step
- **Automatic face detection & cropping** at inference time (OpenCV Haar cascade, with margin and square-padding), so arbitrary photos don't need to be pre-cropped

### 🌐 Application

- FastAPI inference service with **model selection** (`cyclegan` or `animegan`) per request
- Automatic checkpoint download from Hugging Face on first startup if none is found locally
- React + Vite frontend: upload, live preview, model picker, generate, download result
- `/health` endpoint reporting device and per-model load status

### 🐳 Deployment

- Separate Dockerfiles for backend (PyTorch + CUDA base image) and frontend (Node build → Nginx)
- `docker-compose.yml` wiring both services together with a shared checkpoint volume
- GPU passthrough ready (NVIDIA Container Runtime)

---

## 🏗️ Architecture

```text
                         USER
                           │
                           ▼
                ┌────────────────────┐
                │   React Frontend   │
                │                    │
                │  Upload · Preview  │
                │  Pick Model        │
                │  Generate          │
                └─────────┬──────────┘
                          │ POST /predict?model=cyclegan|animegan
                          ▼
                ┌────────────────────┐
                │      FastAPI       │
                │  /health /predict  │
                └─────────┬──────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
    ┌───────────────────┐   ┌─────────────────────────┐
    │  Custom CycleGAN   │   │  AnimeGANv2 (pretrained) │
    │  face-crop → resize│   │  face2paint pipeline     │
    │  G_face2anime      │   │  torch.hub weights       │
    └─────────┬──────────┘   └───────────┬─────────────┘
              └───────────┬──────────────┘
                          ▼
                ┌────────────────────┐
                │   PyTorch + CUDA   │
                └─────────┬──────────┘
                          ▼
                    Anime Image
```

---

## 🧠 Model

### Pretrained checkpoint (custom CycleGAN)

The trained CycleGAN checkpoint is hosted on Hugging Face rather than committed to Git:

👉 [**Download `best_model.pth`**](https://huggingface.co/dileepkumar5175/face_to_anime/resolve/main/best_model.pth)

Place it inside `checkpoints/`:

```text
Face-Anime-AI/
├── checkpoints/
│   └── best_model.pth   (or epoch_XXX.pth — see note below)
├── src/
├── requirements.txt
└── README.md
```

> **Note:** the API's default `CHECKPOINT_PATH` points at `checkpoints/epoch_024.pth`. If you
> download `best_model.pth` instead, either rename it to match or set the `CHECKPOINT_PATH`
> environment variable to the file you have. If no checkpoint is present at startup, the API
> automatically downloads `best_model.pth` from Hugging Face into that path.

The `checkpoints/` directory is excluded from Git via `.gitignore`.

### Why CycleGAN?

CycleGAN removes the need for paired training data. Instead of requiring matched pairs like:

```text
Real Face A  ↔  Anime Face A
Real Face B  ↔  Anime Face B
```

it learns from two independent, unpaired collections:

```text
REAL DOMAIN                  ANIME DOMAIN
Face 1                       Anime 1
Face 2                       Anime 2
Face 3                       Anime 3
  ...                           ...
```

### Two generators, two discriminators

```text
G_face2anime : Real Face → Anime Image
G_anime2face : Anime Image → Real-style Face

D_face  : Real Face vs. Generated Face   (PatchGAN)
D_anime : Real Anime vs. Generated Anime (PatchGAN)
```

### Cycle consistency

```text
                 G_face2anime
Real Face ─────────────────────► Fake Anime
    ▲                                  │
    │                                  ▼
    └──────────── G_anime2face ────────┘

Real Face ≈ Reconstructed Face
```

### Loss functions

```text
L_G = L_adversarial(LSGAN) + λ_cycle · L_cycle(L1) + λ_identity · L_identity(L1)
```

Defaults used in training: `lambda_cycle=10.0`, `lambda_identity=0.5`, `lr=2e-4`, with linear LR
decay beginning halfway through the run and a 50-image fake-sample replay buffer feeding the
discriminators.

---

## 📊 Dataset & Preprocessing

```text
Raw Data → EDA (notebooks/01_EDA.ipynb) → JSON train/test index files
   → CycleGANDataset (pairs real ↔ randomly-sampled anime) → DataLoader → CycleGAN
```

- **Training transforms:** resize (1.12×) → random crop to 256×256 → random horizontal flip → tensor → normalize to `[-1, 1]`
- **Test transforms:** deterministic resize to 256×256 → tensor → normalize
- **Inference preprocessing:** the input photo is first run through `crop_face()`, which uses an OpenCV Haar-cascade face detector to find the largest face, adds a 25% margin, and pads it to a square before resizing — so users can upload a normal photo, not a pre-cropped headshot.

---

## 📈 Results

### Training progression

The same input, translated by checkpoints from different points in training — the model starts
from noisy color blobs and progressively learns anime-style eyes, shading, and line work:

<p align="center">
  <img src="docs/images/epoch_progression.jpg" width="850" alt="CycleGAN output at epochs 1, 5, 10, 15, 20, and 25"/>
</p>

### Latest result

Output from the most recent checkpoint (epoch 25):

<p align="center">
  <img src="docs/images/latest_result.jpg" width="220" alt="Latest CycleGAN face-to-anime result"/>
</p>

> This is the newest checkpoint in `data/sample_images/`, not necessarily the final trained
> model — swap in your own `best_model.pth` output here once training finishes.

### Anime-domain reference samples

A few examples from the anime-face domain the generator is trained to match:

<p align="center">
  <img src="docs/images/anime_domain_samples.jpg" width="850" alt="Sample images from the anime training domain"/>
</p>

---

## 🗂️ Project Structure

```text
Face-Anime-AI/
│
├── data/
│   ├── processed/          # JSON index files for real/anime train & test splits
│   └── sample_images/      # Sample outputs / epoch previews
│
├── checkpoints/            # Model weights (gitignored)
│
├── src/
│   ├── api/
│   │   └── app.py          # FastAPI service (CycleGAN + AnimeGANv2)
│   ├── data/
│   │   ├── dataset.py       # CycleGANDataset + train/test dataset instances
│   │   ├── dataloader.py    # DataLoaders
│   │   ├── preprocessing.py
│   │   └── transforms.py    # train/test transforms + face-crop utility
│   ├── models/
│   │   ├── generator.py     # ResnetGenerator
│   │   ├── discriminator.py # PatchDiscriminator
│   │   ├── cyclegan.py      # CycleGAN wrapper (optimizers, schedulers, train_step)
│   │   ├── animegan.py      # AnimeGANv2 wrapper (torch.hub)
│   │   └── losses.py
│   ├── utils/
│   │   ├── checkpoint.py    # save/load checkpoint
│   │   └── image_pool.py    # fake-image replay buffer
│   ├── inference.py         # CLI inference script
│   └── trainer.py           # Training loop
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── test_evaluation.ipynb
│
├── frontend/                # React + Vite app
│   ├── src/App.jsx
│   └── Dockerfile
│
├── Dockerfile               # Backend image (PyTorch + CUDA + FastAPI)
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

**Requirements:** Python 3.12, a CUDA-capable NVIDIA GPU (recommended, not required), Node.js, Docker Desktop (optional, for containerized runs).

```bash
git clone https://github.com/dileepkumar2626/Face-Anime-AI.git
cd Face-Anime-AI

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Core dependencies: `torch`, `torchvision`, `Pillow`, `opencv-python-headless` (for face cropping),
`scikit-learn`, `fastapi`, `uvicorn`, `python-multipart`.

---

## 🧪 Training

```bash
python src/trainer.py
```

`train()` accepts:

```python
train(
    num_epochs=50,
    lr=2e-4,
    lambda_cycle=10.0,
    lambda_identity=0.5,
    checkpoint_dir="checkpoints",
    log_every=50,
    decay_epoch=None,   # defaults to num_epochs // 2
    device=None,        # auto-detects CUDA
)
```

Each epoch saves a numbered checkpoint (`epoch_NNN.pth`) and updates `best_model.pth` whenever the
average epoch loss improves. Batches print running totals for the discriminator, cycle, and
identity losses every `log_every` steps.

---

## 🖼️ Inference

```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input path/to/face.jpg \
    --output anime_result.jpg
```

Pipeline: `Face Detection & Crop → Resize (256×256) → Normalize → G_face2anime → Denormalize → Save`

---

## 🌐 FastAPI

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

| Endpoint  | Method | Purpose                                                  |
|-----------|--------|-----------------------------------------------------------|
| `/health` | GET    | Device + per-model (`cyclegan`, `animegan`) load status   |
| `/predict`| POST   | Upload an image (`file`), pick `model=cyclegan\|animegan` |
| `/docs`   | GET    | Interactive Swagger UI                                    |

At startup the API tries to load both models independently — if one fails to load (e.g. missing
checkpoint or no internet for the `torch.hub` weights), `/health` reports it as `model_not_loaded`
without taking down the other model. CORS is currently scoped to `http://localhost:5173` for local
frontend development.

---

## 💻 React Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI lets a user upload a photo, pick between **CycleGAN (custom model)** and **AnimeGANv2**, preview
the source image, generate, and download the result. It talks to the API via `VITE_API_URL` and
`POST /predict?model=...`.

---

## 🐳 Docker

```bash
docker compose up
```

- **Backend** — `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` base, installs project requirements
  plus FastAPI/uvicorn, pre-downloads the AnimeGANv2 weights at build time, and serves on `:8000`.
- **Frontend** — multi-stage build: `node:22-alpine` builds the Vite app, then `nginx:alpine` serves
  the static bundle on `:80` (mapped to `:5173` by Compose).
- A named volume (`model_cache`) persists downloaded CycleGAN checkpoints across container restarts.
- GPU passthrough is present in `docker-compose.yml` but commented out (`# gpus: all`) — uncomment
  it on a host with the NVIDIA Container Toolkit installed.

Once running:

```text
Frontend: http://localhost:5173
Backend:  http://localhost:8000
API docs: http://localhost:8000/docs
```

---

## ⚡ GPU Acceleration

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Both training and inference auto-detect CUDA and fall back to CPU otherwise. Development and
testing were done on an NVIDIA RTX 4060.

---

## 🔬 Future Work

### Model & Training
- [ ] Larger, more diverse, and better-balanced dataset
- [ ] Higher-resolution generation beyond 256×256
- [ ] Super-resolution refinement pass on outputs
- [ ] Perceptual (VGG-based) loss for sharper detail

### Evaluation
- [ ] Automated image-quality metrics (FID/KID)
- [ ] A fixed benchmark set for checkpoint-to-checkpoint comparison
- [ ] Systematic evaluation of generalization on unseen faces

### Application
- [ ] Public cloud GPU deployment
- [ ] Production monitoring and API rate limiting
- [ ] Fix the default `CHECKPOINT_PATH` / checkpoint-filename mismatch between training output and API expectations
- [ ] User authentication

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python, JavaScript |
| Deep Learning | PyTorch, torchvision |
| Face Detection | OpenCV (Haar cascade) |
| Reference Model | AnimeGANv2 (`torch.hub`) |
| Architecture | CycleGAN (ResNet generator, PatchGAN discriminator) |
| API | FastAPI, Uvicorn |
| Frontend | React 19, Vite |
| Web Server | Nginx |
| Containerization | Docker, Docker Compose |
| GPU Acceleration | NVIDIA CUDA |

---

## 👨‍💻 Author

**Dileep Kumar** — AI/ML Engineer · Deep Learning · Machine Learning Research

<p align="left">
  <a href="https://github.com/dileepkumar2626">
    <img src="https://img.shields.io/badge/GitHub-Dileep%20Kumar-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
  <a href="https://www.linkedin.com/in/dileep-kumarbh/">
    <img src="https://img.shields.io/badge/LinkedIn-Dileep%20Kumar-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>

---

## 📚 Research

Based on **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**
(Zhu et al., CycleGAN), with the pretrained comparison model from **AnimeGANv2**
(bryandlee/animegan2-pytorch).

---

## 📄 License

Intended for educational, research, and portfolio purposes.