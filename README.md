# 🎨 Face → Anime AI

```{=html}
<p align="center">
```
`<img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />`{=html}
`<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />`{=html}
`<img src="https://img.shields.io/badge/CycleGAN-Unpaired%20Translation-8A2BE2?style=for-the-badge" />`{=html}
`<img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" />`{=html}
`<img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black" />`{=html}
`<img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" />`{=html}
`<img src="https://img.shields.io/badge/CUDA-GPU%20Ready-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<strong>`{=html}Unpaired real-face → anime image translation with a
custom PyTorch CycleGAN pipeline.`</strong>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
A complete computer-vision project covering dataset analysis,
preprocessing, CycleGAN training, checkpointing, inference, a FastAPI
backend, a React frontend, and containerization.
```{=html}
</p>
```

------------------------------------------------------------------------

## ✨ Overview

**Face → Anime AI** learns to translate images between two visual
domains:

``` text
REAL FACE DOMAIN                    ANIME DOMAIN

      Real Face
          │
          ▼
   ┌───────────────┐
   │ G_face2anime  │ ───────────────► Anime
   └───────────────┘

   ┌───────────────┐
   │ G_anime2face  │ ◄─────────────── Anime
   └───────────────┘
          ▲
          │
      Real-style Face
```

Unlike paired image-to-image translation, the training pipeline does
**not require a one-to-one real/anime correspondence**. The dataset
loader samples independently from the two domains.

The repository also contains an application layer:

``` text
Dataset
   ↓
EDA + validation
   ↓
Duplicate detection + train/test split
   ↓
JSON image indexes
   ↓
PyTorch Dataset / DataLoader
   ↓
CycleGAN
   ↓
Checkpoints
   ↓
Inference
   ↓
FastAPI
   ↓
React
   ↓
Docker
```

------------------------------------------------------------------------

# 🧪 Dataset & EDA

The included EDA notebook (`notebooks/01_EDA.ipynb`) records the dataset
inspection performed before training.

### Anime domain

  Property               Observed value
  -------------------- ----------------
  Images found              **302,652**
  Valid images              **302,652**
  Corrupt/unreadable              **0**
  Duplicate images                **0**
  Resolution              **512 × 512**
  Color mode                    **RGB**

Sample images from the actual EDA notebook:

```{=html}
<p align="center">
```
`<img src="docs/images/anime_eda_samples.png" width="900" alt="Anime dataset EDA samples"/>`{=html}
```{=html}
</p>
```
### Real-face domain

  Property                      Observed value
  --------------------------- ----------------
  Images found                     **202,599**
  Valid images                     **202,599**
  Corrupt/unreadable                     **0**
  Duplicate images detected            **131**
  Resolution                     **178 × 218**
  Color mode                           **RGB**

The preprocessing pipeline removes duplicate entries before the
train/test split.

```{=html}
<p align="center">
```
`<img src="docs/images/real_eda_samples.png" width="900" alt="Real face dataset EDA samples"/>`{=html}
```{=html}
</p>
```
### Dataset split

After duplicate removal and an 80/20 split:

``` text
Anime
302,652 total
├── 242,121 train
└── 60,531 test

Real
202,599 total
−     131 duplicates
= 202,468 unique images
├── 161,974 train
└── 40,494 test
```

The generated JSON indexes are stored in:

``` text
data/processed/
├── anime_train/anime_train.json
├── anime_test/anime_test.json
├── real_train/real_train.json
└── real_test/real_test.json
```

The actual raw image datasets are intentionally not stored in the
repository.

------------------------------------------------------------------------

# 🧠 Model Architecture

The implementation uses a four-network CycleGAN:

``` text
                         CYCLE 1

       Real Face ──► G_face2anime ──► Fake Anime
          ▲                              │
          │                              │
          └──── G_anime2face ◄──────────┘


                         CYCLE 2

         Anime ──► G_anime2face ──► Fake Face
          ▲                              │
          │                              │
          └──── G_face2anime ◄──────────┘
```

### Generators

Both generators use a ResNet-style architecture:

-   3-channel RGB input/output
-   64 initial feature channels
-   2 downsampling stages
-   **9 residual blocks**
-   2 upsampling stages
-   Reflection padding
-   Instance normalization
-   ReLU activations
-   `Tanh` output

``` text
RGB Image
   ↓
7×7 convolution
   ↓
Downsample ×2
   ↓
9 × Residual Blocks
   ↓
Upsample ×2
   ↓
7×7 convolution
   ↓
Tanh
   ↓
RGB Image
```

### Discriminators

Two PatchGAN-style discriminators are used:

``` text
D_face
Real Face / Generated Face
          ↓
     Local patches
          ↓
   Real / Fake scores


D_anime
Real Anime / Generated Anime
          ↓
     Local patches
          ↓
   Real / Fake scores
```

The discriminator implementation uses convolutional blocks with:

-   64 base filters
-   Instance normalization
-   LeakyReLU
-   Patch-based output

------------------------------------------------------------------------

# 📉 Loss Functions

The training objective combines adversarial, cycle-consistency, and
identity terms.

### Adversarial loss

The implementation uses an **MSE-based GAN objective** (LSGAN-style).

``` text
Generated image
      ↓
Discriminator
      ↓
MSE against "real" target
```

### Cycle-consistency loss

The model is encouraged to reconstruct the original image after
translating to the other domain and back:

``` text
L_cycle =
    || G_anime2face(G_face2anime(x)) - x ||₁
  + || G_face2anime(G_anime2face(y)) - y ||₁
```

### Identity loss

Identity mapping is also evaluated:

``` text
G_anime2face(real face) ≈ real face
G_face2anime(anime)     ≈ anime
```

### Overall generator objective

The implementation combines:

``` text
L_G =
    L_adv(face → anime)
  + L_adv(anime → face)
  + λ_cycle L_cycle
  + λ_identity L_identity
```

Configured defaults in the implementation include:

``` text
learning rate      = 2e-4
Adam betas         = (0.5, 0.999)
lambda_cycle       = 10.0
lambda_identity    = 0.5
image pool size    = 50
```

The implementation internally scales the identity coefficient by the
cycle coefficient when constructing the final loss.

------------------------------------------------------------------------

# 🖼️ Image Pipeline

Training uses a common model resolution of:

``` text
256 × 256
```

### Training

``` text
Input
  ↓
Resize to ~1.12 × target
  ↓
Random crop → 256 × 256
  ↓
Random horizontal flip
  ↓
ToTensor
  ↓
Normalize to [-1, 1]
  ↓
CycleGAN
```

### Testing / inference

``` text
Input
  ↓
Resize → 256 × 256
  ↓
ToTensor
  ↓
Normalize to [-1, 1]
  ↓
Generator
  ↓
Denormalize
  ↓
Output image
```

Images are converted to RGB when loaded.

------------------------------------------------------------------------

# 💾 Checkpointing

The project saves both epoch-specific checkpoints and a `best_model.pth`
checkpoint.

An epoch checkpoint contains:

``` text
epoch
G_face2anime
G_anime2face
D_face
D_anime
optimizer_G
optimizer_D_face
optimizer_D_anime
scheduler_G
scheduler_D_face
scheduler_D_anime
```

The default checkpoint directory is:

``` text
checkpoints/
```

Model files are excluded from Git using `.gitignore`.

The inference code specifically loads:

``` text
G_face2anime
```

from the checkpoint.

------------------------------------------------------------------------

# 📊 Results

The ZIP used for this README contains the EDA notebook and its recorded
dataset samples, but it **does not contain epoch-by-epoch generated
result images, training curves, or Real-ESRGAN output files**.

Therefore this README deliberately does **not** invent epoch numbers,
generated-output filenames, visual quality claims, or super-resolution
comparisons that are not present in the supplied project snapshot.

For the same reason, the repository should only add an epoch-progression
gallery after the corresponding files are actually committed, for
example:

``` text
Input → checkpoint → generated output
```

That keeps the README reproducible and evidence-based.

------------------------------------------------------------------------

# 🌐 Application

The project includes a FastAPI inference service and a React frontend.

## Backend

The FastAPI application exposes:

``` text
GET  /health
POST /predict
```

### `/health`

Reports:

-   API status
-   selected device
-   checkpoint path
-   whether the generator is loaded

### `/predict`

Accepts:

``` text
JPEG
PNG
WEBP
```

The request is processed as:

``` text
Uploaded image
      ↓
PIL RGB conversion
      ↓
CycleGAN generator
      ↓
Denormalization
      ↓
JPEG response
```

If the checkpoint is missing, the backend can download it from the
configured Hugging Face model URL before loading the generator.

------------------------------------------------------------------------

# 🎨 React Frontend

The frontend is implemented with:

-   React
-   Vite
-   CSS
-   Fetch API

The UI provides:

``` text
┌────────────────────┐
│     ORIGINAL       │
│                    │
│   Upload face      │
│                    │
└─────────┬──────────┘
          │
     TRANSFORM
          │
          ▼
┌────────────────────┐
│    TRANSFORMED     │
│                    │
│   Anime result     │
│                    │
└────────────────────┘
```

The interface supports:

-   Image selection
-   Image preview
-   Transformation request
-   Loading state
-   Error display
-   Generated-image preview
-   Result download
-   Image swapping

------------------------------------------------------------------------

# 🐳 Docker

The project contains separate containers for the backend and frontend.

``` text
                 Docker Compose
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   FastAPI backend             React frontend
      port 8000                   port 5173
          │
          ▼
   PyTorch / CUDA image
          │
          ▼
   /app/checkpoints/
```

### Backend image

The backend Dockerfile is based on:

``` text
pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
```

### Frontend image

The frontend uses:

``` text
node:22-alpine
```

for the build stage and:

``` text
nginx:alpine
```

for serving the production frontend.

### Important GPU note

The backend image is CUDA-capable, but the supplied `docker-compose.yml`
currently has the GPU reservation commented out:

``` yaml
# gpus: all
```

So GPU access through Docker Compose must be explicitly enabled in the
deployment configuration and requires a working NVIDIA Container Toolkit
/ Docker GPU setup.

------------------------------------------------------------------------

# ⚙️ Installation

## Requirements

The repository is built around:

-   Python 3.12+
-   PyTorch
-   Torchvision
-   Pillow
-   scikit-learn
-   FastAPI
-   Uvicorn
-   python-multipart
-   Node.js 22 for the supplied frontend container
-   Docker Desktop / Docker Engine
-   NVIDIA GPU + CUDA support for accelerated training/inference

## Clone

``` bash
git clone https://github.com/dileepkumar2626/Face-Anime-AI.git
cd Face-Anime-AI
```

## Python environment

``` bash
python -m venv .venv
```

### Windows

``` bash
.venv\Scripts\activate
```

### Linux / macOS

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🏋️ Training

The main training entry point is:

``` text
src/trainer.py
```

The trainer supports configuration for:

``` text
num_epochs
learning rate
cycle loss weight
identity loss weight
checkpoint directory
logging interval
learning-rate decay epoch
device
```

Example:

``` python
from src.trainer import train

train(
    num_epochs=200,
    lr=2e-4,
    lambda_cycle=10.0,
    lambda_identity=0.5,
    checkpoint_dir="checkpoints",
    decay_epoch=100,
)
```

The implementation supports CUDA automatically when available:

``` python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

The learning-rate scheduler is designed for a constant-learning-rate
phase followed by linear decay.

------------------------------------------------------------------------

# 🔎 Command-Line Inference

The inference entry point is:

``` text
src/inference.py
```

Run:

``` bash
python -m src.inference \
  --checkpoint checkpoints/best_model.pth \
  --input path/to/face.jpg \
  --output anime_result.jpg
```

The inference pipeline loads only the `G_face2anime` generator from the
checkpoint.

------------------------------------------------------------------------

# 🚀 Run the API

From the repository root:

``` bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

API:

``` text
http://localhost:8000
```

Swagger UI:

``` text
http://localhost:8000/docs
```

Health check:

``` text
http://localhost:8000/health
```

------------------------------------------------------------------------

# 💻 Run the Frontend

``` bash
cd frontend
npm install
npm run dev
```

The Vite development server uses the frontend environment variable:

``` text
VITE_API_URL
```

Example:

``` bash
VITE_API_URL=http://localhost:8000 npm run dev
```

On Windows PowerShell:

``` powershell
$env:VITE_API_URL="http://localhost:8000"
npm run dev
```

------------------------------------------------------------------------

# 🐳 Run with Docker Compose

Build and start:

``` bash
docker compose up --build
```

The supplied compose configuration exposes:

``` text
Frontend  → http://localhost:5173
Backend   → http://localhost:8000
```

The checkpoint volume is:

``` text
model_cache:/app/checkpoints
```

and the backend expects:

``` text
/app/checkpoints/best_model.pth
```

------------------------------------------------------------------------

# 📁 Project Structure

``` text
Face-Anime-AI/
│
├── data/
│   └── processed/
│       ├── anime_train/
│       │   └── anime_train.json
│       ├── anime_test/
│       │   └── anime_test.json
│       ├── real_train/
│       │   └── real_train.json
│       └── real_test/
│           └── real_test.json
│
├── frontend/
│   ├── public/
│   │   ├── favicon.svg
│   │   └── icons.svg
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── Dockerfile
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── notebooks/
│   └── 01_EDA.ipynb
│
├── src/
│   ├── api/
│   │   ├── app.py
│   │   └── init.py
│   │
│   ├── data/
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── test_dataset.ipynb
│   │   └── transforms.py
│   │
│   ├── models/
│   │   ├── cyclegan.py
│   │   ├── discriminator.py
│   │   ├── generator.py
│   │   └── losses.py
│   │
│   ├── utils/
│   │   ├── checkpoint.py
│   │   └── image_pool.py
│   │
│   ├── inference.py
│   └── trainer.py
│
├── awesome-project/
│   ├── pyproject.toml
│   └── uv.lock
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

# 🔬 Engineering & Research Aspects

This project is more than a single pretrained-model demo. It covers
several stages of an ML research workflow:

### Data

-   Dataset validation
-   Corrupt-image checking
-   Resolution analysis
-   Color-mode analysis
-   Duplicate detection
-   Reproducible train/test split
-   JSON-based dataset indexing

### Modeling

-   Custom ResNet generator implementation
-   Custom PatchGAN discriminator
-   Cycle-consistency objective
-   Identity regularization
-   Image replay pools
-   Learning-rate scheduling

### Experimentation

-   Training-loss logging
-   Epoch checkpoints
-   Fixed test preprocessing
-   Qualitative model evaluation workflow

### Deployment

-   Standalone inference
-   REST API
-   Browser frontend
-   Docker containers
-   CUDA-capable backend image

------------------------------------------------------------------------

# ⚠️ Current Limitations

The current implementation has several practical limitations:

1.  **Training resolution is 256 × 256**, so fine facial details can be
    lost.
2.  **The real dataset starts at 178 × 218**, meaning training
    resolution requires upscaling.
3.  **CycleGAN is unpaired**, so exact identity preservation is not
    guaranteed.
4.  GAN losses alone are not sufficient to judge visual quality.
5.  The repository snapshot does not include epoch-by-epoch generated
    samples or quantitative image-quality metrics.
6.  The supplied Docker Compose file does not currently enable GPU
    reservation because `gpus: all` is commented out.
7.  The checkpoint is intentionally excluded from Git and handled
    separately.
8.  The inference pipeline currently returns a JPEG generated at the
    model's working resolution; a separate super-resolution stage is not
    included in the supplied source tree.

------------------------------------------------------------------------

# 🛣️ Future Work

Possible next experiments include:

-   Train for longer and compare fixed checkpoints
-   Add an automated validation/evaluation pipeline
-   Save generated samples at every selected epoch
-   Add training curves for generator/discriminator/cycle/identity
    losses
-   Experiment with batch size and augmentation policies
-   Improve face alignment/cropping
-   Experiment with perceptual or feature-based losses
-   Add a dedicated super-resolution stage after translation
-   Compare different anime-domain models
-   Add experiment tracking
-   Add automated API tests
-   Enable GPU reservations in Docker Compose for NVIDIA deployments
-   Add CI/CD for the frontend and backend

------------------------------------------------------------------------

# 📚 References

### CycleGAN

Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A.\
**Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
Networks.**

https://arxiv.org/abs/1703.10593

### Official CycleGAN implementation

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

### Real-ESRGAN

Wang et al.\
**Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure
Synthetic Data.**

https://github.com/xinntao/Real-ESRGAN

------------------------------------------------------------------------

# 👨‍💻 Author

**Dileep Kumar**

AI/ML Engineer • Deep Learning • Computer Vision • ML Research

```{=html}
<p>
```
`<a href="https://github.com/dileepkumar2626">`{=html} GitHub
`</a>`{=html}  • 
`<a href="https://www.linkedin.com/in/dileep-kumarbh/">`{=html} LinkedIn
`</a>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## ⭐ Project Summary

``` text
Real Faces
    │
    ▼
Data Validation + EDA
    │
    ▼
Unpaired Dataset
    │
    ▼
Custom PyTorch CycleGAN
    │
    ├── G_face2anime
    ├── G_anime2face
    ├── D_face
    └── D_anime
    │
    ▼
256 × 256 Anime Translation
    │
    ▼
PyTorch Inference
    │
    ▼
FastAPI
    │
    ▼
React
    │
    ▼
Docker / CUDA-ready deployment
```

**Face → Anime AI** demonstrates an end-to-end approach to building,
evaluating, and serving a custom generative computer-vision system.
