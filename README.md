# 🎨 Face → Anime AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CycleGAN-Image%20Translation-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
</p>

<p align="center">
  <strong>Transform real human faces into anime-style artwork using CycleGAN.</strong>
</p>

<p align="center">
  A complete end-to-end deep learning application — from dataset exploration and model training to GPU inference, REST API, React frontend, and Docker deployment.
</p>

<p align="center">
  <a href="#-demo">Demo</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-model">Model</a> •
  <a href="#-results">Results</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-deployment">Deployment</a>
</p>

---

## 🖼️ Demo

> **Real Face → Anime**

<p align="center">
  <img src="docs/images/demo.png" width="850" alt="Face to Anime Demo"/>
</p>

<!--
Replace the image above with your actual before/after screenshot.

Recommended:
docs/
└── images/
    └── demo.png
-->

---

## ✨ What is Face → Anime AI?

**Face → Anime AI** is an unpaired image-to-image translation system built with **CycleGAN**.

The model learns two visual domains:

```text
                    ┌─────────────────────┐
                    │      CycleGAN       │
                    │                     │
 Real Face ───────► │  Generator A → B    │ ───────► Anime
                    │                     │
 Anime ───────────► │  Generator B → A    │ ───────► Real Face
                    └─────────────────────┘
```

The project is designed as a complete ML application rather than only a model experiment.

```text
Dataset
   ↓
EDA
   ↓
Preprocessing
   ↓
PyTorch Dataset / DataLoader
   ↓
CycleGAN Training
   ↓
Checkpointing
   ↓
Inference
   ↓
FastAPI
   ↓
React
   ↓
Docker + CUDA
   ↓
GPU Deployment
```

---

# 🚀 Features

### 🤖 Machine Learning

- CycleGAN-based unpaired image-to-image translation
- ResNet-based generators
- PatchGAN discriminators
- LSGAN adversarial objective
- Cycle-consistency loss
- Identity loss
- RGB image pipeline
- 256 × 256 image generation
- CUDA GPU inference

### 🧪 Data Pipeline

- Dataset exploration and analysis
- Train/test organization
- Image preprocessing
- Configurable transformations
- PyTorch `Dataset`
- PyTorch `DataLoader`
- JSON-based image indexing

### 🌐 Application

- FastAPI inference API
- React web interface
- Image upload and preview
- Real-time transformation
- Generated image preview
- Result download
- API health endpoint

### 🐳 Deployment

- Dockerized backend
- Dockerized frontend
- Docker Compose
- NVIDIA GPU support
- CUDA-enabled PyTorch inference

---

# 🏗️ Architecture

```text
                         USER
                           │
                           ▼
                ┌────────────────────┐
                │     React UI       │
                │                    │
                │  Upload Face       │
                │  Preview Image     │
                │  Generate Anime    │
                └─────────┬──────────┘
                          │
                          │ HTTP POST
                          ▼
                ┌────────────────────┐
                │      FastAPI       │
                │                    │
                │     /predict       │
                │     /health        │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │    Inference       │
                │     Pipeline       │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   G Face → Anime   │
                │                    │
                │   ResNet Generator │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │    PyTorch CUDA    │
                │                    │
                │    NVIDIA GPU      │
                └─────────┬──────────┘
                          │
                          ▼
                    Anime Image
```

---

# 🧠 Model

## 🤖 Pre-trained Model

The trained model checkpoint is hosted on Hugging Face and is not included directly in this GitHub repository.

### 📥 Download

Download the pre-trained model:

👉 [**Download `best_model.pth`**](https://huggingface.co/dileepkumar5175/face_to_anime/resolve/main/best_model.pth)

After downloading, place the file inside the `checkpoints/` directory:

```text
Face-Anime-AI/
├── checkpoints/
│   └── best_model.pth
├── src/
├── app.py
├── requirements.txt
└── README.md
```

The `checkpoints/` directory is excluded from GitHub using `.gitignore` because model checkpoints are not stored in the source repository.


## CycleGAN

The project uses **CycleGAN**, which allows image-to-image translation without requiring paired examples.

Instead of requiring:

```text
Real Face A  ↔  Anime Face A
Real Face B  ↔  Anime Face B
Real Face C  ↔  Anime Face C
```

the model can learn from separate domains:

```text
REAL DOMAIN                  ANIME DOMAIN

Face 1                       Anime 1
Face 2                       Anime 2
Face 3                       Anime 3
Face 4                       Anime 4
  ...                           ...
```

---

## 🔄 Two Generators

### Generator A

```text
G_face2anime

Real Face
    ↓
Anime Image
```

### Generator B

```text
G_anime2face

Anime Image
    ↓
Real-style Face
```

---

## 👁️ Two Discriminators

```text
D_face

Real Face ──────────┐
                    ├──► Real / Fake
Generated Face ─────┘
```

```text
D_anime

Real Anime ─────────┐
                    ├──► Real / Fake
Generated Anime ────┘
```

The discriminators use a **PatchGAN-style architecture** to evaluate local image regions.

---

# 🔁 Cycle Consistency

The key idea behind CycleGAN is that translating an image to another domain and back should approximately reconstruct the original.

```text
                 G_face2anime
Real Face ─────────────────────► Fake Anime
    ▲                                  │
    │                                  │
    │                                  ▼
    └──────────── G_anime2face ────────┘
```

Therefore:

```text
Real Face ≈ Reconstructed Face
```

And in the opposite direction:

```text
Anime → Real → Anime
```

should also preserve the original structure.

---

# 📉 Loss Functions

The generator is optimized using multiple objectives.

### Adversarial Loss

Encourages generated images to look realistic to the discriminator.

### Cycle Consistency Loss

Preserves the important structure of the input image.

```text
L_cycle =
    || G_B(G_A(x)) - x ||₁
  + || G_A(G_B(y)) - y ||₁
```

### Identity Loss

Encourages the generator to preserve images that are already in the target domain.

### Overall Generator Objective

```text
L_G =
    L_adversarial
    + λ_cycle L_cycle
    + λ_identity L_identity
```

---

# 📊 Dataset Pipeline

```text
             RAW DATA
                 │
                 ▼
          ┌─────────────┐
          │     EDA     │
          └──────┬──────┘
                 │
                 ▼
       Data Cleaning / Split
                 │
                 ▼
        Processed Dataset
                 │
                 ▼
          JSON File Lists
                 │
                 ▼
        PyTorch Dataset
                 │
                 ▼
           DataLoader
                 │
                 ▼
             CycleGAN
```

Images are converted to RGB before entering the model.

### Training Transformations

```text
Resize
  ↓
Random Crop
  ↓
Random Horizontal Flip
  ↓
ToTensor
  ↓
Normalize [-1, 1]
```

### Testing Transformations

Testing uses deterministic preprocessing to make checkpoint comparisons more consistent.

---

# 📈 Results

The model is currently being evaluated through:

### Quantitative Evaluation

- Generator losses
- Discriminator losses
- Cycle-consistency loss
- Identity loss
- Training stability

### Qualitative Evaluation

Fixed test images are passed through different checkpoints to compare:

```text
Input
  ↓
Epoch 1
  ↓
Epoch 10
  ↓
Epoch 20
  ↓
Epoch 50
  ↓
Final Model
```

This makes it possible to observe improvements in:

- Facial structure
- Anime style
- Sharpness
- Color consistency
- Detail preservation
- Generalization

> The model is still under active experimentation and improvement.

---

# 🗂️ Project Structure

```text
Face-Anime-AI/
│
├── data/
│   ├── raw_data/
│   └── processed/
│
├── checkpoints/
│   └── best_model.pth
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── cyclegan.py
│   │
│   ├── losses/
│   │   └── losses.py
│   │
│   ├── inference.py
│   └── trainer.py
│
├── frontend/
│   ├── src/
│   ├── Dockerfile
│   └── package.json
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
└── README.md
```

---

# ⚙️ Installation

## Requirements

Recommended environment:

- Python 3.12
- PyTorch
- CUDA-capable NVIDIA GPU for accelerated inference/training
- Node.js
- Docker Desktop

---

## 1. Clone

```bash
git clone https://github.com/dileepkumar2626/Face-Anime-AI.git

cd Face-Anime-AI
```

---

## 2. Python Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧪 Training

Training can be started through the training pipeline:

```bash
python src/trainer.py
```

Example configuration:

```python
train(
    num_epochs=200,
    batch_size=1,
    lr=2e-4,
    lambda_cycle=10.0,
    lambda_identity=0.5
)
```

The training process supports:

- Epoch-based training
- Loss logging
- Model checkpointing
- Generator/discriminator optimization
- GPU acceleration when CUDA is available

---

# 💾 Checkpoints

Checkpoints contain the learned parameters for:

```text
G_face2anime
G_anime2face
D_face
D_anime
```

and optimizer states.

This allows trained models to be loaded later for inference or continued training.

---

# 🖼️ Inference

Generate an anime image from a face image:

```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input path/to/face.jpg \
    --output anime_result.jpg
```

Inference pipeline:

```text
Input Image
     ↓
RGB Conversion
     ↓
Resize
     ↓
Normalize
     ↓
Generator
     ↓
Denormalize
     ↓
Output Image
```

---

# 🌐 FastAPI

The trained model is exposed through a REST API.

Run locally:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Check API/model status |
| `/predict` | POST | Generate anime image |
| `/docs` | GET | Interactive API documentation |

Open:

```text
http://localhost:8000/docs
```

---

# 💻 React Frontend

The frontend provides a simple interface for:

```text
Upload Image
      ↓
Preview
      ↓
Transform
      ↓
FastAPI
      ↓
Anime Result
      ↓
Download
```

Run locally:

```bash
cd frontend
npm install
npm run dev
```

The frontend communicates with:

```text
POST /predict
```

---

# 🐳 Docker

The project is fully containerized.

### Backend

```text
FastAPI
+
PyTorch
+
CUDA
+
CycleGAN
```

### Frontend

```text
React
+
Nginx
```

---

## Build Backend

From the project root:

```bash
docker build -t face-anime-api:gpu .
```

---

## Build Frontend

```bash
cd frontend

docker build -t face-anime-frontend:latest .
```

---

## Run Everything

From the project root:

```bash
docker compose up
```

Frontend:

```text
http://localhost:5173
```

Backend:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

---

# ⚡ GPU Acceleration

The application automatically selects CUDA when available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Docker is configured to expose the NVIDIA GPU to the backend container.

The current development environment has been tested with an NVIDIA RTX 4060.

```text
Docker
   ↓
NVIDIA Container Runtime
   ↓
CUDA
   ↓
PyTorch
   ↓
RTX 4060
   ↓
CycleGAN
```

---

# 🌍 Deployment

The application is currently **containerized and GPU-ready for local deployment**.

The next deployment stage is to move the Dockerized application to a cloud GPU server:

```text
Local Machine
     │
     ▼
Docker Image
     │
     ▼
Cloud GPU Server
     │
     ▼
Docker Compose
     │
     ▼
FastAPI + CycleGAN
     │
     ▼
Public API
     │
     ▼
React Frontend
     │
     ▼
🌍 Public Application
```

---

# 🔬 Future Work

### Model

- [ ] Improve dataset quality
- [ ] Increase dataset diversity
- [ ] Experiment with larger datasets
- [ ] Learning-rate scheduling
- [ ] Fake-image replay buffer
- [ ] Improve training stability
- [ ] Higher-resolution generation
- [ ] Super-resolution refinement

### Evaluation

- [ ] Automated image-quality metrics
- [ ] More systematic benchmark dataset
- [ ] Compare multiple checkpoints
- [ ] Compare baseline vs improved model
- [ ] Evaluate generalization on unseen faces

### Application

- [ ] Public cloud deployment
- [ ] Production monitoring
- [ ] API rate limiting
- [ ] Improved inference performance
- [ ] User authentication
- [ ] Production domain

### Research

- [ ] Compare CycleGAN with modern diffusion-based approaches
- [ ] Investigate perceptual losses
- [ ] Experiment with modern image-to-image architectures

---

# 🎯 Why This Project?

The goal of this project is not only to train an image translation model.

It explores the **complete lifecycle of a deep-learning application**:

```text
                 ┌─────────────┐
                 │   Research  │
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │   Dataset   │
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │     EDA     │
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │Preprocessing│
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │    Model    │
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │   Training  │
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │  Evaluation │
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │  Inference  │
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │   FastAPI   │
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │    React    │
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │   Docker    │
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │ GPU Deploy  │
                 └─────────────┘
```

---

# 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | Pillow / Torchvision |
| Architecture | CycleGAN |
| Generator | ResNet |
| Discriminator | PatchGAN |
| Adversarial Objective | LSGAN |
| Cycle Loss | L1 Cycle Consistency |
| Identity Loss | L1 Identity Loss |
| API | FastAPI |
| Frontend | React |
| Web Server | Nginx |
| Containerization | Docker |
| Orchestration | Docker Compose |
| GPU Acceleration | NVIDIA CUDA |
| Version Control | Git / GitHub |

---

# 👨‍💻 Author

## Dileep Kumar

**AI/ML Engineer · Deep Learning · Machine Learning Research**

I build machine-learning and deep-learning systems and enjoy turning research ideas into working implementations.

<p align="left">
  <a href="https://github.com/dileepkumar2626">
    <img src="https://img.shields.io/badge/GitHub-Dileep%20Kumar-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>

  <a href="https://www.linkedin.com/in/dileep-kumarbh/">
    <img src="https://img.shields.io/badge/LinkedIn-Dileep%20Kumar-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>

---

# 📚 Research

This project is based on the CycleGAN research:

**Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**

The implementation uses PyTorch to build the training and inference pipeline.

---

# 📄 License

This project is intended for educational, research, and portfolio purposes.
