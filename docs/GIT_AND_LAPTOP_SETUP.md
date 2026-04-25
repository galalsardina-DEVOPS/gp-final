# Git + Laptop Docker Setup

## 1. Before Pushing

Do not push secrets or local generated files:

- `.env` stays local.
- `.venv/` stays local.
- `data/` stays local unless you intentionally want to publish training data.
- `artifacts/` stays local because retrained models can be regenerated.

The current `drowsy_model.h5` is about 11 MB, so it can be pushed to GitHub. If future models become larger than 100 MB, use Git LFS or S3.

## 2. Create GitHub Repository

Create an empty repo on GitHub, for example:

```text
https://github.com/YOUR_USERNAME/ARK.git
```

Do not add README/gitignore/license from GitHub if the local project already has files.

## 3. Push From WSL

```bash
cd /mnt/f/ARK
git init
git add .
git status
git commit -m "Initial drowsy driver platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ARK.git
git push -u origin main
```

If Git asks for login, use GitHub browser login or a Personal Access Token.

## 4. Clone On The Laptop

On the laptop:

```bash
git clone https://github.com/YOUR_USERNAME/ARK.git
cd ARK
cp .env.example .env
```

Edit `.env`:

```env
SERVICE_MODE=local_only
LOCAL_MODEL_BACKEND=keras
LOCAL_MODEL_PATH=./drowsy_model.h5
```

## 5. Run With Docker Compose

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000/healthz
http://localhost:8000/docs
```

Stop:

```bash
docker compose down
```

## 6. Run Camera

Docker Compose runs the API. For laptop camera, run the camera script outside Docker on Windows or directly on Linux.

Windows PowerShell:

```powershell
cd path\to\ARK
py -3.10 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install opencv-python mediapipe numpy pillow requests python-dotenv tensorflow==2.15.1
python scripts\check_camera.py
python scripts\run_monitor.py --source 0
```

## 7. Run Retraining With Docker

Make sure `data/train_data` exists:

```text
data/train_data/alert
data/train_data/drowsy
```

Then:

```bash
docker compose run --rm retrain
```

