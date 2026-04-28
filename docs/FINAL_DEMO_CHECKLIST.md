# Final Demo Checklist

Use this checklist before presenting the project.

## 1. Local API

```bash
cd /mnt/f/ARK
docker compose up --build
```

Open:

```text
http://localhost:8000/healthz
http://localhost:8000/docs
http://localhost:8000/dashboard
```

Expected result:

```text
/healthz returns 200 OK.
```

## 2. Real-Time Monitor

Check camera:

```bash
cd ~/ARK
python scripts/check_camera.py
```

Run camera if available:

```bash
python scripts/run_monitor.py --source 0
```

Optional Docker camera run on Linux:

```bash
xhost +local:docker
CAMERA_DEVICE=/dev/video0 docker compose --profile camera run --rm monitor-camera
```

Run video fallback:

```bash
python scripts/run_monitor.py --source /mnt/f/ARK/vvv.mp4
```

Expected result:

```text
The preview window shows ALERT or DROWSY plus EAR, MAR, eye counter, and route.
```

## 3. Retraining

Make sure the dataset exists:

```text
data/train_data/alert
data/train_data/drowsy
```

Run:

```bash
docker compose run --rm retrain
```

Expected result:

```text
artifacts/production/model.keras exists.
artifacts/production/manifest.json exists.
```

Install weekly retraining:

```bash
bash scripts/install_weekly_retraining_cron.sh
```

## 4. Dashboard And Logs

Open:

```text
http://localhost:8000/dashboard
```

Expected result:

```text
The dashboard shows total events, drowsy events, route counts, and recent logs.
```

## 5. Hybrid Mode

In `.env`:

```env
SERVICE_MODE=hybrid
REMOTE_ENDPOINT_URL=http://EC2_PUBLIC_IP:8000/predict
LOCAL_MODEL_BACKEND=keras
LOCAL_MODEL_PATH=./drowsy_model.h5
```

Expected result:

```text
When AWS is reachable, route is remote.
When AWS is stopped or internet is unavailable, route falls back to local.
```

## 6. Presentation Talking Points

```text
The project detects driver drowsiness using face landmarks and a CNN model.
It supports hybrid inference: AWS when online, local model when offline.
It includes an auto-retraining pipeline that trains a candidate model and promotes it if it passes metrics.
It saves drowsy events in SQLite, shows them in a dashboard, and can upload drowsy frames to S3.
Docker Compose is used to run the API and retraining job consistently across machines.
```
