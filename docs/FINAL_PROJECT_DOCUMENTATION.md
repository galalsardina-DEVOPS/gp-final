# Final Project Documentation

## Project Title

Hybrid Drowsy Driver Detection With Auto Retraining And DevOps Deployment

## Problem

Driver drowsiness is a safety risk. The project detects signs of drowsiness from a camera stream and raises a visual alert when the driver appears sleepy or yawning.

## Objectives

- Detect drowsiness in real time from a camera or video.
- Use face landmarks to calculate eye and mouth behavior.
- Use a CNN model as an additional drowsiness classifier.
- Run locally when offline.
- Use AWS remote inference when internet is available.
- Support automatic retraining from newly labeled frames.
- Package the API and retraining job with Docker Compose.

## System Architecture

```text
Camera / Video
  -> Driver Monitor
  -> Face Mesh landmarks
  -> EAR and MAR calculations
  -> Hybrid classifier
  -> AWS remote model if online
  -> Local fallback model if offline
  -> ALERT or DROWSY result
```

Retraining architecture:

```text
Labeled frames
  -> data/train_data/alert and data/train_data/drowsy
  -> retraining job
  -> candidate model
  -> metric check
  -> promoted production model
```

## Main Components

- `app/main.py`: FastAPI API with `/healthz` and `/predict`.
- `scripts/run_monitor.py`: starts the real-time monitor.
- `scripts/check_camera.py`: checks webcam availability.
- `scripts/label_frames.py`: labels frames as `alert` or `drowsy`.
- `scripts/retrain_once.py`: runs one retraining cycle.
- `src/drowsy_platform/monitor.py`: face landmark logic, EAR, MAR, and overlay.
- `src/drowsy_platform/hybrid.py`: chooses remote AWS or local fallback.
- `src/drowsy_platform/training.py`: trains and promotes a candidate model.
- `docker-compose.yml`: runs API, retraining, and optional Linux camera service.

## Detection Method

The monitor uses MediaPipe Face Mesh to detect face landmarks. It calculates:

- EAR: Eye Aspect Ratio, used to detect closed eyes over time.
- MAR: Mouth Aspect Ratio, used to detect yawning.
- CNN score: model prediction for drowsiness.

The final decision combines temporal eye closure, yawning, and CNN output.

## Hybrid Inference

The project supports three modes:

- `local_only`: always use the local model.
- `remote_only`: always use the remote AWS endpoint.
- `hybrid`: use AWS when reachable, otherwise fall back to local.

Important environment variables:

```env
SERVICE_MODE=hybrid
REMOTE_ENDPOINT_URL=http://EC2_PUBLIC_IP:8000/predict
LOCAL_MODEL_BACKEND=keras
LOCAL_MODEL_PATH=./drowsy_model.h5
```

## Auto Retraining

Retraining uses labeled images in this format:

```text
data/train_data/
  alert/
  drowsy/
```

Run retraining:

```bash
docker compose run --rm retrain
```

If the candidate model passes the configured threshold, it is promoted to:

```text
artifacts/production/model.keras
artifacts/production/manifest.json
```

## AWS Deployment

The practical AWS demo uses:

- EC2 to run the Dockerized API.
- S3 to store training data and promoted model artifacts.
- Cron or GitHub Actions to run scheduled retraining.

The local device runs in hybrid mode and sends frames to the AWS `/predict` endpoint when online.

## DevOps

The project includes:

- Dockerfile for container image creation.
- Docker Compose for local API and retraining.
- GitHub Actions CI workflow.
- GitHub Actions retraining workflow.
- AWS deployment workflow for ECR image publishing.

## Demo Plan

1. Start API with Docker Compose.
2. Open `/healthz` to prove the API is running.
3. Run the real-time monitor with camera.
4. Show `ALERT` and `DROWSY` overlay.
5. Run `docker compose run --rm retrain`.
6. Show the promoted model in `artifacts/production`.
7. Explain hybrid mode with AWS endpoint and local fallback.

## Limitations

- Webcam access is easier from the host OS than inside Docker.
- The demo dataset is small, so retraining is for pipeline demonstration, not a production-grade model.
- More real-world data is needed for high accuracy across lighting, camera angles, and driver differences.

## Future Work

- Add a larger labeled dataset.
- Add model registry and version tracking.
- Add automatic upload of new camera samples to S3.
- Add alert sound and dashboard logging.
- Deploy remote inference using ECS/Fargate or SageMaker.

