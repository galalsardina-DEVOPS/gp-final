# AWS Hybrid + Auto Retraining Guide

This guide uses the fastest demo-friendly AWS setup:

- AWS EC2 runs the remote inference API with Docker.
- Your laptop/WSL runs the edge monitor in hybrid mode.
- S3 stores training data and promoted model artifacts.
- Retraining can run manually, by cron, or from GitHub Actions.

## 0. What You Will Show

```text
Laptop/WSL monitor
  -> if online: call AWS /predict
  -> if offline/AWS fails: use local fallback model

Retraining
  -> labeled frames in data/train_data
  -> train candidate model
  -> promote model to artifacts/production
  -> optionally sync production model to S3
```

## 1. Create S3 Buckets

Use one bucket for the project. Replace the bucket name with a globally unique name:

```bash
export AWS_REGION=eu-north-1
export BUCKET=drowsy-driver-galal-demo

aws s3 mb s3://$BUCKET --region $AWS_REGION
```

Suggested S3 layout:

```text
s3://$BUCKET/train_data/
s3://$BUCKET/models/production/
```

Upload local labeled training data:

```bash
aws s3 sync ./data/train_data s3://$BUCKET/train_data
```

## 2. Create EC2 Server

In AWS Console:

- Open EC2.
- Launch Ubuntu Server 22.04 or 24.04.
- Instance size: `t3.medium` is okay for API demo. Use bigger only for heavy retraining.
- Security Group inbound rules:
  - SSH `22` from your IP only
  - Custom TCP `8000` from your IP for demo
- Download or use your SSH key.

SSH into the server:

```bash
ssh -i your-key.pem ubuntu@EC2_PUBLIC_IP
```

Install Docker and Git:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2 git awscli
sudo usermod -aG docker ubuntu
newgrp docker
```

## 3. Put The Project On EC2

Option A: if the project is on GitHub:

```bash
git clone YOUR_REPO_URL ARK
cd ARK
```

Option B: copy from your laptop:

```bash
scp -i your-key.pem -r /mnt/f/ARK ubuntu@EC2_PUBLIC_IP:/home/ubuntu/ARK
ssh -i your-key.pem ubuntu@EC2_PUBLIC_IP
cd /home/ubuntu/ARK
```

## 4. Configure AWS API Mode

On EC2, create `.env`:

```bash
cp .env.example .env
nano .env
```

Use this for the AWS server:

```env
SERVICE_MODE=local_only
LOCAL_MODEL_BACKEND=keras
LOCAL_MODEL_PATH=./drowsy_model.h5
LOCAL_MODEL_THRESHOLD=0.65

DATASET_PATH=./data/train_data
TRAIN_OUTPUT_ROOT=./artifacts/runs
TRAIN_MODEL_TYPE=simple
TRAIN_BASE_WEIGHTS=none
TRAIN_EPOCHS=2
TRAIN_BATCH_SIZE=8
TRAIN_VALIDATION_SPLIT=0.2
TRAIN_MIN_PROMOTE_ACCURACY=0.01
TRAIN_PROMOTE_DELTA=0.00
```

Start the API:

```bash
docker compose up --build
```

Test from your browser:

```text
http://EC2_PUBLIC_IP:8000/healthz
```

## 5. Configure Laptop Hybrid Mode

On your WSL laptop:

```bash
cd /mnt/f/ARK
nano .env
```

Use:

```env
SERVICE_MODE=hybrid
REMOTE_ENDPOINT_URL=http://EC2_PUBLIC_IP:8000/predict
REMOTE_TIMEOUT_SECONDS=3.0
NETWORK_PROBE_URL=https://aws.amazon.com

LOCAL_MODEL_BACKEND=keras
LOCAL_MODEL_PATH=./drowsy_model.h5
LOCAL_MODEL_THRESHOLD=0.65
```

Run:

```bash
source .venv/bin/activate
python scripts/run_monitor.py --source /mnt/f/ARK/vvv.mp4
```

In the video overlay, `ROUTE:remote` means AWS is being used. If you stop EC2 or disconnect internet, the code falls back to local.

## 6. Retraining On EC2

On EC2, pull training data from S3:

```bash
cd /home/ubuntu/ARK
mkdir -p data/train_data
aws s3 sync s3://$BUCKET/train_data ./data/train_data
```

Run retraining:

```bash
docker compose run --rm retrain
```

If successful, the production model appears at:

```text
artifacts/production/model.keras
artifacts/production/manifest.json
```

Upload the production model to S3:

```bash
aws s3 sync ./artifacts/production s3://$BUCKET/models/production
```

To make the EC2 API use the promoted model, edit `.env`:

```env
LOCAL_MODEL_PATH=./artifacts/production/model.keras
```

Restart:

```bash
docker compose down
docker compose up --build
```

## 7. Automatic Retraining On EC2

Start cron:

```bash
sudo service cron start
crontab -e
```

Add this line, replacing the bucket name:

```bash
0 2 * * * cd /home/ubuntu/ARK && aws s3 sync s3://drowsy-driver-galal-demo/train_data ./data/train_data && docker compose run --rm retrain && aws s3 sync ./artifacts/production s3://drowsy-driver-galal-demo/models/production >> /home/ubuntu/ARK/artifacts/retrain.log 2>&1
```

Check logs:

```bash
tail -n 100 /home/ubuntu/ARK/artifacts/retrain.log
```

## 8. Pull Latest Model To Laptop

Install AWS CLI on WSL if needed:

```bash
sudo apt update
sudo apt install -y awscli
aws configure
```

Sync the latest model:

```bash
cd /mnt/f/ARK
bash scripts/sync_model_from_s3.sh s3://drowsy-driver-galal-demo/models/production
```

Set in `.env`:

```env
LOCAL_MODEL_PATH=./artifacts/production/model.keras
```

## 9. Presentation Script

Say:

```text
I deployed the inference API on AWS.
The edge device runs in hybrid mode.
When internet is available, it sends frames to AWS.
When AWS or internet is unavailable, it automatically falls back to the local model.
For retraining, new labeled frames are stored in S3.
A scheduled retraining job trains a candidate model, evaluates it, promotes it, and publishes the production artifact back to S3.
```

## Official References

- ECR image push uses `aws ecr get-login-password` with `docker login`.
- ECS runs containers from task definitions and services.
- S3 `sync` uploads/downloads changed files between local folders and S3.
- GitHub Actions can use AWS credentials via `aws-actions/configure-aws-credentials`.
