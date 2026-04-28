# Linux Real Camera Setup

This project can use a real webcam on Linux. Use the host Python method for the most reliable demo. Use the Docker Compose camera profile only if the camera and GUI are available to Docker.

## Option A: Recommended Host Camera Run

Start the API with Docker:

```bash
cd ~/ARK
docker compose up --build
```

Open another terminal and run the camera monitor on the Linux host:

```bash
cd ~/ARK
python3.10 -m venv .venv
source .venv/bin/activate
sudo apt install -y alsa-utils
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python scripts/check_camera.py
python scripts/run_monitor.py --source 0
```

If `check_camera.py` reports camera index `1`, use:

```bash
python scripts/run_monitor.py --source 1
```

## Option B: Docker Compose Camera Profile

Check the camera device:

```bash
ls /dev/video*
```

Allow Docker containers to open the local display:

```bash
xhost +local:docker
```

Run the camera service:

```bash
cd ~/ARK
CAMERA_DEVICE=/dev/video0 docker compose --profile camera run --rm monitor-camera
```

If your camera is `/dev/video1`:

```bash
CAMERA_DEVICE=/dev/video1 docker compose --profile camera run --rm monitor-camera
```

After the demo, you can close the display permission:

```bash
xhost -local:docker
```

## Troubleshooting

If Docker cannot access the webcam, use Option A. Docker Desktop on Linux may not expose hardware devices as directly as native Docker Engine.

If OpenCV cannot open the camera, test Linux camera access:

```bash
sudo apt update
sudo apt install -y v4l-utils
v4l2-ctl --list-devices
```

If the preview window does not open, make sure a graphical desktop session is running and `$DISPLAY` is set:

```bash
echo $DISPLAY
```
