# Web Deployment Guide

## Overview

The project has two deployment pieces:
- **Backend**: FastAPI server (`enhancer/api/server.py`) — handles file processing
- **Frontend**: `web/index.html` — drop-in page for your website

---

## Local test (before deploying)

```powershell
# Install API dependencies
pip install fastapi uvicorn[standard] python-multipart

# Start the API server
python -m enhancer.api.server
# Running at: http://localhost:8000
# Docs at:    http://localhost:8000/docs

# Open web/index.html in your browser
# The page will talk to localhost:8000 automatically
```

---

## Deploying the backend

### Option A: Any Linux VPS / cloud VM (DigitalOcean, Linode, AWS EC2)

Recommended if you want full control. Needs a machine with enough RAM (~4 GB+).

```bash
# On your server
git clone <your-repo> /opt/enhancer
cd /opt/enhancer
pip install -r requirements.txt

# Download checkpoint
wget -O checkpoints/image_sr.pth <your-checkpoint-url>

# Run with gunicorn/uvicorn
pip install gunicorn
uvicorn enhancer.api.server:create_app --factory --host 0.0.0.0 --port 8000

# Or as a systemd service (recommended for production)
```

Systemd service file (`/etc/systemd/system/enhancer.service`):
```ini
[Unit]
Description=Video Enhancer API
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/enhancer
ExecStart=/usr/local/bin/uvicorn enhancer.api.server:create_app --factory --host 0.0.0.0 --port 8000
Restart=always
Environment=CHECKPOINT_DIR=/opt/enhancer/checkpoints
Environment=FEEDBACK_DIR=/opt/enhancer/feedback_data
Environment=ALLOW_FEEDBACK=true

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable enhancer
systemctl start enhancer
```

### Option B: Render.com (free tier, easiest)

1. Push your project to GitHub
2. Go to render.com → New Web Service
3. Connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn enhancer.api.server:create_app --factory --host 0.0.0.0 --port $PORT`
6. Add environment variables: `CHECKPOINT_DIR`, `FEEDBACK_DIR`, `ALLOW_FEEDBACK`

Note: Free tier sleeps after inactivity. Use paid tier for production.

### Option C: Railway.app

Similar to Render. Add a `railway.toml`:
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn enhancer.api.server:create_app --factory --host 0.0.0.0 --port $PORT"
```

### Option D: Docker (any cloud)

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV CHECKPOINT_DIR=/app/checkpoints
ENV FEEDBACK_DIR=/app/feedback_data
EXPOSE 8000
CMD ["uvicorn", "enhancer.api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t enhancer .
docker run -p 8000:8000 -v ./checkpoints:/app/checkpoints enhancer
```

---

## Deploying the frontend

### If you have an existing website

1. Copy `web/index.html` into your site
2. Change the `API_BASE` constant at the top of the `<script>` tag to your server URL:
   ```js
   const API_BASE = 'https://your-api-server.com';
   ```
3. That's it — the page is self-contained (no build step, no npm)

### CORS note

The API already has CORS enabled for all origins (`allow_origins=["*"]`).
In production, tighten this to your domain:

```python
# In server.py, change:
allow_origins=["https://yourwebsite.com"]
```

### HTTPS

Your server must be behind HTTPS in production (required by browsers for file uploads).
Use Nginx + Let's Encrypt, or let your cloud provider (Render/Railway/etc.) handle it.

Nginx config snippet:
```nginx
server {
    listen 443 ssl;
    server_name api.yourwebsite.com;

    ssl_certificate     /etc/letsencrypt/live/api.yourwebsite.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourwebsite.com/privkey.pem;

    # Increase upload limit for videos
    client_max_body_size 500M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;   # allow long processing times
    }
}
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT_DIR` | `./checkpoints` | Where model .pth files live |
| `FEEDBACK_DIR` | `./feedback_data` | Where feedback pairs are stored |
| `ALLOW_FEEDBACK` | `true` | Set to `false` to disable feedback collection |

---

## File size limits

Large videos need increased limits:
- Nginx: `client_max_body_size 500M`
- FastAPI/uvicorn: no built-in limit (handled by Nginx)
- Render/Railway: check their docs for upload limits (free tiers often cap at 100MB)

---

## Feedback loop in production

1. Users enhance images on your website
2. They click 👍/👎 — feedback stored in `FEEDBACK_DIR` on your server
3. When you have 10+ accepted pairs, trigger training:
   ```bash
   curl -X POST https://your-api-server.com/feedback/train
   ```
4. Model updates automatically and saves new checkpoint
5. All future enhancements use the improved model

Check feedback status anytime:
```bash
curl https://your-api-server.com/feedback/status
```
