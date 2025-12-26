all projectin apps 
use cd to to choose special project
ex- cd apps, cd voice-backned, cd backend



FINAL FLOW

GitHub (code)
   ↓
EC2 Ubuntu terminal
   ↓
.env file (API keys)
   ↓
systemd service
   ↓
Nginx
   ↓
GitHub auto-deploy

✅ STEP 1: GitHub se project EC2 par lao (1 time)
mkdir -p /home/ubuntu/apps
cd /home/ubuntu/apps
git clone https://github.com/USERNAME/REPO_NAME.git voice-bot


(Ab code EC2 par aa gaya)

cd voice-bot/backend
ls

✅ STEP 2: Python virtual environment banao
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate

✅ STEP 3: ENV FILE banao (API keys yahin jayengi)
🔐 Env file ka sahi place
/etc/voice-bot-backend.env

Command:
sudo nano /etc/voice-bot-backend.env

File ke andar paste karo (example):
OPENAI_API_KEY=xxxx
SARVAM_API_KEY=xxxx
REDIS_URL=xxxx
AWS_ACCESS_KEY_ID=xxxx
AWS_SECRET_ACCESS_KEY=xxxx
AWS_DEFAULT_REGION=us-east-1
ENVIRONMENT=production


Save:

CTRL + O → Enter

CTRL + X

Security:

sudo chmod 600 /etc/voice-bot-backend.env

for api key expose :->

✅ STEP 4: systemd SERVICE banao (MOST IMPORTANT)
Service file:
sudo nano /etc/systemd/system/voice-bot-backend.service

Paste karo:
[Unit]
Description=Voice Bot Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/apps/voice-bot/backend
EnvironmentFile=/etc/voice-bot-backend.env
ExecStart=/home/ubuntu/apps/voice-bot/backend/venv/bin/gunicorn \
-k uvicorn.workers.UvicornWorker \        add workers depend on you
-b 127.0.0.1:8090 \
main:app
Restart=always

[Install]
WantedBy=multi-user.target


Enable + start:

sudo systemctl daemon-reload
sudo systemctl enable voice-bot-backend
sudo systemctl start voice-bot-backend


Check:

sudo systemctl status voice-bot-backend


👉 active (running) hona chahiye

✅ STEP 5: Nginx connect karo (domain → backend)
sudo nano /etc/nginx/sites-available/voice-bot


Paste:

server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8090;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}


Enable + restart:

sudo ln -s /etc/nginx/sites-available/voice-bot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx


Browser test:

http://api.yourdomain.com

✅ STEP 6: HTTPS (optional but recommended)
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx

✅ STEP 7: GitHub AUTO-DEPLOY setup
.github/workflows/deploy.yml
name: Deploy Backend

on:
  push:
    branches: [ main ]
    paths: ['backend/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_KEY }}
          script: |
            cd /home/ubuntu/apps/voice-bot/backend
            git pull origin main
            source venv/bin/activate
            pip install -r requirements.txt
            sudo systemctl restart voice-bot-backend


GitHub Secrets:

EC2_HOST = EC2 Elastic IP

EC2_KEY = .pem file content

🔁 DAILY USE (ab tum kya karoge)

Sirf:

git add .
git commit -m "update"
git push origin main


👉 Deploy automatically ho jayega
👉 Env file safe rahegi
