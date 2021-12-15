#!/bin/bash

## 1. crawling

# 1.1 Chrome 설치
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
apt-get update 
apt-get install google-chrome-stable

# 1.2 Chrome 버전 확인
google-chrome --version

# 1.3 chromedriver 다운로드 및 압축 해제 # chromedirver file이 폴더 내에 없거나 깨졌을때 실행
# wget https://chromedriver.storage.googleapis.com/96.0.4664.45/chromedriver_linux64.zip
# unzip chromedriver_linux64.zip

# 1.4 필요 라이브러리 설치
pip install xlrd 
apt-get install xvfb
pip install pyvirtualdisplay 
pip install webdriver_manager
pip install tqdm
pip install selenium
pip install bs4

## 2. serving

# 2.1 fastapi
pip install requests==2.26.0
pip install fastapi==0.70.0
pip install uvicorn==0.15.0
pip install gunicorn==20.0.4
pip install python-dotenv==0.19.1
pip install aiofiles==0.7.0
pip install python-multipart==0.0.5
pip install jinja2==3.0.2
pip install pytest==6.2.5

# 2.2 airflow
pip install apache-airflow==2.2.0


## 3. Summarization Model

# 3.1 data
pip install pyarrow
pip install pandas

# 3.2 model
pip install torch
pip install transformers
pip install wandb
