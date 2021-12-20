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
# 로컬에서는
# https://chromedriver.storage.googleapis.com/index.html?path=96.0.4664.45/ 에서 직접 다운로드

# 1.4 필요 라이브러리 설치
pip install xlrd 
apt-get install xvfb
pip install pyvirtualdisplay 
pip install webdriver_manager
pip install tqdm
pip install selenium
pip install bs4