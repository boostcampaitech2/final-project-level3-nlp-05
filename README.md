# 귀가노니

귀가노니는 어제 있었던 주요 뉴스의 요약된 내용을 오디오 형태로 편하게 들을 수 있는 서비스입니다.

<br>

## 가상환경 생성

```
python -m venv .venv
source .venv/bin/activate
```

<br>

## 필요 패키지 설치

각각의 모듈별 폴더에 정의된 `requirements.sh`을 실행하여 필요한 패키지들을 설치합니다.

```
sh ./requirements.sh
```

<br>

## 데이터 생성 배치 프로세싱

크롤링, 클러스터링, 요약, TTS 모듈을 한번에 실행하여 어제 날짜의 데이터를 생성하는 쉘 스크립트를 실행합니다.  
쉘 스크립트의 argument로 데이터를 생성하고자 하는 날짜를 입력할 수 있습니다.

```
sh ./batch_data_processing.sh 20211220
```

<br>

## 크롤링

다음 뉴스 기사 데이터를 일자 및 카테고리별로 크롤링하여 json 파일로 저장합니다.  
카테고리의 종류는 총 8가지가 존재합니다.
- 사회(society), 정치(politics), 경제(economic), 국제(foreign), 문화(culture), 연예(entertain), 스포츠(sports), IT(digital)

<br>

```
python ./crawling/daum_news_title_crawling.py --date 20211220 --category society
```

```
python ./crawling/daum_news_crawling_multiprocessing.py --date 20211220 --category society
```

<br>

## 뉴스 기사 클러스터링

크롤링된 뉴스 기사들을 토픽별로 클러스터링하여 크기가 큰 상위 3개의 클러스터를 대표하는 뉴스 기사들을 json 파일로 저장합니다.

```
python ./clustering/retriever.py --date 20211220 --category society
```

<br>

## 뉴스 기사 요약

클러스터링된 뉴스 기사에 대한 요약문 및 추출 요약 문장 index 정보를 json 파일로 저장합니다.

```
python ./summary/inference.py --data_dir ./data --date 20211220
```

<br>

## TTS

뉴스 기사의 요약 문장을 mp3 파일로 변환하여 저장합니다.

```
python ./tts/inference_tts.py --date 20211220
```

<br>

## FastAPI

fastapi를 이용해 구현된 웹페이지를 실행합니다.  
웹 서버 실행 후 `http://127.0.0.1:8000`으로 접속이 가능합니다.

```
cd serving
uvicorn app.main:app --reload
```

<br>

## airflow (실험중)

<details>
  <summary>상세 정보</summary>

### 설정
```
cd airflow
export AIRFLOW_HOME=.
sh airflow_setting.sh
```

### 웹서버 실행
```
airflow webserver --port 8080
```

### 웹서버 실행
```
airflow scheduler
```

</detail>
