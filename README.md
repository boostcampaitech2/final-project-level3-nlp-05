# 귀가노니

## 가상환경 생성
```
python -m venv .venv
source .venv/bin/activate
```

<br>

## 패키지 설치
```
sh requirements.sh
```

<br>

## 다음 뉴스 데이터 크롤링

### 기사 제목 및 URL 크롤링

```
python ./crawling/daum_news_title_crawling.py --date 20211214 --categories "society"
```

### 기사 본문 크롤링

```
python ./crawling/daum_news_crawling_multiprocessing.py --date 20211214 --category society --page_count 50 --max_page 1000
```

<br>

## 뉴스 기사 클러스터링

```
python ./clustering/retriever.py --date 20211214 --category society
```

## 뉴스 기사 요약

```
python ./summary/inference.py --date 20211214 --category society
```

<br>

## TTS

```

```

<br>

## 배치 데이터 프로세싱

- 크롤링, 클러스터링, 요약, TTS 배치 단위 실행

```
sh ./batch_data_processing
```

<br>

## FastAPI

```
cd serving
uvicorn app.main:app --reload
```

- `http://127.0.0.1`로 접속 가능

<br>

## airflow

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