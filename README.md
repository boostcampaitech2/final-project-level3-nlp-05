# final-project-level3-nlp-05

final-project-level3-nlp-05 created by GitHub Classroom

## 가상환경 생성
```
python -m venv .venv
source .venv/bin/activate
```
## 패키지 설정
```
sh requirements.sh
```
## airflow

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

## 요약 모델
