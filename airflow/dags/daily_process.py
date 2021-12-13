from datetime import timedelta

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

with DAG(
    dag_id="daily_process",
    description="crawling, clustering, summary, tts",
    start_date=days_ago(2),
    schedule_interval="0 17 * * *",
    tags=["my_tags"]
) as dag:

    t1 = BashOperator(
        task_id="crawling",
        bash_command="python ../../crawling/daum_news_title_crawling.py",
        owner="admin",                   # 이 작업의 오너입니다. 보통 작업을 담당하는 사람 이름을 넣습니다.
        retries=3,                       # 이 태스크가 실패한 경우, 3번 재시도 합니다.
        retry_delay=timedelta(minutes=5) # 재시도하는 시간 간격은 5분입니다.
    )

    t2 = BashOperator(
        task_id="finished",
        bash_command="echo finished",
        owner="admin",                   # 이 작업의 오너입니다. 보통 작업을 담당하는 사람 이름을 넣습니다.
        retries=3,                       # 이 태스크가 실패한 경우, 3번 재시도 합니다.
        retry_delay=timedelta(minutes=5) # 재시도하는 시간 간격은 5분입니다.
    )

    t1 >> t2
