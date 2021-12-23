from datetime import timedelta
from pathlib import Path

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator

dag = DAG(
    dag_id="batch_data_processing",
    description="generate data in batch",
    start_date=days_ago(n=2),
    schedule_interval="0 15 * * *",
    tags=["crawling", "clustering", "summary", "tts"]
)


# crawling_title_society = BashOperator(
#     task_id="crawling_title_scoiety",
#     bash_command=f"python {str(Path.cwd().parent)}/crawling/daum_news_title_crawling.py --date $(date '+%Y%m%d') --categories society",
#     retries=3,
#     retry_delay=timedelta(minutes=5),
#     dag=dag
# )

# crawling_article_society = BashOperator(
#     task_id="crawling_article_society",
#     bash_command=f"python {str(Path.cwd().parent)}/crawling/daum_news_crawling_multiprocessing.py --date $(date '+%Y%m%d') --category society",
#     retries=3,
#     retry_delay=timedelta(minutes=5),    
#     dag=dag
# )

root_dir = str(Path.cwd().parent)

crawling_title_digital = BashOperator(
    task_id="crawling_title_digital",
    bash_command=f"python {root_dir}/crawling/daum_news_title_crawling.py --date 20211220 --categories digital --root_dir {root_dir}",
    retries=3,
    retry_delay=timedelta(minutes=5),
    dag=dag
)

crawling_article_digital = BashOperator(
    task_id="crawling_article_digital",
    bash_command=f"python {root_dir}/crawling/daum_news_crawling_multiprocessing.py --date 20211220 --category digital --root_dir {root_dir}",
    retries=3,
    retry_delay=timedelta(minutes=5),    
    dag=dag
)

# crawling_title_society >> crawling_article_society

crawling_title_digital >> crawling_article_digital