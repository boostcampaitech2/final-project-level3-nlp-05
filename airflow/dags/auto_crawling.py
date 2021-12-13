from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import time

dag = DAG(
    'dagflow',  # DAG id
    start_date=days_ago(n=1),  # 언제부터 DAG이 시작되는가
    schedule_interval='0/1 * * * * ?',  # 10시와 16시에 하루 두 번 실행
    catchup=False)

def prt1():
    print('print1---start')
    time.sleep(3)
    print('print1---end')

def prt2():
    print('print2---start')
    time.sleep(3)
    print('print2---end')


t1 = PythonOperator(task_id='task_1',
                    python_callable=prt1,
                    provide_context=True,
                    dag=dag)
t2 = PythonOperator(task_id='task_2',
                    python_callable=prt2,
                    provide_context=True,
                    dag=dag)
t1 >> t2