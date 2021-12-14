from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import time

args = {'owner': 'admin', 'start_date': days_ago(n=1)}
dag = DAG(
    dag_id='dagflow',  # DAG id
    default_args=args,
    schedule_interval='* * * * *',
)

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
                    depends_on_past=True,
                    dag=dag)
t2 = PythonOperator(task_id='task_2',
                    python_callable=prt2,
                    depends_on_past=True,
                    dag=dag)
t1 >> t2