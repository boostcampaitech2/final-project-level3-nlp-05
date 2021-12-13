from airflow.models import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash_operator import BashOperator

args = {'owner': 'admin', 'start_date': days_ago(n=1)}
dag  = DAG(dag_id='my_first_dag',
           default_args=args,
           schedule_interval='* * * * *')

t1 = BashOperator(task_id='print_date',
                  bash_command='date',
                  dag=dag)

t2 = BashOperator(task_id='sleep',
                  bash_command='sleep 3',
                  dag=dag)

t3 = BashOperator(task_id='print_whoami',
                  bash_command='whoami',
                  dag=dag)

t1 >> t2 >> t3