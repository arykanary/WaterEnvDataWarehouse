from airflow import DAG
from airflow.operators.python import PythonOperator

from common.dag_settings import START_DATE
from common import dag_car_functions


# DAG
with DAG(
    f'Create_Weekly_Predictions',
    schedule=None,
    # schedule='@weekly',
    start_date=START_DATE,
    catchup=False,
    tags=['Results', 'Plots']
) as dag:
    
    train_model = PythonOperator(
        task_id='train_model', 
        python_callable=dag_car_functions._train_model,
        provide_context=True
    )

    train_model