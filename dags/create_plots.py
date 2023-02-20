from airflow import DAG
from airflow.operators.python import PythonOperator

from common.dag_settings import START_DATE
from common import dag_car_functions


# DAG
with DAG(
    f'Create_Weekly_Plots',
    schedule=None,
    # schedule='@weekly',
    start_date=START_DATE,
    catchup=False,
    tags=['Results', 'Plots']
) as dag:
    
    get_combined_view = PythonOperator(
        task_id='get_combined_view', 
        python_callable=dag_car_functions._get_combined_view,
        provide_context=True
    )
    
    update_plots = PythonOperator(
        task_id='update_plots', 
        python_callable=dag_car_functions._update_plots,
        provide_context=True
    )

    get_combined_view >> update_plots