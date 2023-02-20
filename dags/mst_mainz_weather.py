from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.utils.task_group import TaskGroup

from common.dag_settings import START_DATE, GET_DAG_FREQ
from common import dag_mst_functions


# Meta
dag_mst_functions.SERVICE = 'MST'
dag_mst_functions.LOCATION = 'Mainz'


# DAG
with DAG(
    f"{dag_mst_functions.SERVICE}_{dag_mst_functions.LOCATION}_Weather",
    schedule=GET_DAG_FREQ,
    start_date=START_DATE,
    catchup=True,
    tags=[dag_mst_functions.SERVICE, dag_mst_functions.LOCATION, "Weather"]
) as dag:
    
    # Get parameters
    get_quantity_values = PythonOperator(
        task_id='get_quantity_values',
        python_callable=dag_mst_functions._get_quantity_values,
        provide_context=True
    )

    get_location_values = PythonOperator(
        task_id='get_location_values',
        python_callable=dag_mst_functions._get_location_values,
        provide_context=True
    )

    # Check availability
    check_dwh = PythonOperator(
        task_id='check_date_dwh',
        python_callable=dag_mst_functions._check_dwh,
        provide_context=True
    )

    # Get & Load data
    get_mst_data = PythonOperator(
        task_id='get_mst_data',
        python_callable=dag_mst_functions._get_mst_data,
        # retries=5,
        # retry_delay=timedelta(minutes=2)
    )

    load2db = PostgresOperator(
        task_id='load2db',
        postgres_conn_id='environment_env_db_conn',
        sql='sql/load2db_mst.sql',
    )

    # DAG flow
    [get_quantity_values, get_location_values] >> check_dwh >> get_mst_data >> load2db
    