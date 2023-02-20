from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.postgres_operator import PostgresOperator

from datetime import datetime, timedelta

from common.dag_settings import START_DATE, GET_DAG_FREQ
from common import dag_rws_functions


# Meta
dag_rws_functions.SERVICE = 'RWS'
dag_rws_functions.LOCATION = 'Arnhem'
dag_rws_functions.QUANTITY = 'WaterLevel'


# DAG
with DAG(
    f'{dag_rws_functions.SERVICE}_{dag_rws_functions.LOCATION}_{dag_rws_functions.QUANTITY}',
    schedule=GET_DAG_FREQ,
    start_date=START_DATE,
    catchup=True,
    tags=[dag_rws_functions.SERVICE, dag_rws_functions.LOCATION, dag_rws_functions.QUANTITY]
) as dag:
    
    # Get parameters
    get_quantity_values = PythonOperator(
        task_id='get_quantity_values',
        python_callable=dag_rws_functions._get_quantity_values,
        provide_context=True
    )

    get_location_values = PythonOperator(
        task_id='get_location_values',
        python_callable=dag_rws_functions._get_location_values,
        provide_context=True
    )

    # Check availability
    check_rws_db = PythonOperator(
        task_id='check_rws_db', 
        python_callable=dag_rws_functions._check_rws_db,
        retries=5,
        retry_delay=timedelta(minutes=2)
    )

    check_dwh = PythonOperator(
        task_id='check_date_dwh',
        python_callable=dag_rws_functions._check_dwh,
        provide_context=True
    )

    # Get & Load data
    get_import_data = PythonOperator(
        task_id='get_import_data',
        python_callable=dag_rws_functions._get_rws_data,
        retries=5,
        retry_delay=timedelta(minutes=2)
    )

    load2db = PostgresOperator(
        task_id='load2db',
        postgres_conn_id='environment_env_db_conn',
        sql='sql/load2db_rws.sql',
    )
    
    # DAG flow
    get_quantity_values >> check_rws_db
    get_location_values >> check_rws_db

    get_quantity_values >> check_dwh
    get_location_values >> check_dwh
    
    [check_rws_db, check_dwh] >> get_import_data >> load2db
    