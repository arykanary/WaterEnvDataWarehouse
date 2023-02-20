""""""
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.exceptions import AirflowSkipException

from datetime import datetime
from meteostat import Point, Daily

from common.functions import get_exe_date


# Meta (these should be set for every specific DAG)
SERVICE = ''
LOCATION = ''
QUANTITY = "AverageTemperature", "Precipitation", "WindDirection", "WindSpeed", "AirPressure"


def _get_location_values(ti, **kwargs):
    """Get the location values necessary for the rest of the workflow"""
    exec_date = get_exe_date(kwargs).strftime("%Y-%m-%d")
    pg_hook = PostgresHook(postgres_conn_id='environment_env_db_conn')
    sql=f"SELECT * FROM environment_data.locations WHERE loc_name='{LOCATION}';"
    
    return pg_hook.get_pandas_df(sql=sql).T.squeeze('columns').to_dict()


def _get_quantity_values(ti, **kwargs):
    """Get the quantity values necessary for the rest of the workflow"""
    sql_quantity = "', '".join(QUANTITY)

    pg_hook = PostgresHook(postgres_conn_id='environment_env_db_conn')
    sql=f"SELECT * FROM environment_data.quantities WHERE quantity_name IN ('{sql_quantity}');"

    df = pg_hook.get_pandas_df(sql=sql)
    df.index = QUANTITY
    return df.T.to_dict()


def _check_dwh(ti, **kwargs):
    """Check if data is not already in the data warehouse"""
    quan_xcom = ti.xcom_pull(task_ids='get_quantity_values', key='return_value')
    loc_xcom = ti.xcom_pull(task_ids='get_location_values', key='return_value')
    exec_date = get_exe_date(kwargs)

    pg_hook = PostgresHook(postgres_conn_id='environment_env_db_conn')
    for k, v in quan_xcom.items():   
        sql=f"SELECT COUNT(measurement_id) FROM environment_data.measurements "\
            f"WHERE measure_year={exec_date.year} AND measure_month={exec_date.month} AND measure_day={exec_date.day} "\
            f"AND location_id={loc_xcom['location_id']} AND quantity_id={v['quantity_id']};"

        if not pg_hook.get_records(sql=sql)[0][0] < 1:
            raise AirflowSkipException(f"Data already available for {get_exe_date(kwargs).strftime('%Y-%m-%d')}")


def _get_mst_data(ti, **kwargs):
    """"""
    quan_xcom = ti.xcom_pull(task_ids='get_quantity_values', key='return_value')
    loc_xcom = ti.xcom_pull(task_ids='get_location_values', key='return_value')
    exec_date = get_exe_date(kwargs)

    date = datetime(exec_date.year, exec_date.month, exec_date.day)
    result = Daily(Point(loc_xcom['lattitude'], loc_xcom['longitude']), date, date).fetch().to_dict()
    result = {k: list(v.values())[0] for k, v in result.items()}
    
    ti.xcom_push(key='year',    value=date.year)
    ti.xcom_push(key='month',   value=date.month)
    ti.xcom_push(key='day',     value=date.day)
    ti.xcom_push(key='loc_id',  value=loc_xcom['location_id'])
    
    # 'year': 2023, 'month': 2, 'day': 7, 'tavg': -0.1, 'tmin': -5.0, 'tmax': 4.7, 'prcp': 0.0, 'snow': nan, 'wdir': 112.0, 'wspd': 6.3, 'wpgt': 14.8, 'pres': 1039.8, 'tsun': nan
    # "AverageTemperature", "Precipitation", "WindDirection", "WindSpeed", "AirPressure"
    ti.xcom_push(key='quan_id_tavg', value=quan_xcom['AverageTemperature']['quantity_id'])
    ti.xcom_push(key='tavg',   value=result['tavg'])
    ti.xcom_push(key='quan_id_prcp', value=quan_xcom['Precipitation']['quantity_id'])
    ti.xcom_push(key='prcp',   value=result['prcp'])
    ti.xcom_push(key='quan_id_wdir', value=quan_xcom['WindDirection']['quantity_id'])
    ti.xcom_push(key='wdir',   value=result['wdir'])
    ti.xcom_push(key='quan_id_wspd', value=quan_xcom['WindSpeed']['quantity_id'])
    ti.xcom_push(key='wspd',   value=result['wspd'])
    ti.xcom_push(key='quan_id_pres', value=quan_xcom['AirPressure']['quantity_id'])
    ti.xcom_push(key='pres',   value=result['pres'])
