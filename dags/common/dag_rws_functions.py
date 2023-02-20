""""""
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.exceptions import AirflowSkipException

from common.functions import RWSData, get_exe_date


# Meta (these should be set for every specific DAG)
SERVICE = ''
LOCATION = ''
QUANTITY = ''


def _get_location_values(ti, **kwargs):
    """Get the location values necessary for the rest of the workflow"""
    exec_date = get_exe_date(kwargs).strftime("%Y-%m-%d")
    pg_hook = PostgresHook(postgres_conn_id='environment_env_db_conn')
    sql=f"SELECT * FROM environment_data.locations WHERE loc_name='{LOCATION}';"
    
    return pg_hook.get_pandas_df(sql=sql).T.squeeze('columns').to_dict()


def _get_quantity_values(ti, **kwargs):
    """Get the quantity values necessary for the rest of the workflow"""
    pg_hook = PostgresHook(postgres_conn_id='environment_env_db_conn')
    sql=f"SELECT * FROM environment_data.quantities WHERE quantity_name='{QUANTITY}';"

    return pg_hook.get_pandas_df(sql=sql).T.squeeze('columns').to_dict()


def _check_rws_db(ti, **kwargs):
    """Check if data is available in the RWS database"""
    quan_xcom = ti.xcom_pull(task_ids='get_quantity_values', key='return_value')
    loc_xcom = ti.xcom_pull(task_ids='get_location_values', key='return_value')

    rws = RWSData(
        loc=(loc_xcom['rws_code'], loc_xcom['rws_x'], loc_xcom['rws_y']),
        compartment=quan_xcom['rws_compartment'],
        unit=quan_xcom['rws_unit'],
        quantity=quan_xcom['rws_quantity'],
        characteristic=quan_xcom['rws_characteristic'],
        fix_data=True
    )
 
    mdf = rws.check_loc_datetime(start=get_exe_date(kwargs).replace(hour=0, minute=0, second=0),
                                 end=get_exe_date(kwargs).replace(hour=23, minute=59, second=0))
    
    if not mdf['Succesvol'] and mdf['WaarnemingenAanwezig']:
        raise AirflowSkipException(f"Data not available for {get_exe_date(kwargs).strftime('%Y-%m-%d')} Succesvol: {mdf['Succesvol']} & WaarnemingenAanwezig: {mdf['WaarnemingenAanwezig']}")


def _get_rws_data(ti, **kwargs):
    """"""
    quan_xcom = ti.xcom_pull(task_ids='get_quantity_values', key='return_value')
    loc_xcom = ti.xcom_pull(task_ids='get_location_values', key='return_value')
    exec_date = get_exe_date(kwargs)

    rws = RWSData(
        loc=(loc_xcom['rws_code'], loc_xcom['rws_x'], loc_xcom['rws_y']),
        compartment=quan_xcom['rws_compartment'],
        unit=quan_xcom['rws_unit'],
        quantity=quan_xcom['rws_quantity'],
        characteristic=quan_xcom['rws_characteristic'],
        fix_data=True
    )
    
    df = rws.data_loc_datetime(
        start=get_exe_date(kwargs).replace(hour=0, minute=0, second=0),
        end=get_exe_date(kwargs).replace(hour=23, minute=59, second=0)
    )[0][1]
    df =  df.groupby([df.index.date]).mean().round(1)
    
    ti.xcom_push(key='value', value=df['Meetwaarde.Waarde_Numeriek'][0])
    ti.xcom_push(key='year', value=df.index[0].year)
    ti.xcom_push(key='month', value=df.index[0].month)
    ti.xcom_push(key='day', value=df.index[0].day)
    ti.xcom_push(key='loc_id', value=loc_xcom['location_id'])
    ti.xcom_push(key='quan_id', value=quan_xcom['quantity_id'])


def _check_dwh(ti, **kwargs):
    """Check if data is not already in the data warehouse"""
    quan_xcom = ti.xcom_pull(task_ids='get_quantity_values', key='return_value')
    loc_xcom = ti.xcom_pull(task_ids='get_location_values', key='return_value')
    exec_date = get_exe_date(kwargs)

    pg_hook = PostgresHook(postgres_conn_id='environment_env_db_conn')
    sql=f"SELECT COUNT(measurement_id) FROM environment_data.measurements "\
        f"WHERE measure_year={exec_date.year} AND measure_month={exec_date.month} AND measure_day={exec_date.day} "\
        f"AND location_id={loc_xcom['location_id']} AND quantity_id={quan_xcom['quantity_id']};"

    if not pg_hook.get_records(sql=sql)[0][0] < 1:
        raise AirflowSkipException(f"Data already available for {get_exe_date(kwargs).strftime('%Y-%m-%d')}")
