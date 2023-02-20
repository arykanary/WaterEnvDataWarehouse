INSERT INTO environment_data.measurements (
    measure_year, measure_month, measure_day, measure_value, location_id, quantity_id
)
VALUES
    ({{ ti.xcom_pull(task_ids='get_mst_data', key='year') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='month') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='day') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='tavg') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='loc_id') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='quan_id_tavg') }}
    ),
    ({{ ti.xcom_pull(task_ids='get_mst_data', key='year') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='month') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='day') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='prcp') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='loc_id') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='quan_id_prcp') }}
    ),
    ({{ ti.xcom_pull(task_ids='get_mst_data', key='year') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='month') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='day') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='wdir') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='loc_id') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='quan_id_wdir') }}
    ),
    ({{ ti.xcom_pull(task_ids='get_mst_data', key='year') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='month') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='day') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='wspd') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='loc_id') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='quan_id_wspd') }}
    ),
    ({{ ti.xcom_pull(task_ids='get_mst_data', key='year') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='month') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='day') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='pres') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='loc_id') }},
     {{ ti.xcom_pull(task_ids='get_mst_data', key='quan_id_pres') }}
    );