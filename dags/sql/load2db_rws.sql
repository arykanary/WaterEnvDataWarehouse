INSERT INTO environment_data.measurements (
            measure_year, measure_month, measure_day, measure_value, location_id, quantity_id
        )
        VALUES (
            {{ ti.xcom_pull(task_ids='get_import_data', key='year') }},
            {{ ti.xcom_pull(task_ids='get_import_data', key='month') }},
            {{ ti.xcom_pull(task_ids='get_import_data', key='day') }},
            {{ ti.xcom_pull(task_ids='get_import_data', key='value') }},
            {{ ti.xcom_pull(task_ids='get_import_data', key='loc_id') }},
            {{ ti.xcom_pull(task_ids='get_import_data', key='quan_id') }}
        );