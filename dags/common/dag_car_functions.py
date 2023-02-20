""""""
from airflow.providers.postgres.hooks.postgres import PostgresHook

from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt


from common.functions import get_exe_date


def _get_combined_view(ti, **kwargs):
    """Check if data is available in the RWS database"""    
    pg_hook = PostgresHook(postgres_conn_id='environment_env_db_conn')
    results = pg_hook.get_pandas_df(
        sql=f"SELECT * FROM environment_data.combined_result;"
    )

    ti.xcom_push(key='results', value=results.to_dict())


def _update_plots(ti, **kwargs):
    """"""
    result_drive = r'/opt/airflow/RWS_Voorspellingen/Plots'
    xcom_results = ti.xcom_pull(task_ids='get_combined_view', key='results')

    # Create and Pivot dataframe
    results = pd.DataFrame(xcom_results)
    new_index = [datetime(year=y, month=m, day=d) for y, m, d in results[['measure_year', 'measure_month', 'measure_day']].values]
    results.index = pd.DatetimeIndex(new_index)
    results = results.pivot(columns=['loc_name', 'quantity_name', 'unit'], values='measure_value')

    # Plot
    for key in results.columns:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8))

        ax1.plot(results.index, results[key])
        ax1.set_title('-'.join(key))

        ax2.hist(results[key], bins=20)
        ax2.set_title('-'.join(key))
        
        key = '_'.join(key[:2])
        fig.savefig(os.path.join(result_drive, f'Plots_{key}.jpg'))
        plt.cla()


def _train_model(ti, **kwargs):
    """"""
    import numpy as np

    from sklearn.neighbors import KNeighborsRegressor
    # from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    # from sklearn.svm import SVR
    # from sklearn.tree import DecisionTreeRegressor
    # from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_validate


    pg_hook = PostgresHook(postgres_conn_id='environment_env_db_conn')
    features = pg_hook.get_pandas_df(
        sql=f"SELECT * FROM environment_data.pivoted;"
    )
    
    # Columns
    # 'measure_year', 'measure_month', 'measure_day'
    # lobith_*, basel_*, duisburg_*, koblenz_*, koblenz_*, mainz_*, arnhem_*, 
    # *_waterlevel, *_waterflow, *_averagetemperature, *__precipitation, *__winddirection, *__windspeed, *__airpressure

    target = features.pop('arnhem_averagetemperature')

    model = KNeighborsRegressor(5)

    cv_scores = cross_validate(model, features, target, cv=5, return_estimator=True)
    # 'fit_time': array([0.00242138, 0.00119114, 0.00112271, 0.001127  , 0.00111079]),
    # 'score_time': array([0.00176048, 0.00109124, 0.00108242, 0.00101519, 0.00116086]),
    # 'estimator': [KNeighborsRegressor(n_neighbors=15), KNeighborsRegressor(n_neighbors=15), KNeighborsRegressor(n_neighbors=15), KNeighborsRegressor(n_neighbors=15), KNeighborsRegressor(n_neighbors=15)],
    # 'test_score': array([ 0.18922556, -0.2566519 , -0.04208293, -0.00208123, -0.66121043])

    scores = cv_scores['test_score']
    max_score = int(np.argmax(scores))
    print(max_score)

    best_model = cv_scores['estimator'][max_score]
    
    print(
        features.iloc[0],
        target.iloc[0],
        best_model.predict(features.iloc[[0]]),
        sep='\n'   
    )
