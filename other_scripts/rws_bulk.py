from datetime import datetime, timedelta
import sys
import os
import pandas as pd
sys.path.append(r'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\dags')
from common import functions



root = r'other_scripts\data'
start = datetime(1760, 1, 1)
# start = datetime(2020, 1, 1)
end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

lq_dict = {
    # # Arnhem now
    # 'Arnhem_WaterLevel_now' : ({'loc': ('ARNH', 700021.921999557, 5762290.374687570),
    #                             'compartment': 'OW',
    #                             'unit': 'cm',
    #                             'quantity': "WATHTE",
    #                             'characteristic': 'NAP'},
    #                             4,
    #                             3),
    # 'Arnhem_WaterFlow_now' : ({'loc': ('ARNH', 700021.921999557, 5762290.374687570),
    #                            'compartment': 'OW',
    #                            'unit': 'm3/s',
    #                            'quantity': "Q",
    #                            'characteristic': 'NAP'},
    #                            5,
    #                            3),
    # # Lobith now
    # 'Lobith_WaterLevel_now' : ({'loc': ("LOBI", 713748.841038993, 5748948.95208459),
    #                             'compartment': 'OW',
    #                             'unit': 'cm',
    #                             'quantity': "WATHTE",
    #                             'characteristic': 'NAP'},
    #                             4,
    #                             5),
    # 'Lobith_WaterFlow_now' : ({'loc': ("LOBI", 713748.841038993, 5748948.95208459),
    #                            'compartment': 'OW',
    #                            'unit': 'm3/s',
    #                            'quantity': "Q",
    #                            'characteristic': 'NAP'},
    #                            5,
    #                            5),
    # # Arnhem old
    # 'Arnhem_WaterLevel_old' : ({'loc': ('ARNHM', 700000.776754401, 5762308.24570394),
    #                             'compartment': 'OW',
    #                             'unit': 'cm',
    #                             'quantity': "WATHTE",
    #                             'characteristic': 'NAP'},
    #                             4,
    #                             3),
    # 'Arnhem_WaterFlow_old' : ({'loc': ('ARNHM', 700000.776754401, 5762308.24570394),
    #                            'compartment': 'OW',
    #                            'unit': 'm3/s',
    #                            'quantity': "Q",
    #                            'characteristic': 'NAP'},
    #                            5,
    #                            3),
    # Lobith old
    'Lobith_WaterLevel_old' : ({'loc': ("LOBH", 713748.798641064, 5748949.045232340),
                                'compartment': 'OW',
                                'unit': 'cm',
                                'quantity': "WATHTE",
                                'characteristic': 'NAP'},
                                4,
                                5),
    'Lobith_WaterFlow_old' : ({'loc': ("LOBH", 713748.798641064, 5748949.045232340),
                               'compartment': 'OW',
                               'unit': 'm3/s',
                               'quantity': "Q",
                               'characteristic': 'NAP'},
                               5,
                               5),
}

for key, (value, qid, lid) in lq_dict.items():
    try:
        print(key)
        rws = functions.RWSData(**value)
        df = pd.DataFrame(rws.get_date_range(start, end).round(1).dropna())
        df.index.names = 'measure_year', 'measure_month', 'measure_day'
        df.columns = ['measure_value']
        df['quantity_id'] = qid
        df['location_id'] = lid

        df.to_csv(os.path.join(root, f'{key}.csv'))
    except Exception as e:
        print(f'{e.__class__.__name__} at line {e.__traceback__.tb_lineno} with message {e}')
