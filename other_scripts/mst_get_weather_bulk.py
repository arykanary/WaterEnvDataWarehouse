"""
https://meteostat.net/en/


- https://meteostat.net/en/station/06275?t=2000-01-01/2000-01-31
- https://meteostat.net/en/place/de/duisburg?s=D3670&t=2023-01-23/2023-01-30
- https://meteostat.net/en/place/de/koblenz?s=10516&t=2023-01-23/2023-01-30
- https://meteostat.net/en/place/de/mainz?s=D3137&t=2023-01-23/2023-01-30
- https://meteostat.net/en/place/ch/basel?s=06601&t=2023-01-23/2023-01-30
- https://meteostat.net/en/place/ch/koblenz?s=06666&t=2023-01-23/2023-01-30
"""
import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from meteostat import Point, Daily


root = r'other_scripts\data'
start = datetime(1850, 1, 1)
# start = datetime(2023, 2, 1)
end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=1)

qid_dict = {
      'tavg': (6,  'AverageTemperature'),
      'prcp': (7,  'Precipitation'),
      'wdir': (8,  'WindDirection'),
      'wspd': (9,  'WindSpeed'),
      'pres': (10, 'AirPressure')
}

locations = {
      'Arnhem':         (Point(51.9800, 5.9111), 3),
      'Lobith':         (Point(51.8625, 6.1181), 5),
      'Duisburg':       (Point(51.4325, 6.7652), 6),
      'Koblenz(DE)':    (Point(50.3536, 7.5788), 7),
      'Mainz':          (Point(49.9842, 8.2791), 8),
      'Basel':          (Point(47.5584, 7.5733), 9),
      'Koblenz(CH)':    (Point(47.6097, 8.2375), 10),
}

for key, (loc, lid) in locations.items():
    data = Daily(loc, start, end).fetch()
    data.index = pd.MultiIndex.from_arrays(
        [data.index.year, data.index.month, data.index.day],
        names=['measure_year', 'measure_month', 'measure_day']
    )

    for c, (qid, name) in qid_dict.items():
        df = pd.DataFrame(data[c])
        df.columns = ['measure_value']
        df['quantity_id'] = qid
        df['location_id'] = lid
        
        df.to_csv(os.path.join(root, f'{key}_{name}.csv'))
