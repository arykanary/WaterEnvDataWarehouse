import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def get_dat():
    conn_string = "host='localhost' dbname='weather_env' user='postgres' password='postgres'"
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("""SELECT * FROM environment_data.pivoted;""")
    records = cur.fetchall()
    cur.close()
    df = pd.DataFrame(records, columns=[x[0] for x in cur.description])
    df.index = pd.to_datetime(df[['measure_year', 'measure_month', 'measure_day']].rename({'measure_year': 'year', 'measure_month': 'month', 'measure_day': 'day'}, axis=1))
    return df


def corr(a, b):
    a = (a - np.mean(a))/(np.std(a)*len(a))
    b = (b - np.mean(b))/(np.std(b))
    return np.correlate(a, b)[0]


def transform_df(df, ndays=14, month_num=-10):
    subset = df['arnhem_waterlevel'].values
    gdf = pd.DataFrame([subset[a:b] for a, b in enumerate(range(ndays, subset.shape[0]))]).dropna()
    
    if month_num < 0:
        month_num = gdf.shape[0]-month_num
    val = gdf.loc[month_num]
    cdf = pd.Series([corr(val, x) for _, x in gdf.iterrows()]).abs().sort_values(ascending=False).dropna()

    # cdf_head = cdf[cdf>.99].head(20).dropna()
    cdf_head = cdf.head(100).dropna()
    likely = cdf_head.min()
    ind = cdf_head.index + ndays
    
    result = gdf.loc[[x for x in ind if x in gdf.index]].sort_index().T
    nlikely = result.shape[1]
    title = f'The results of the {ndays} days after the {nlikely} most similar {ndays} day periods (more than {likely:.1%} similarity)'
    return likely, title, result


ndays = 14
df = get_dat()

# minimize to get the 

def minimize(upper=100):
    def trans_wrap(ndays):
        like = 1
        m = 1
        while True:
            try:
                _like, _, _ = transform_df(df, int(ndays), m)
                like *= _like
                print('month:', m, _like)
                m += 1

                if m > 3:  break
            except KeyError:
                break
        return like
    
    high = 1.
    for n in range(1, upper):
        print('days:', n)
        out = trans_wrap(n)
        if out < high:
            result = n
    return result, high


# print(minimize(3))
# exit()

# plot
_, title, result = transform_df(df, ndays)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12, 7))
plt.suptitle(title)

# time
result.plot(ax=ax1)

# stats
sub = result.copy()
sub['mean'] = sub.mean(axis=1)
sub['median'] = sub.median(axis=1)
sub['max'] = sub.max(axis=1)
sub['min'] = sub.min(axis=1)
sub[['mean', 'median', 'max', 'min',]].plot(ax=ax2)

# kde
x = pd.MultiIndex.from_product([result.columns, result.index]).get_level_values(1)
y = result.values.flatten()
ybins = 100
k = gaussian_kde([x, y])
xi, yi = np.mgrid[x.min():x.max():ndays*1j, y.min():y.max():ybins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
ax3.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')

# Hist
y = result.values.flatten()
ax4.hist(y, 30, density=True, histtype='step')

# Save
fig.tight_layout()
fig.savefig('notebooks/similarity.png')
