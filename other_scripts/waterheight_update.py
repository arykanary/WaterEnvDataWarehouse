import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seasonal
from scipy.fft import fft, fftfreq, rfft, irfft

from common import RWSData, Emailer, df_dict_xlsx, fig2html


def send_waterheight_update(loc=('ARNH', 700021.921999557, 5762290.37468757), days_back=16*7, days_forward=7, n_fits=10):
    """

    Do every day:
    1. Get data of the last week
    2. Fit N n-degree polynomials to the data (y=height, x=n_samples)
    3. Save data and fit parameters
    4. Create fits for future week/day(s)
    5. Send email

    :param loc:
    :param days_back:
    :param days_forward:
    :param n_fits:
    :return:
    """
    rws = RWSData(start=datetime.now() - timedelta(days=days_back), end=datetime.now(), loc=loc)
    data = rws.data_loc_datetime()[0][1]['Meetwaarde.Waarde_Numeriek']
    data.sort_index()
    data.name = 'MeasuredData'
    print('Got the data')

    period = np.asarray(seasonal.periodogram_peaks(data.values, int(0.01 * len(data)), int(len(data.values)/2), 0.005))
    print(period.shape)
    per_arr = np.sum([np.sin(np.pi*np.arange(0, len(data.values))/n[0]) * n[1]**0.5 for n in period], axis=0)

    plt.plot(data.values, label='Data')
    trend_line = seasonal.fit_trend(data.values, 'line')
    plt.plot(trend_line, label='Line')
    per_arr += trend_line
    plt.plot(per_arr, label='Periods')

    plt.legend()
    plt.show()
    exit()

    base = data.index.values.astype(np.int64)
    params = {}
    dframes = [data]

    for n in range(n_fits+1):
        param, *_ = np.polyfit(base, data.values, n, full=True)
        params[n] = dict(enumerate(param))
        fit = np.poly1d(param)
        fit_data = data.copy()
        fit_data.index += timedelta(days=days_forward)
        fit_data.loc[:] = fit(fit_data.index.values.astype(np.int64))
        fit_data.name = f'PolyFitData ({n})'
        dframes.append(fit_data)

    param_df = pd.DataFrame(params)
    param_df.index = param_df.index.map(lambda x: f'Deg {x}')
    param_df.columns = param_df.columns.map(lambda x: f'Fit {x}')

    data_fits = pd.concat(dframes, axis=1)

    last = np.where(~np.isnan(data_fits['MeasuredData']))[0][-1]
    mmm_data = data_fits.iloc[last:, 1:]
    data_fits['Mean'] = mmm_data.mean(axis=1)
    data_fits['Max'] = mmm_data.max(axis=1)
    data_fits['Min'] = mmm_data.min(axis=1)
    data_fits.iloc[:, 1:] += data_fits.iloc[last]['MeasuredData'] - data_fits.iloc[last]['Mean']

    df_dict_xlsx({'Parameters': param_df, 'Data_Fits': data_fits},
                 f'_data\\FitResults\\{datetime.now().strftime("%Y_%m_%d")}.xlsx')

    fig, ax1 = plt.subplots()
    data_fits[['MeasuredData', 'Min', 'Mean', 'Max']].plot(ax=ax1)
    ax1.legend()

    sub_set = data_fits.iloc[[last,  -1]]
    sub_set = sub_set[['MeasuredData', 'Min', 'Mean', 'Max']].round(1)

    fig.set_size_inches(8, 5, forward=True)
    fig.tight_layout()
    # plt.show()
    exit()
    fig.savefig(f'_data\\FitResults\\{datetime.now().strftime("%Y_%m_%d")}.png',
                format='png')
    mail_fig = fig2html(fig)

    worst_case = round(data_fits["Min"].min(), 1)
    current_case = round(data_fits["MeasuredData"].iloc[last], 1)
    em = Emailer()

    subject = f'WaterHeights of {datetime.now().strftime("%d %B %Y")}'
    content = f'''
    Hi All,<br><br>
    Current waterheight is {current_case} cm but  could fall to {worst_case} cm
     somewhere in the coming {days_forward} days.<br><br>
    This is based on polynomial fits up to {n_fits}th degree on the data of last {days_back} days and
     predicted {days_forward} days into the future.<br>
    There is absolutely no guarantee that is will come true.<br>
    Please inspect the figure below to estimate the validity of this prediction.<br><br>
    Happy day,<br>
    Wouter<br><br>
    {sub_set.to_html()}<br><br>
    {mail_fig}
    '''
    recipients = [
        'wouter.kramer@me.com',
        # 'wkramer1993@gmail.com'
    ]
    for recipient in recipients:
        em.sendmail(recipient=recipient, subject=subject, content=content)
    print('Send an update to:', *recipients)


if __name__ == '__main__':
    send_waterheight_update()
