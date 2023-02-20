import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import argrelmax
import matplotlib.pyplot as plt
import os

from common import RWSData


def analyse_data_dist(filepath=r'_data\dataset.csv', out_path=r'_results\waterheight_distributions.png',
                      col='Meetwaarde.Waarde_Numeriek'):
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)

    fig, ((ax1,  ax2),  (ax3,  ax4), (ax5, ax6)) = plt.subplots(
        3, 2, dpi=200, facecolor='aliceblue', figsize=(10, 12), tight_layout=True)

    ax1.hist(data[col], bins=100, density=True, histtype='step')
    ax1.set_xlabel('Waterheight [cm]')
    ax1.set_ylabel('Probability')

    ax2.plot(data[col])
    ax2_x = ax2.get_xticks()
    ax2.set_xticks(ax2_x[::3])
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Waterheight [cm]')

    ax3.hist2d(data.index.year.astype(int), data[col], bins=(len(set(data.index.year)),  50), density=True)
    ax3.set_xticks([x for x in ax3.get_xticks() if x % 1 == 0])
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Waterheight [cm]')

    ax4.hist2d(data.index.month.astype(int), data[col], bins=(len(set(data.index.month)), 50), density=True)
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Waterheight [cm]')

    ax5.hist2d(data.index.day.astype(int), data[col], bins=(len(set(data.index.day)),   50), density=True)
    ax5.set_xlabel('Day')
    ax5.set_ylabel('Waterheight [cm]')

    ax6.hist2d(data.index.hour.astype(int), data[col], bins=(len(set(data.index.hour)),  50), density=True)
    ax6.set_xlabel('Hour')
    ax6.set_ylabel('Waterheight [cm]')

    fig.suptitle('Measured Water-height Histograms')
    plt.savefig(out_path)
    plt.clf()


def analyse_data_spec(filepath=r'_data\dataset.csv', out_path=r'_results\waterheight_spectrum.png',
                      col='Meetwaarde.Waarde_Numeriek'):
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    data = data.resample(timedelta(hours=1)).mean()
    data.fillna(inplace=True, method='ffill')

    N = len(data[col])
    dt = 3600
    yf = fft(data[col].values)
    xf = fftfreq(N, dt)[:N // 2]
    psd = dt / N * np.abs(yf[0:N // 2])**2
    # psd = np.convolve(psd, np.ones(100) / 100, mode='valid')

    fig, (ax1,  ax2) = plt.subplots(2, 1, dpi=200, facecolor='aliceblue', figsize=(10, 12), tight_layout=True)
    ax1.plot(xf, psd)
    ax1.set_yscale('log')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('PSD [cm^2*s]')

    ax2.plot(data[col])
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Water-height [cm]')

    fig.suptitle('Measured Water-height Spectral Density')
    plt.savefig(out_path)
    plt.clf()


def analyse_data_history(filepath=r'_data\dataset.csv', out_path=r'_results\waterheight_history.png',
                         col='Meetwaarde.Waarde_Numeriek'):
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    data = data.resample(timedelta(hours=1)).mean()
    data.fillna(inplace=True, method='ffill')

    N = len(data[col])
    dt = 3600
    yf = fft(data[col].values)
    xf = fftfreq(N, dt)[:N // 2]
    psd = dt / N * np.abs(yf[0:N // 2])**2
    # psd = np.convolve(psd, np.ones(100) / 100, mode='valid')

    fig, (ax1,  ax2) = plt.subplots(2, 1, dpi=200, facecolor='aliceblue', figsize=(10, 12), tight_layout=True)
    ax1.plot(xf, psd)
    ax1.set_yscale('log')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('PSD [cm^2*s]')

    ax2.plot(data[col])
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Water-height [cm]')

    fig.suptitle('Measured Water-height Spectral Density')
    plt.savefig(out_path)
    plt.clf()


def prep_data(filepath=r'_data\dataset.csv', out_path=r'_data\dataset_prep.csv'):
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    data = data.resample(timedelta(days=1)).mean()
    data.fillna(inplace=True, method='ffill')
    data.iloc[:] = data.values.round(0)

    data.insert(0, 'Year',   data.index.year)
    data.insert(1, 'Month',  data.index.month)
    data.insert(2, 'Day',    data.index.day)

    data.to_csv(out_path)


if __name__ == '__main__':
    # analyse_data_dist()
    analyse_data_spec()
    # prep_data()
