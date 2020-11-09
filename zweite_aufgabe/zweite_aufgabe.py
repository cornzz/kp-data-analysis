import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def iqr_filter(data):
    iqr_data = data.copy()

    q25 = data['temperature'].quantile(.25)
    q75 = data['temperature'].quantile(.75)
    iqr = q75 - q25

    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr

    iqr_data.loc[(iqr_data['temperature'] < lower) | (iqr_data['temperature'] > upper), 'temperature'] = np.nan
    return iqr_data


def z_score_filter(data):
    z_score_data = data.copy()
    z_std = data.copy()

    mean = z_score_data['temperature'].mean()
    sd = z_score_data['temperature'].std(ddof=0)

    z_std['temperature'] = (z_score_data['temperature'] - mean) / sd
    z_score_data.loc[(z_std['temperature'] < -3) | (z_std['temperature'] > 3), 'temperature'] = np.nan
    return z_score_data


df = pd.read_csv('data-cleaning.csv')

