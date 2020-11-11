import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def iqr_filter(data):
    q25 = data['temperature'].quantile(.25)
    q75 = data['temperature'].quantile(.75)
    iqr = q75 - q25

    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr

    # Replace values that dont meet condition with NaN
    cond = (data['temperature'] > lower) & (data['temperature'] < upper)
    data['temperature'].where(cond, inplace=True)


def z_score_filter(data):
    z_std = data.copy()

    # Calculate mean and standard deviation of data (ddof=0 so the divisor is N instead of N-1)
    mean = data['temperature'].mean()
    sd = data['temperature'].std(ddof=0)

    # Calculate z-standardized data
    z_std['temperature'] = (data['temperature'] - mean) / sd
    # Replace values that dont meet condition with NaN
    cond = (z_std['temperature'] > -3) & (z_std['temperature'] < 3)
    data['temperature'].where(cond, inplace=True)


def linear_int(start, end, length):
    return np.linspace(start, end, length + 2)[1:length+1]


def step_int(start, end, length):
    half_length = length // 2
    return np.concatenate([np.repeat(start, half_length), np.repeat(end, length - half_length)])


# Read data into dataframe
df = pd.read_csv('data-cleaning.csv', parse_dates=['timestamp'])

# Replace outliers with NaN
iqr_filter(df)
z_score_filter(df)

# Create dataframe with complete date range from lowest to highest timestamp in df
dr = pd.DataFrame(pd.date_range(min(df['timestamp']), max(df['timestamp']), freq='10min'), columns=['timestamp'])
# Left join date range with df to get a complete list of timestamps with NaN where measurements are missing
df = dr.merge(df, how='left', on='timestamp')
# Get indexes of missing values
mv = np.where(df['temperature'].isnull())[0]
# Get indexes where difference between current and last is greater than 1
split = np.where(np.diff(mv) > 1)[0] + 1
# Split missing values into intervals that have to be interpolated
mv = np.split(mv, split)

df_linear_int = df.copy()
df_step_int = df.copy()

# Do linear and step interpolation for each interval
for interval in mv:
    first = min(interval)
    last = max(interval)

    start_temp = df.iloc[first-1, 1]
    end_temp = df.iloc[last+1, 1]

    df_linear_int.iloc[first:last+1, 1] = linear_int(start_temp, end_temp, len(interval))
    df_step_int.iloc[first:last+1, 1] = step_int(start_temp, end_temp, len(interval))

# Plot examples of missing value intervals
ax = plt.gca()
df_linear_int.iloc[24480:24535].plot(x='timestamp', y='temperature', ax=ax, label='Linear interpolated data')
df_step_int.iloc[24480:24535].plot(x='timestamp', y='temperature', ax=ax, label='Step interpolated data')
df.iloc[24480:24535].plot(x='timestamp', y='temperature', ax=ax, label='Original data')
plt.show()

ax = plt.gca()
df_linear_int.iloc[9995:10005].plot(x='timestamp', y='temperature', ax=ax, label='Linear interpolated data')
df_step_int.iloc[9995:10005].plot(x='timestamp', y='temperature', ax=ax, label='Step interpolated data')
df.iloc[9995:10005].plot(x='timestamp', y='temperature', ax=ax, label='Original data')
plt.show()
