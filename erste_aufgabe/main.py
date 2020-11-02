import requests
import sqlite3 as sql
import pandas as pd
import numpy as np
from requests.auth import HTTPBasicAuth
from zipfile import ZipFile


def download_data(refresh_old_data):
    if refresh_old_data:
        # Download old data
        url_old = 'https://wwwdb.inf.tu-dresden.de/misc/WS2021/PDS/the.db'
        response = requests.get(url_old, auth=HTTPBasicAuth('', ''))
        with open('the.db', 'wb') as f:
            f.write(response.content)

    # Download new data
    url_new = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/recent/10minutenwerte_TU_01048_akt.zip'
    response = requests.get(url_new)
    with open('new_data.zip', 'wb') as f:
        f.write(response.content)
    with ZipFile('new_data.zip', 'r') as z:
        files = z.namelist()
        open('new_data.txt', 'wb').write(z.read(files[0]))


def main():
    download_data(True)

    # Create dataframe from new data with renamed columns
    df = pd.read_csv(open('new_data.txt'),
                     delimiter=";",
                     header=0,
                     names=['id', 'timestamp', 'qn', 'airpressure', 'temperature', 'temperature_ground', 'humidity',
                            'temperature_dew', 'eor'])
    # Drop irrelevant data
    df = df.query("timestamp > 202010131050 & timestamp < 202010232400")\
           .drop(['qn', 'eor'], axis=1)
    # Append dataframe to existing table
    conn = sql.connect('the.db')
    df.to_sql('dwd', conn, if_exists='append', index=False)

    # Export table 'dwd' as json
    df = pd.read_sql('SELECT * FROM dwd', conn)
    with open('db_updated.json.txt', 'w') as f:
        df.to_json(f)

    # Create dataframe with every 6th timestamp
    times = df[['timestamp']].iloc[::6].reset_index(drop=True)
    # Create dataframe with average of every 6 temperature values
    temps = df[['temperature']].groupby(np.arange(len(df)) // 6).mean()
    # Concatenate columns and create new table 'hourly_avg'
    avg_df = pd.concat([times, temps], axis=1)
    avg_df.to_sql('hourly_avg', conn, if_exists='replace', index=False)

    conn.close()


if __name__ == '__main__':
    main()

