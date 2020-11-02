import sqlite3 as sql
import pandas
import requests
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
    with open('data.zip', 'wb') as f:
        f.write(response.content)
    with ZipFile('data.zip', 'r') as z:
        files = z.namelist()
        open('data.txt', 'wb').write(z.read(files[0]))


def main():
    download_data(True)

    # Create table 'new' in db with new data
    conn = sql.connect('the.db')
    d = pandas.read_csv(open('data.txt'), delimiter=";")
    d.to_sql('new', conn, if_exists='replace', index=False)

    # Insert data from 'new' table into 'dwd' table
    c = conn.cursor()
    statement = '''INSERT INTO dwd
        SELECT MESS_DATUM, STATIONS_ID, TT_10, TM5_10, TD_10, RF_10, PP_10 FROM new
        WHERE MESS_DATUM > 202010131050 AND MESS_DATUM < 202010232400'''
    c.execute(statement)
    c.execute('DROP TABLE new')
    conn.commit()
    conn.close()


if __name__ == '__main__':
    main()

