from helper import *
from solution import *
from collections import defaultdict
from graphplot import *
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.interpolate import make_interp_spline
import seaborn as sns


# files for reading, change the route name or file name if necessary
count_file = "count2021-2022.csv"
rain_file = "rainfall-all-years/IDCJAC0009_086338_1800_Data.csv"
temp_file = "temperature-all-years/IDCJAC0010_086338_1800_Data.csv"
solar_file = "solar-all-years/IDCJAC0016_086338_1800_Data.csv"
covid_file = "cases_daily_state.csv"
travel_file = "340101.xlsx"

#  data cleansing part
df = data_cleansing(count_file, rain_file, temp_file, solar_file)


df_covid = pd.read_csv('All time (all active).csv')
df_covid.columns = df_covid.columns.str.replace('Unnamed: 0', 'Date')
df_covid['All active cases'] = df_covid['All active cases'].str.replace(',', '').astype(float)
time = df_covid['Date'].str.len()>5
df_covid = df_covid.loc[time]
df_covid['Date'] = pd.to_datetime(df_covid['Date'],dayfirst=True)
df_covid = df_covid[(df_covid['Date']>='2021-01-01') & (df_covid['Date']<='2022-05-31')]
df_covid.set_index('Date', inplace=True)


df_ped_total = df.groupby([df['Date_Time'].dt.date]).sum()['Hourly_Counts']
df_q15= pd.concat([df_covid, df_ped_total], axis=1, join='inner')
df_q15.index.name = 'Date_Time'


# In and post lockdown analysis for the number of pedestrians
q15_pre = df_q15['2021-01-01':'2021-10-21']
q15_post = df_q15['2021-10-22':]
y_q15_pre = q15_pre['Hourly_Counts']
x_q15_pre = q15_pre['All active cases']
y_q15_post = q15_post['Hourly_Counts']
x_q15_post = q15_post['All active cases']
plt.figure(figsize=(10,5))
plt.scatter(x_q15_pre, y_q15_pre, label='Pre-lockdown')
plt.show()

plt.figure(figsize=(10,5))
plt.scatter(x_q15_post, y_q15_post, label='Post-lockdown')
plt.show()
