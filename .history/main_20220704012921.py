from helper import *
from graphplot import *
from solution import *
import pandas as pd
import numpy as np

# files for reading, change the route name or file name if necessary
count_file = "count2021-2022.csv"
rain_file = "rainfall-all-years/IDCJAC0009_086338_1800_Data.csv"
temp_file = "temperature-all-years/IDCJAC0010_086338_1800_Data.csv"
solar_file = "solar-all-years/IDCJAC0016_086338_1800_Data.csv"
covid_file = "cases_daily_state.csv"
travel_file = "340101.xlsx"

#  data cleansing part
df = data_cleansing(count_file, rain_file, temp_file, solar_file)

# answer for question 1
df_q1 = pedestrian_stats(df, 2022, [8, 13, 17])
print(df_q1)

# answer for question 2
pedestrian_scatter(df, 2021, 2022, 'Maximum temperature (Degree C)')

# answer for question 3
pedestrian_scatter(df, 2021, 2022, 'Rainfall amount (millimetres)')

# answer for question 4
pedestrian_scatter(df, 2021, 2022, 'Daily global solar exposure (MJ/m*m)')

#answer for question 5
pedestrian_hist(df, 2021, 2022)

#answer for question 6
sensor_hist(df, 2022, 1, 20)

#answer for question 7
pedestrian_hist_rain(df, 2021, 2022)

#answer for question 8
pedestrian_hist_rain_temp(df, 2021, 2022, 20)

#answer for question 9
time_series_sensor(df, 2021, 2022, 'May')

#answer for question 10
result = model_for_count(df, 3, 12, 13)
print(result)

#answer for question 11
unusual_day(df, 3)

# answer for question 12
daily_difference(df, 3, 9)

# answer for question 13
sensor_correlation(df, 3, 9)

#answer for question 15
invest_covid_travel(df, covid_file, travel_file)
invest_lockdown(df)