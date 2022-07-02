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

#  data cleansing part
df = data_cleansing(count_file, rain_file, temp_file, solar_file)

# answer for question 1
df_q1 = pedestrianStats(df, 2022, [8, 13, 17])
print(df_q1)

# answer for question 2
pedestrianScatter(df, 2021, 2022, 'Maximum temperature (Degree C)')

# answer for question 3
pedestrianScatter(df, 2021, 2022, 'Rainfall amount (millimetres)')

# answer for question 4
pedestrianScatter(df, 2021, 2022, 'Daily global solar exposure (MJ/m*m)')

#answer for question 5
pedestrianHist(df, 2021, 2022)

#answer for question 6
sensorHist(df, 2022, 1, 20)

#answer for question 7
pedestrianHistRain(df, 2021, 2022)

#answer for question 8
pedestrianHistRainTemp(df, 2021, 2022, 20)

#answer for question 9
timeSeriesSensor(df, 2021, 2022, 'May')

#answer for question 10
result = modelForCount(df, 3, 12, 13)
print(result)

#answer for question 11
unusualDay(df, 3)
