from solution import data_cleansing, pedestrian_stats, pedestrian_scatter, \
    pedestrian_hist, sensor_hist, pedestrian_hist_rain, pedestrian_hist_rain_temp, \
    time_series_sensor, model_for_count, unusual_day, daily_difference, \
    sensor_correlation, invest_covid_travel, invest_lockdown

# files for reading, change the route name or file name if necessary
COUNT_FILE = "count2021-2022.csv"
RAIN_FILE = "rainfall-all-years/IDCJAC0009_086338_1800_Data.csv"
TEMP_FILE = "temperature-all-years/IDCJAC0010_086338_1800_Data.csv"
SOLAR_FILE = "solar-all-years/IDCJAC0016_086338_1800_Data.csv"
COVID_FILE = "cases_daily_state.csv"
TRAVEL_FILE = "340101.xlsx"

#  data cleansing part
df = data_cleansing(COUNT_FILE, RAIN_FILE, TEMP_FILE, SOLAR_FILE)

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

# # answer for question 12
# daily_difference(df, 3, 9)

# # answer for question 13
# sensor_correlation(df, 3, 9)

# #answer for question 15
# invest_covid_travel(df, COVID_FILE, TRAVEL_FILE)
# invest_lockdown(df)
