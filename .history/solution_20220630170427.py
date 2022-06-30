from helper import *
from graphplot import *
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# define constant
WEEKDAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# gerenal function for data cleansing
def data_cleansing(count_file, rain_file, temp_file, solar_file):
    # read files
    pedestrain_df = pd.read_csv(count_file)
    rainfall_df = pd.read_csv(rain_file)
    temperature_df = pd.read_csv(temp_file)
    solar_df = pd.read_csv(solar_file)
    # add date_key column in format 'YearMonthMdate'
    month_dict = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
    pedestrain_df["date_key"] = pedestrain_df["Year"].astype(str) + pedestrain_df["Month"].map(month_dict).astype(str) + pedestrain_df["Mdate"].astype(str)
    
    # join rainfall dataframe based on column date_key
    rainfall_df["date_key"] = rainfall_df["Year"].astype(str) + rainfall_df["Month"].astype(str) + rainfall_df["Day"].astype(str)
    rainfall_df.drop(["Product code","Bureau of Meteorology station number", "Period over which rainfall was measured (days)","Quality", "Year", "Month", "Day"],axis = 1, inplace = True)
    rainfall_dict = list(rainfall_df.set_index(rainfall_df.date_key).drop("date_key", axis = 1).to_dict().values())[0]

    # join temperature dataframe based on column date_key
    temperature_df["date_key"] = temperature_df["Year"].astype(str) + temperature_df["Month"].astype(str) + temperature_df["Day"].astype(str)
    temperature_df.drop(["Product code","Bureau of Meteorology station number",  "Year", "Month", "Day", "Days of accumulation of maximum temperature", "Quality"],axis = 1, inplace = True)
    temperature_dict = list(temperature_df.set_index(temperature_df.date_key).drop("date_key", axis = 1).to_dict().values())[0]

    # join solar dataframe based on column date_key
    solar_df["date_key"] = solar_df["Year"].astype(str) + solar_df["Month"].astype(str) + solar_df["Day"].astype(str)
    solar_df.drop(["Product code","Bureau of Meteorology station number",  "Year", "Month", "Day"],axis = 1, inplace = True)
    solar_dict = list(solar_df.set_index(solar_df.date_key).drop("date_key", axis = 1).to_dict().values())[0]

    pedestrain_df["Rainfall amount (millimetres)"] = pedestrain_df["date_key"].map(rainfall_dict)
    pedestrain_df["Maximum temperature (Degree C)"] = pedestrain_df["date_key"].map(temperature_dict)
    pedestrain_df["Daily global solar exposure (MJ/m*m)"] = pedestrain_df["date_key"].map(solar_dict)
    df = pedestrain_df

    # remove any row with null value in rainfall amount, max temperature, solar exposure
    xcut = (df["Rainfall amount (millimetres)"].isnull() | df["Maximum temperature (Degree C)"].isnull() | df["Daily global solar exposure (MJ/m*m)"].isnull())
    df = df[xcut == False].reset_index(drop = True)
    df.Date_Time = pd.to_datetime(df.Date_Time)

    return df


# the generalized function of question 1
def pedestrianStats(dataframe, year, hours):
    new_df = pd.DataFrame(columns=['time', 'median', 'mean', 'max', 'min'])

    for time in hours:
        hourly_df = getCountHourly(dataframe, year, time)
        result = summaryHourlyCount(hourly_df, time)
        new_df = pd.concat([new_df, result], ignore_index=True)

    return new_df


# the generalized function of question 2, 3 and 4
# y-axis is the hourly_counts of pedestrians
# x-aixs is treated as a variable, could be rainfall, temperature and solar
def pedestrianScatter(df, year1, year2, x_axis):
    daily_overall_2021 = pd.DataFrame(df[df.Year == year1].groupby(df.Date_Time.dt.strftime('%y-%m-%d')).agg({'Hourly_Counts':'sum',x_axis:'mean'}))  
    daily_overall_2022 = pd.DataFrame(df[df.Year == year2].groupby(df.Date_Time.dt.strftime('%y-%m-%d')).agg({'Hourly_Counts':'sum',x_axis:'mean'}))  

    daily_overall_2021.plot.scatter(x = x_axis, y = "Hourly_Counts", title = x_axis+" vs pedestrian in "+str(year1))
    plt.savefig(str(year1)+'_scatter_plot'+x_axis+'.png')

    daily_overall_2022.plot.scatter(x = x_axis, y = "Hourly_Counts", title = x_axis+" vs pedestrian in "+str(year2))
    plt.savefig(str(year2)+'_scatter_plot'+x_axis+'.png')