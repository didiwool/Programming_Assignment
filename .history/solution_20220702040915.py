from helper import *
from graphplot import *
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.interpolate import make_interp_spline


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


# the general function of question 2, 3 and 4
# y-axis is the hourly_counts of pedestrians
# x-aixs is treated as a variable, could be rainfall, temperature and solar
def pedestrianScatter(df, year1, year2, x_axis):
    daily_overall_2021 = pd.DataFrame(df[df.Year == year1].groupby(df.Date_Time.dt.strftime('%y-%m-%d')).agg({'Hourly_Counts':'sum',x_axis:'mean'}))  
    daily_overall_2022 = pd.DataFrame(df[df.Year == year2].groupby(df.Date_Time.dt.strftime('%y-%m-%d')).agg({'Hourly_Counts':'sum',x_axis:'mean'}))  

    daily_overall_2021.plot.scatter(x = x_axis, y = "Hourly_Counts", title = x_axis+" vs pedestrian in "+str(year1))
    plt.savefig(str(year1)+'_scatter_plot_'+x_axis[:20]+'.png')

    daily_overall_2022.plot.scatter(x = x_axis, y = "Hourly_Counts", title = x_axis+" vs pedestrian in "+str(year2))
    plt.savefig(str(year2)+'_scatter_plot_'+x_axis[:20]+'.png')


# the general function of question 5
def pedestrianHist(df, year1, year2):
    df1 = pd.DataFrame(df[df.Year == year1].groupby([df.Date_Time.dt.strftime('%y-%m-%d'), df.Day]).agg({'Hourly_Counts':'sum'})).groupby('Day').mean()
    df1 = dailyCount(df1)

    title1 = "Mean daily overall pedestrain count for each day of week in "+str(year1)
    file1 =  str(year1)+"_busy_daily.png"
    barWithTitle(df1, title1, "Day of week", "mean daily overall pesdestrain count", file1)

    df2 = pd.DataFrame(df[df.Year == year2].groupby([df.Date_Time.dt.strftime('%y-%m-%d'), df.Day]).agg({'Hourly_Counts':'sum'})).groupby('Day').mean()
    df2 = dailyCount(df2)

    title2 = "Mean daily overall pedestrain count for each day of week in "+str(year2)
    file2 =  str(year2)+"_busy_daily.png"
    barWithTitle(df2, title2, "Day of week", "mean daily overall pesdestrain count", file2)


# general function of question 6
def sensorHist(df, year, start_no, end_no):
    new_df= pd.DataFrame(pd.DataFrame(df[(df.Year == year) & (df.Sensor_ID.isin(range(start_no, end_no+1)))].groupby([df.Sensor_ID, df.Date_Time.dt.strftime('%y-%m-%d')]).agg({'Hourly_Counts':'sum'}))).groupby('Sensor_ID').mean()
    title = "Mean daily overall pedestrain count for each day of week in "+str(year)
    file =  str(year)+"_busy_sensor.png"
    barWithTitle(new_df, title, "sensor ID", "mean daily overall pesdestrain count", file)


# the general function of question 7
def pedestrianHistRain(df, year1, year2):
    df1 = pd.DataFrame(df[(df.Year == year1) & (df["Rainfall amount (millimetres)"]>0)].groupby([df.Date_Time.dt.strftime('%y-%m-%d'), df.Day]).agg({'Hourly_Counts':'sum'})).groupby('Day').mean()

    df1 = dailyCount(df1)

    title1 = "Mean daily overall pedestrain count for each raining day of week in "+str(year1)
    file1 =  str(year1)+"_busy_daily_rain.png"
    barWithTitle(df1, title1, "Day of week", "mean daily overall pesdestrain count", file1)

    df2 = pd.DataFrame(df[(df.Year == year2) & (df["Rainfall amount (millimetres)"]>0)].groupby([df.Date_Time.dt.strftime('%y-%m-%d'), df.Day]).agg({'Hourly_Counts':'sum'})).groupby('Day').mean()

    df2 = dailyCount(df2)

    title2 = "Mean daily overall pedestrain count for each raining day of week in "+str(year2)
    file2 =  str(year2)+"_busy_daily_rain.png"
    barWithTitle(df2, title2, "Day of week", "mean daily overall pesdestrain count", file2)


# the general function of question 8
def pedestrianHistRainTemp(df, year1, year2, max_temp):
    df1 = pd.DataFrame(df[(df.Year == year1) & (df["Rainfall amount (millimetres)"]>0) & (df["Maximum temperature (Degree C)"] < max_temp)].groupby([df.Date_Time.dt.strftime('%y-%m-%d'), df.Day]).agg({'Hourly_Counts':'sum'}))
    df1 = pd.DataFrame(df1[df1.Hourly_Counts != 0]).groupby('Day').mean()
    df1 = dailyCount(df1)

    title1 = "Mean daily overall pedestrain count for each cold, raining day of week in "+str(year1)
    file1 =  str(year1)+"_busy_daily_rain_cold.png"
    barWithTitle(df1, title1, "Day of week", "mean daily overall pesdestrain count", file1)

    df2 = pd.DataFrame(df[(df.Year == year2) & (df["Rainfall amount (millimetres)"]>0) & (df["Maximum temperature (Degree C)"] < max_temp)].groupby([df.Date_Time.dt.strftime('%y-%m-%d'), df.Day]).agg({'Hourly_Counts':'sum'}))
    df2 = pd.DataFrame(df2[df2.Hourly_Counts != 0]).groupby('Day').mean()
    df2 = dailyCount(df2)

    title2 = "Mean daily overall pedestrain count for each cold, raining day of week in "+str(year2)
    file2 =  str(year2)+"_busy_daily_rain_cold.png"
    barWithTitle(df2, title2, "Day of week", "mean daily overall pesdestrain count", file2)


# the general function of question 9
def timeSeriesSensor(df, year1, year2, month):
    df1 = pd.DataFrame(df[(df.Year == year1) & (df.Month == month)].groupby([df.Sensor_ID, df.Date_Time.dt.strftime('%m-%d')]).agg({"Hourly_Counts":"sum"}))
    df2 = pd.DataFrame(df[(df.Year == year2) & (df.Month == month)].groupby([df.Sensor_ID, df.Date_Time.dt.strftime('%m-%d')]).agg({"Hourly_Counts":"sum"}))
    compare_merged = pd.merge(left=df1, right=df2, left_on=['Sensor_ID','Date_Time'], right_on=['Sensor_ID','Date_Time'])
    e_distance = defaultdict(float)
    n = df['Sensor_ID'].max()

    # compute euclidean distance
    for sensor in range(1,n+1):
        count_2021 = []
        count_2022 = []
        for k in compare_merged.to_dict()['Hourly_Counts_x'].keys():
            if k[0] == sensor:
                count_2021.append(compare_merged.to_dict()['Hourly_Counts_x'][k])
                count_2022.append(compare_merged.to_dict()['Hourly_Counts_y'][k])
        if count_2021:
            e_distance[sensor] = np.linalg.norm(np.array(count_2021)-np.array(count_2022))
            
    # find maximum
    max_sensor = max(e_distance, key=e_distance.get) 
    max_change = e_distance[max_sensor]
    print("Sensor with the most change is sensor_id = " + str(max_sensor) + ", and the greatest change is " + str(max_change) + ".")



def modelForCount(df, sensor_id, start_time, end_time):
        # q10
    # take variables: 
    # rainfall of the previous day; 
    # solar exposure of the previous day; 
    # max temp of previous day;
    # pedestrain count from sensor 3 in the past hours
    # get the count of a nearby 
    # pedestrain count of sensor 3 the same time yesterday
    if sensor_id == 1:
        nearby = 2
    else:
        nearby = sensor_id - 1

    train_data = np.array(df[(df.Sensor_ID == sensor_id) & (df.Time >= start_time) & (df.Time <= end_time-1) & (df.Year ==2022) &(df.Month.isin(["January", "February", "March", "April"])) & (df.Date_Time.dt.strftime('%m-%d') != '01-01')].sort_values(by = ['Date_Time'])['Hourly_Counts'])
    (rain_prev, solar_prev, temp_prev, sensor3_past1, nearby_past1, sensor3_pastday) = dataForCount(df, nearby, sensor_id, start_time, end_time, 'train')
    x = np.concatenate((rain_prev.reshape(-1,1), solar_prev.reshape(-1,1), temp_prev.reshape(-1,1), sensor3_past1.reshape(-1,1), nearby_past1.reshape(-1,1), sensor3_pastday.reshape(-1,1)), axis = 1)
    y = train_data


    model = LinearRegression().fit(x, y)
    train_error = model.score(x, y)
    print(f"coefficient of determination: {train_error}")
    print(f"intercept: {model.intercept_}")
    print(f"coefficients: {model.coef_}")


    # compute test data
    (rain_prev, solar_prev, temp_prev, sensor3_past1, nearby_past1, sensor3_pastday) = dataForCount(df, nearby, sensor_id, start_time, end_time, 'test')
    x_test = np.concatenate((rain_prev.reshape(-1,1), solar_prev.reshape(-1,1), temp_prev.reshape(-1,1), sensor3_past1.reshape(-1,1), nearby_past1.reshape(-1,1), sensor3_pastday.reshape(-1,1)), axis = 1)
    y_test = np.array(df[(df.Sensor_ID == 3) & (df.Time == 12) & (df.Year ==2022) &(df.Month == 'May' ) & (df.Date_Time.dt.strftime('%m-%d') != '05-01')].sort_values(by = ['Date_Time'])['Hourly_Counts'])

    y_predict = model.predict(x_test)
    # test_error = model.score(x_test, y_test)
    mean_sq_e = mean_squared_error(y_test, y_predict)

    print(f"mean square error of the model: {mean_sq_e}")


    # compute absolute error
    dates = df[(df.Sensor_ID == sensor_id) & (df.Time >= start_time) & (df.Time <= end_time-1) & (df.Year ==2022) &(df.Month == 'May') & (df.Date_Time.dt.strftime('%m-%d') != '05-01')][['Date_Time','Day']]
    absolute_error = {"Monday":0, "Tuesday":0, "Wednesday":0, "Thursday":0, "Friday":0, "Saturday":0, "Sunday":0}
    for i in range(len(y_test)):
        absolute_error[dates.iloc[i,]['Day']] += np.abs(y_test[i] - y_predict[i])

    for k in absolute_error.keys():
        absolute_error[k] = absolute_error[k]/dates[dates['Day']==k].count()[0]

    return absolute_error


def unusualDay(df, sensor_id):
    if sensor_id == 1:
        nearby = 2
    else:
        nearby = sensor_id - 1

    train_data = np.array(df[(df.Sensor_ID == sensor_id) & (df.Year ==2022) &(df.Month.isin(["January", "February", "March", "April", "May"])) & (df.Date_Time.dt.strftime('%m-%d') != '01-01')].sort_values(by = ['Date_Time'])['Hourly_Counts'])
    rain_prev = np.array(df[(df.Sensor_ID == sensor_id) & (df.Year ==2022)&(df.Month.isin(["January", "February", "March", "April", "May"]) ) & (df.Date_Time.dt.strftime('%m-%d') != '05-31')].sort_values(by = ['Date_Time'])['Rainfall amount (millimetres)'])
    solar_prev = np.array(df[(df.Sensor_ID == sensor_id) & (df.Year ==2022) &(df.Month.isin(["January", "February", "March", "April", "May"]) ) & (df.Date_Time.dt.strftime('%m-%d') != '05-31')].sort_values(by = ['Date_Time'])['Daily global solar exposure (MJ/m*m)'])
    temp_prev = np.array(df[(df.Sensor_ID == sensor_id) & (df.Year ==2022) &(df.Month.isin(["January", "February", "March", "April", "May"])) & (df.Date_Time.dt.strftime('%m-%d') != '05-31')].sort_values(by = ['Date_Time'])['Maximum temperature (Degree C)'])
    sensor2_pastday = np.array(df[(df.Sensor_ID == nearby) & (df.Year ==2022) &(df.Month.isin(["January", "February", "March", "April", "May"])) & (df.Date_Time.dt.strftime('%m-%d') != '05-31')].sort_values(by = ['Date_Time'])['Hourly_Counts'])
    sensor3_pastday = np.array(df[(df.Sensor_ID == sensor_id) & (df.Year ==2022) &(df.Month.isin(["January", "February", "March", "April", "May"])) & (df.Date_Time.dt.strftime('%m-%d') != '05-31')].sort_values(by = ['Date_Time'])['Hourly_Counts'])

    x = np.concatenate((rain_prev.reshape(-1,1), solar_prev.reshape(-1,1), temp_prev.reshape(-1,1), sensor2_pastday.reshape(-1,1), sensor3_pastday.reshape(-1,1)), axis = 1)
    y = train_data

    model = LinearRegression().fit(x, y)
    new_df = df[(df.Sensor_ID == 3) & (df.Time >= 0) & (df.Time <= 23) & (df.Year ==2022) & (df.Month.isin(["January", "February", "March", "April", "May"])) & (df.Date_Time.dt.strftime('%m-%d') != '01-01')]
    new_df["distance"] = abs(y - model.predict(x))
    new_df = new_df.groupby(['Month', 'Mdate', 'Sensor_ID', 'Day', 'Year', 'Rainfall amount (millimetres)', 'Maximum temperature (Degree C)', 'Daily global solar exposure (MJ/m*m)']).agg({'distance':'sum'}).reset_index()
    result = new_df.sort_values(["distance", "Month", "Mdate"], ascending=False).reset_index().head(3)
    
    month = result.iloc[0, 1]
    mdate = result.iloc[0, 2].astype(int)
    lastday = mdate - 1
    lastmonth = month

    if mdate == 1:
        if month in ['January', 'May']:
            lastday = 30
        elif month == 'February':
            lastday = 28
        else:
            lastday = 31


    train_data = np.array(df[(df.Sensor_ID == sensor_id) & (df.Year ==2022) & (df.Month == month) & (df.Mdate == mdate)].sort_values(by = ['Date_Time'])['Hourly_Counts'])
    rain_prev = np.array(df[(df.Sensor_ID == sensor_id) & (df.Year ==2022) & (df.Month == lastmonth) & (df.Mdate == lastday)].sort_values(by = ['Date_Time'])['Rainfall amount (millimetres)'])
    solar_prev = np.array(df[(df.Sensor_ID == sensor_id) & (df.Year ==2022) & (df.Month == lastmonth) & (df.Mdate == lastday)].sort_values(by = ['Date_Time'])['Daily global solar exposure (MJ/m*m)'])
    temp_prev = np.array(df[(df.Sensor_ID == sensor_id) & (df.Year ==2022) & (df.Month == lastmonth) & (df.Mdate == lastday)].sort_values(by = ['Date_Time'])['Maximum temperature (Degree C)'])
    sensor2_pastday = np.array(df[(df.Sensor_ID == nearby) & (df.Year ==2022) & (df.Month == lastmonth) & (df.Mdate == lastday)].sort_values(by = ['Date_Time'])['Hourly_Counts'])
    sensor3_pastday = np.array(df[(df.Sensor_ID == sensor_id) & (df.Year ==2022) & (df.Month == lastmonth) & (df.Mdate == lastday)].sort_values(by = ['Date_Time'])['Hourly_Counts'])

    x = np.concatenate((rain_prev.reshape(-1,1), solar_prev.reshape(-1,1), temp_prev.reshape(-1,1), sensor2_pastday.reshape(-1,1), sensor3_pastday.reshape(-1,1)), axis = 1)
    y = train_data
    new_df = df[(df.Sensor_ID == 3) & (df.Year ==2022) & (df.Month == month) & (df.Mdate == mdate)]
    new_df['predicted'] = model.predict(x)
    ax = new_df.plot.scatter(x = 'Time', y = 'Hourly_Counts', c = 'green')
    X_Y_Spline = make_interp_spline(new_df['Time'], new_df['predicted']) 
    X_ = np.linspace(new_df['Time'].min(), new_df['Time'].max(), 500)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_, c= 'lightblue', ax=ax)
    plt.savefig('plot_1.png')
