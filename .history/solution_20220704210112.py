"""
Solution module for syndicate assignment.
"""
from collections import defaultdict
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
from graphplot import bar_with_title, time_series_for_two, \
    unusual_day_plot
from helper import get_count_hourly, summary_hourly_count, \
    daily_count, data_for_count, test_data_for_count, \
    compute_distance, pearson_distance, \
    diff_conclusion, find_extreme_item

# define constant
WEEKDAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday"]


def data_cleansing(count_file, rain_file, temp_file, solar_file):
    """
    Gerenal function for data cleansing.
    Take pedestrian count file, rainfall file, maximum temperature file
    and solar exposure file, join the four files on date, clean the data
    by removing null values and return the joint dataframe.
    """
    # read files
    pedestrian_df = pd.read_csv(count_file)
    rainfall_df = pd.read_csv(rain_file)
    temperature_df = pd.read_csv(temp_file)
    solar_df = pd.read_csv(solar_file)

    # add date_key column in format 'YearMonthMdate'
    month_dict = {"January": 1, "February": 2, "March": 3,
                  "April": 4, "May": 5, "June": 6, "July": 7, "August": 8,
                  "September": 9, "October": 10, "November": 11,
                  "December": 12}
    pedestrian_df["date_key"] = pedestrian_df["Year"].astype(str) \
        + pedestrian_df["Month"].map(month_dict).astype(str) \
        + pedestrian_df["Mdate"].astype(str)

    # join rainfall dataframe based on column date_key
    rainfall_df["date_key"] = rainfall_df["Year"].astype(str) + \
        rainfall_df["Month"].astype(str) + rainfall_df["Day"].astype(str)
    rainfall_df.drop([
        "Product code",
        "Bureau of Meteorology station number",
        "Period over which rainfall was measured (days)",
        "Quality", "Year", "Month", "Day"], axis=1, inplace=True)
    rainfall_dict = list(rainfall_df.set_index(rainfall_df.date_key).
                         drop("date_key", axis=1).to_dict().values())[0]

    # join temperature dataframe based on column date_key
    temperature_df["date_key"] = temperature_df["Year"].astype(str) \
        + temperature_df["Month"].astype(str) \
        + temperature_df["Day"].astype(str)

    temperature_df.drop([
        "Product code", "Bureau of Meteorology station number",
        "Year", "Month", "Day", "Days of accumulation of maximum temperature",
        "Quality"], axis=1, inplace=True)

    temperature_dict = list(temperature_df.set_index(
        temperature_df.date_key).drop(
            "date_key", axis=1).to_dict().values())[0]

    # join solar dataframe based on column date_key
    solar_df["date_key"] = solar_df["Year"].astype(str) \
        + solar_df["Month"].astype(str) + solar_df["Day"].astype(str)
    solar_df.drop([
        "Product code", "Bureau of Meteorology station number",
        "Year", "Month", "Day"], axis=1, inplace=True)
    solar_dict = list(solar_df.set_index(
        solar_df.date_key).drop(
            "date_key", axis=1).to_dict().values())[0]

    pedestrian_df["Rainfall amount (millimetres)"] \
        = pedestrian_df["date_key"].map(rainfall_dict)
    pedestrian_df["Maximum temperature (Degree C)"] \
        = pedestrian_df["date_key"].map(temperature_dict)
    pedestrian_df["Daily global solar exposure (MJ/m*m)"] \
        = pedestrian_df["date_key"].map(solar_dict)
    dataframe = pedestrian_df

    # remove any row with null value in rainfall
    # amount, max temperature, solar exposure
    xcut = (dataframe["Rainfall amount (millimetres)"].isnull()
            | dataframe["Maximum temperature (Degree C)"].isnull()
            | dataframe["Daily global solar exposure (MJ/m*m)"].isnull())
    dataframe = dataframe[xcut == False].reset_index(drop=True)
    dataframe.Date_Time = pd.to_datetime(dataframe.Date_Time)

    return dataframe


def pedestrian_stats(dataframe, year, hours):
    """
    Generalized function of question 1.
    Take a dataframe of pedestrian counts, return the relevent statistics
    (median, mean, max, min) of the pedestrian count at the given year and
    hours. Return the statistics in the tabulated form.
    """
    new_df = pd.DataFrame(columns=['time', 'median', 'mean', 'max', 'min'])

    for time in hours:
        hourly_df = get_count_hourly(dataframe, year, time)
        result = summary_hourly_count(hourly_df, time)
        new_df = pd.concat([new_df, result], ignore_index=True)

    return tabulate(new_df, headers='keys', tablefmt='psql')


def pedestrian_scatter(dataframe, year1, year2, x_axis):
    """
    General function of question 2, 3 and 4.
    Plot the scatter plots of pedestrian counts versus the column indicated
    by x_axis, for year1 and year2.
    The scatter plot has y-axis as the hourly_counts of pedestrians, x-aixs
    is treated as a variable, could be rainfall, temperature or solar.
    """
    daily_overall_2021 = pd.DataFrame(
        dataframe[dataframe.Year == year1].
        groupby(dataframe.Date_Time.dt.strftime('%y-%m-%d')).
        agg({'Hourly_Counts': 'sum', x_axis: 'mean'}))
    daily_overall_2022 = pd.DataFrame(
        dataframe[dataframe.Year == year2].
        groupby(dataframe.Date_Time.dt.strftime('%y-%m-%d')).
        agg({'Hourly_Counts': 'sum', x_axis: 'mean'}))

    daily_overall_2021.plot.scatter(
        x=x_axis, y="Hourly_Counts",
        title=x_axis + " vs pedestrian in " + str(year1))
    plt.tight_layout()
    plt.savefig(str(year1) + '_scatter_plot_' + x_axis[:20] + '.png')

    daily_overall_2022.plot.scatter(
        x=x_axis, y="Hourly_Counts",
        title=x_axis + " vs pedestrian in " + str(year2))
    plt.tight_layout()
    plt.savefig(str(year2) + '_scatter_plot_' + x_axis[:20] + '.png')


def pedestrian_hist(dataframe, year1, year2):
    """
    General function of question 5.
    Plot the histogram (bar plot) for mean daily overall pedestrian counts for
    each day of week. With pedestrian data from df, generate and save one plot
    each for year1 and year2.
    """
    df1 = pd.DataFrame(dataframe[dataframe.Year == year1].groupby(
            [dataframe.Date_Time.dt.strftime('%y-%m-%d'), dataframe.Day]).
            agg({'Hourly_Counts': 'sum'})).groupby('Day').mean()
    df1 = daily_count(df1)

    title1 = "Mean daily overall pedestrian count for each day of week in " \
        + str(year1)
    file1 = str(year1) + "_busy_daily.png"
    bar_with_title(df1, title1, "Day of week",
                   "mean daily overall pesdestrain count", file1)

    df2 = pd.DataFrame(dataframe[dataframe.Year == year2].groupby(
        [dataframe.Date_Time.dt.strftime('%y-%m-%d'),
         dataframe.Day]).agg({'Hourly_Counts': 'sum'})).groupby('Day').mean()
    df2 = daily_count(df2)

    title2 = "Mean daily overall pedestrian count for each day of week in " + \
        str(year2)
    file2 = str(year2) + "_busy_daily.png"
    bar_with_title(df2, title2, "Day of week",
                   "mean daily overall pesdestrain count", file2)


def sensor_hist(dataframe, year, start_no, end_no):
    """
    General function of question 6.
    Plot histogram(bar plot) for mean daily overall pedestrian count for
    sensors from start_no to end_no. Data
    """
    new_df = pd.DataFrame(pd.DataFrame(dataframe[
        (dataframe.Year == year) &
        (dataframe.Sensor_ID.isin(range(start_no, end_no + 1)))]
        .groupby([
            dataframe.Sensor_ID,
            dataframe.Date_Time.dt.strftime('%y-%m-%d')])
        .agg({'Hourly_Counts': 'sum'}))).groupby('Sensor_ID').mean()
    title = "Mean daily overall pedestrian count for sensor 1-20 in " + \
        str(year)
    file = str(year) + "_busy_sensor.png"
    bar_with_title(new_df, title, "sensor ID",
                   "mean daily overall pesdestrain count", file)


def pedestrian_hist_rain(dataframe, year1, year2):
    """
    General function of question 7.
    Plot the histogram (bar plot) for mean daily overall pedestrian counts
    for each raining day of week. With pedestrian data from df, generate
    and save one plot each for year1 and year2.
    """
    df1 = pd.DataFrame(dataframe[
        (dataframe.Year == year1) &
        (dataframe["Rainfall amount (millimetres)"] > 0)]
        .groupby([
            dataframe.Date_Time.dt.strftime('%y-%m-%d'),
            dataframe.Day])
        .agg({'Hourly_Counts': 'sum'})).groupby('Day').mean()

    df1 = daily_count(df1)

    title1 = "Mean daily overall pedestrian count" + \
        "for each raining day of week in " + str(year1)
    file1 = str(year1) + "_busy_daily_rain.png"
    bar_with_title(df1, title1, "Day of week",
                   "mean daily overall pesdestrain count", file1)

    df2 = pd.DataFrame(dataframe[
        (dataframe.Year == year2) &
        (dataframe["Rainfall amount (millimetres)"] > 0)]
        .groupby([
            dataframe.Date_Time.dt.strftime('%y-%m-%d'),
            dataframe.Day])
        .agg({'Hourly_Counts': 'sum'})).groupby('Day').mean()

    df2 = daily_count(df2)

    title2 = "Mean daily overall pedestrian count for" + \
        "each raining day of week in " + str(year2)
    file2 = str(year2) + "_busy_daily_rain.png"
    bar_with_title(df2, title2, "Day of week",
                   "mean daily overall pesdestrain count", file2)


def pedestrian_hist_rain_temp(dataframe, year1, year2, max_temp):
    """
    General function of question 8.
    Plot the histogram (bar plot) for mean daily overall pedestrian counts
    for each raining, cold day of week. With pedestrian data from df,
    generate and save one plot each for year1 and year2.
    """
    df1 = pd.DataFrame(dataframe[
        (dataframe.Year == year1) &
        (dataframe["Rainfall amount (millimetres)"] > 0) &
        (dataframe["Maximum temperature (Degree C)"] < max_temp)]
        .groupby([
            dataframe.Date_Time.dt.strftime('%y-%m-%d'),
            dataframe.Day]).agg({'Hourly_Counts': 'sum'}))

    df1 = pd.DataFrame(df1[df1.Hourly_Counts != 0]).groupby('Day').mean()
    df1 = daily_count(df1)

    title1 = "Mean daily overall pedestrian count for each " + \
        "cold, raining day of week in " + str(year1)
    file1 = str(year1) + "_busy_daily_rain_cold.png"
    bar_with_title(df1, title1, "Day of week",
                   "mean daily overall pesdestrain count", file1)

    df2 = pd.DataFrame(dataframe[(
        dataframe.Year == year2) &
        (dataframe["Rainfall amount (millimetres)"] > 0) &
        (dataframe["Maximum temperature (Degree C)"] < max_temp)]
        .groupby([
            dataframe.Date_Time.dt.strftime('%y-%m-%d'),
            dataframe.Day]).agg({'Hourly_Counts': 'sum'}))
    df2 = pd.DataFrame(df2[df2.Hourly_Counts != 0]).groupby('Day').mean()
    df2 = daily_count(df2)

    title2 = "Mean daily overall pedestrian count for" + \
        " each cold, raining day of week in " + str(year2)
    file2 = str(year2) + "_busy_daily_rain_cold.png"
    bar_with_title(df2, title2, "Day of week",
                   "mean daily overall pesdestrain count", file2)


def time_series_sensor(dataframe, year1, year2, month):
    """
    General function of question 9.
    Take any month of year1 and year2 from dataframe, generate the time
    series data for all sensors. Identify the sensor with greatest change
    between the same month of the two years by computing the euclidean
    distance. Print the greatest changed sensor id, sensor name and the
    change.
    Plot the bar plot of the two months of the greatest changed sensor.
    """
    df1 = pd.DataFrame(dataframe[
        (dataframe.Year == year1) & (dataframe.Month == month)]
        .groupby([
            dataframe.Sensor_ID, dataframe.Date_Time.dt.strftime('%m-%d')])
        .agg({"Hourly_Counts": "sum"}))
    df2 = pd.DataFrame(dataframe[
        (dataframe.Year == year2) & (dataframe.Month == month)]
        .groupby([
            dataframe.Sensor_ID, dataframe.Date_Time.dt.strftime('%m-%d')])
        .agg({"Hourly_Counts": "sum"}))
    compare_merged = pd.merge(
        left=df1, right=df2, left_on=['Sensor_ID', 'Date_Time'],
        right_on=['Sensor_ID', 'Date_Time'])
    e_distance = defaultdict(float)
    limit = dataframe['Sensor_ID'].max()

    # compute euclidean distance
    for sensor in range(1, limit+1):
        count_2021 = []
        count_2022 = []
        for k in compare_merged.to_dict()['Hourly_Counts_x'].keys():
            if k[0] == sensor:
                count_2021.append(
                    compare_merged.to_dict()['Hourly_Counts_x'][k])
                count_2022.append(
                    compare_merged.to_dict()['Hourly_Counts_y'][k])
        if count_2021:
            e_distance[sensor] = np.linalg.norm(
                np.array(count_2021) - np.array(count_2022))

    # find maximum
    max_sensor = max(e_distance, key=e_distance.get)
    max_change = e_distance[max_sensor]
    max_sensor_name = dataframe[
        dataframe.Sensor_ID == max_sensor]['Sensor_Name'].unique()[0]
    print('Output of question 9: ')
    print("Sensor with the most change is sensor_id = " + str(max_sensor) +
          ", and the greatest change is " +
          str(max_change) + ". The name of the sensor is: " + max_sensor_name)

    # plot and save scatter plot of max sensor
    count_2021_max_sensor = pd.DataFrame(dataframe[
        (dataframe.Sensor_ID == max_sensor) &
        (dataframe.Month == month) & (dataframe.Year == year1)]
        .groupby(dataframe.Date_Time.dt.strftime('%m-%d'))
        .sum()['Hourly_Counts'])
    count_2021_max_sensor.reset_index()
    count_2022_max_sensor = pd.DataFrame(dataframe[
        (dataframe.Sensor_ID == max_sensor) &
        (dataframe.Month == month) & (dataframe.Year == year2)]
        .groupby(dataframe.Date_Time.dt.strftime('%m-%d'))
        .sum()['Hourly_Counts'])
    count_2021_max_sensor.reset_index()

    title1 = "Bar plot for sensor with maximum change in " \
        + month + " " + str(year1)
    file1 = str(year1) + "_max_change_sensor.png"
    bar_with_title(count_2021_max_sensor, title1, "Date",
                   "Total daily overall pesdestrain count", file1)

    title2 = "Bar plot for sensor with maximum change in " \
        + month + " " + str(year2)
    file2 = str(year2) + "_max_change_sensor.png"
    bar_with_title(count_2022_max_sensor, title2, "Date",
                   "Total daily overall pesdestrain count", file2)


def model_for_count(dataframe, sensor_id, start_time, end_time):
    """
    General function for q10.
    Build a linear regression model for pedestrian count of the given
    sensor_id, using data between start_time and end_time.
    For the regression, use the following as predictors:
    - rainfall of the previous day;
    - solar exposure of the previous day;
    - max temp of previous day;
    - pedestrian count from sensor 3 in the past hours
    - get the count of a nearby
    - pedestrian count of sensor 3 the same time yesterday

    Print the final equation of the model.
    Fit the model with last month from the time period, return the accuracy
    metric for each day of the week as a dictionary.
    """
    if sensor_id == 1:
        nearby = 2
    else:
        nearby = sensor_id - 1

    train_data = np.array(dataframe[
        (dataframe.Sensor_ID == sensor_id) &
        (dataframe.Time >= start_time) & (dataframe.Time <= end_time - 1) &
        (dataframe.Year == 2022) &
        (dataframe.Month.isin(["January", "February", "March", "April"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '01-01')
        ].sort_values(by=['Date_Time'])['Hourly_Counts'])
    (rain_prev, solar_prev, temp_prev, sensor3_past1, nearby_past1,
        sensor3_pastday) \
        = data_for_count(dataframe, nearby, sensor_id, start_time, end_time)
    factor = np.concatenate((
        rain_prev.reshape(-1, 1), solar_prev.reshape(-1, 1),
        temp_prev.reshape(-1, 1), sensor3_past1.reshape(-1, 1),
        nearby_past1.reshape(-1, 1),
        sensor3_pastday.reshape(-1, 1)), axis=1)
    target = train_data

    model = LinearRegression().fit(factor, target)
    train_error = model.score(factor, target)
    print(f"coefficient of determination: {train_error}")
    print(f"intercept: {model.intercept_}")
    print(f"coefficients: {model.coef_}")

    result = {}
    # compute test data
    for day in WEEK:
        (rain_prev, solar_prev, temp_prev, sensor3_past1, nearby_past1,
            sensor3_pastday) \
            = test_data_for_count(
                dataframe, nearby, sensor_id, start_time, end_time, day)
        x_test = np.concatenate((
            rain_prev.reshape(-1, 1), solar_prev.reshape(-1, 1),
            temp_prev.reshape(-1, 1), sensor3_past1.reshape(-1, 1),
            nearby_past1.reshape(-1, 1),
            sensor3_pastday.reshape(-1, 1)), axis=1)
        y_test = np.array(dataframe[(
            dataframe.Day == day) & (dataframe.Sensor_ID == 3) &
            (dataframe.Time == 12) & (dataframe.Year == 2022) &
            (dataframe.Month == 'May') &
            (dataframe.Date_Time.dt.strftime('%m-%d') != '05-01')]
            .sort_values(by=['Date_Time'])['Hourly_Counts'])

        y_predict = model.predict(x_test)
        result[day] = mean_squared_error(y_test, y_predict)
    print('Mean squared error of each day of week is: ')
    print(result)
    output = find_extreme_item(result)
    return output


def unusual_day(dataframe):
    """
    Identify three unusual days for sensor with sensor_id in 2022.
    Build regression model with the following as predictors:
    - rainfall of the previous day;
    - solar exposure of the previous day;
    - max temp of previous day;
    - pedestrian count from sensor 3 in the past hours
    - get the count of a nearby
    - pedestrian count of sensor 3 the same time yesterday
    Print the three most unusal day, plot the predictions with actual values
    for the three days.
    """
    nearby = 2

    train_data = np.array(dataframe[
        (dataframe.Sensor_ID == 3) & (dataframe.Year == 2022) &
        (dataframe.Month.isin(
            ["January", "February", "March", "April", "May"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '01-01')]
        .sort_values(by=['Date_Time'])['Hourly_Counts'])
    rain_prev = np.array(dataframe[
        (dataframe.Sensor_ID == 3) & (dataframe.Year == 2022) &
        (dataframe.Month.isin(
            ["January", "February", "March", "April", "May"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '05-31')]
        .sort_values(by=['Date_Time'])['Rainfall amount (millimetres)'])
    solar_prev = np.array(dataframe[
        (dataframe.Sensor_ID == 3) & (dataframe.Year == 2022) &
        (dataframe.Month.isin(
            ["January", "February", "March", "April", "May"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '05-31')]
        .sort_values(by=['Date_Time'])['Daily global solar exposure (MJ/m*m)'])
    temp_prev = np.array(dataframe[
        (dataframe.Sensor_ID == 3) & (dataframe.Year == 2022) &
        (dataframe.Month.isin(
            ["January", "February", "March", "April", "May"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '05-31')]
        .sort_values(by=['Date_Time'])['Maximum temperature (Degree C)'])
    sensor2_past = np.array(dataframe[
        (dataframe.Sensor_ID == nearby) & (dataframe.Year == 2022) &
        (dataframe.Month.isin(
            ["January", "February", "March", "April", "May"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '05-31')]
        .sort_values(by=['Date_Time'])['Hourly_Counts'])
    sensor3_past = np.array(dataframe[
        (dataframe.Sensor_ID == 3) & (dataframe.Year == 2022) &
        (dataframe.Month.isin(
            ["January", "February", "March", "April", "May"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '05-31')]
        .sort_values(by=['Date_Time'])['Hourly_Counts'])

    factors = np.concatenate((
        rain_prev.reshape(-1, 1), solar_prev.reshape(-1, 1),
        temp_prev.reshape(-1, 1), sensor2_past.reshape(-1, 1),
        sensor3_past.reshape(-1, 1)), axis=1)

    model = LinearRegression().fit(factors, train_data)

    new_df = dataframe[
        (dataframe.Sensor_ID == 3) & (dataframe.Time >= 0)
        & (dataframe.Time <= 23) & (dataframe.Year == 2022)
        & (dataframe.Month.isin(
            ["January", "February", "March", "April", "May"]))
        & (dataframe.Date_Time.dt.strftime('%m-%d') != '01-01')]
    new_df["distance"] = abs(train_data - model.predict(factors))
    new_df = new_df.groupby([
        'Month', 'Mdate', 'Sensor_ID', 'Day', 'Year',
        'Rainfall amount (millimetres)', 'Maximum temperature (Degree C)',
        'Daily global solar exposure (MJ/m*m)']) \
        .agg({'distance': 'sum'}).reset_index()
    result = new_df.sort_values(
        ["distance", "Month", "Mdate"], ascending=False) \
        .reset_index().head(3)

    for i in range(3):
        unusual_day_plot(dataframe, result, i, 3, nearby, model)

    return result


def daily_difference(dataframe, sensor1, sensor2):
    """
    For sensor1 and sensor2, compute the time seires in May 2022. Compute the
    euclidean distance between the two time series, print the day with maximum
    and minimum difference.
    """
    sensor3 = dataframe[
        (dataframe.Year == 2022) &
        (dataframe.Month == 'May') &
        (dataframe.Sensor_ID == sensor1)][[
            "Time", "Date_Time", "Hourly_Counts"]]
    sensor3["Date_Time"] = sensor3.Date_Time.dt.strftime('%m-%d')

    sensor9 = dataframe[
        (dataframe.Year == 2022) &
        (dataframe.Month == 'May') &
        (dataframe.Sensor_ID == sensor2)][[
            "Time", "Date_Time", "Hourly_Counts"]]
    sensor9["Date_Time"] = sensor9.Date_Time.dt.strftime('%m-%d')

    compare_merged = pd.merge(
        left=sensor3, right=sensor9,
        left_on=['Time', 'Date_Time'], right_on=['Time', 'Date_Time'])
    e_distance = defaultdict(float)

    # compute euclidean distance
    e_distance = compute_distance(e_distance, compare_merged)
    # find maximum
    diff_conclusion(e_distance, 'Euclidean distance')


def sensor_correlation(dataframe, sensor1, sensor2):
    """
    For sensor1 and sensor2, compute the time seires in May 2022 for weekday
    between 9am - 17pm. Compute the pearson correlation between the two time
    series, print the day with maximum and minimum absolute pearson
    correlation.
    """
    sensor3 = dataframe[(
        dataframe.Year == 2022)
        & (dataframe.Month == 'May') & (dataframe.Sensor_ID == sensor1)
        & (dataframe.Time >= 9) & (dataframe.Time <= 17)
        & (dataframe.Day.isin(WEEKDAY))][[
            "Time", "Date_Time", "Hourly_Counts"]]
    sensor3["Date_Time"] = sensor3.Date_Time.dt.strftime('%m-%d')

    sensor9 = dataframe[(
        dataframe.Year == 2022)
        & (dataframe.Month == 'May') & (dataframe.Sensor_ID == sensor2)
        & (dataframe.Time >= 9) & (dataframe.Time <= 17)
        & (dataframe.Day.isin(WEEKDAY))][[
            "Time", "Date_Time", "Hourly_Counts"]]
    sensor9["Date_Time"] = sensor9.Date_Time.dt.strftime('%m-%d')

    compare_merged = pd.merge(
        left=sensor3, right=sensor9,
        left_on=['Time', 'Date_Time'], right_on=['Time', 'Date_Time'])
    pearson_coef = defaultdict(float)

    # compute euclidean distance
    pearson_coef = pearson_distance(pearson_coef, compare_merged)
    # find maximum
    diff_conclusion(pearson_coef, \
        'Pearson correlation coefficient')


def join_covid_travel(dataframe, file1, file2):
    """
    Function that joins the three dataframe together.
    df is the original pedestrian data frame,
    file1 is the covid-19 cases data,
    file2 is the international traveller data.
    Returns a joint dataframe.
    """

    # create temp df
    df_temp = dataframe
    df_temp['Date_Time'] = pd.to_datetime(
        df_temp['Date_Time']).dt.strftime('%Y-%m-%d')

    # read the covid data file
    covid_df = pd.read_csv(file1)
    covid_df = covid_df[['Date', 'VIC']]
    covid_df['Date'] = pd.to_datetime(
        covid_df['Date']).dt.strftime('%Y-%m-%d')

    # read the international arrival data file
    travel_df = pd.DataFrame(pd.read_excel(file2, 'Data1'))[[
        'Unnamed: 0',
        'Number of movements ;  Short-term Visitors arriving ;']]
    travel_df.rename(columns={
        'Unnamed: 0': 'Date_Time',
        'Number of movements ;  Short-term Visitors arriving ;': 'Arrival'},
        inplace=True)
    travel_df['Date_Time'] = pd.to_datetime(
        travel_df['Date_Time']).dt.strftime('%Y-%m')

    # join covid to pedestrian data frame
    covid_pedestrian_df = pd.merge(
        left=df_temp, right=covid_df,
        left_on='Date_Time', right_on='Date')
    covid_pedestrian_df = covid_pedestrian_df \
        .groupby(covid_pedestrian_df.Date_Time, as_index=False) \
        .agg({'Hourly_Counts': 'sum', 'VIC': 'mean'})
    monthly_overall = pd.DataFrame(
        covid_pedestrian_df.groupby(pd.to_datetime(
            covid_pedestrian_df.Date_Time).dt.strftime('%Y-%m'))
        .agg({'Hourly_Counts': 'mean', 'VIC': 'mean'}))

    # join travel to pedestrian data frame
    monthly_overall = pd.merge(
        left=monthly_overall, right=travel_df,
        left_on='Date_Time', right_on='Date_Time')

    return monthly_overall


def invest_covid_travel(dataframe, file1, file2):
    """
    Investigate relationship between pedestrian count
    and covid-19 confirm cases and international travel restriction.
    Take pedestrian dataframe df, covid data file1, arrival data file2,
    plot and save the time series comparison plots.
    """
    # join dataframe
    monthly_overall = join_covid_travel(dataframe, file1, file2)

    # plot and save arrival vs pedestrian
    time_series_for_two(
        monthly_overall, "Arrival", "Hourly_Counts",
        'Internation arrival monthly', 'Daily pedestrian count',
        "Time serie data for monthly " + \
        "internation arrival and average pedestrian count")

    # plot and save covid-19 vs pedestrian
    time_series_for_two(monthly_overall[
        pd.to_datetime(monthly_overall.Date_Time) < '2021-09-01'],
        "VIC", "Hourly_Counts", 'Average daily covid-19 cases',
        'Average daily pedestrian count',
        "Time serie data for monthly covid-19 cases and " + \
        "average pedestrian count before September 2021", '1')
    time_series_for_two(monthly_overall[
        pd.to_datetime(monthly_overall.Date_Time) >= '2021-09-01'],
        "VIC", "Hourly_Counts", 'Average daily covid-19 cases',
        'Average daily pedestrian count',
        "Time serie data for monthly covid-19 cases and " + \
        "average pedestrian count before September 2021", '2')


def invest_lockdown(dataframe):
    """
    Utilize Victoria lockdown history from 2021-1-1 to 2021-10-21
    to visualize impact of lockdown on pedestrian counts in Melbourne.
    Plot the resulting visualizations and save them.
    """
    # generate array-like dataframe of daily pedestrian counts
    # from 2021-01-01 to 2021-10-21 for plotting heat map
    df_array = dataframe[dataframe['Date_Time'] < '2021-10-21'] \
        .groupby(['Date_Time', 'Time'], as_index=False).agg(sum)
    df_array = pd.DataFrame(
        df_array.groupby('Date_Time', as_index=False).agg(list)[[
            'Date_Time', 'Hourly_Counts']])
    df_heat = pd.DataFrame(list(df_array['Hourly_Counts']),
                           columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23],
                           index=df_array.Date_Time)

    # plot heat map
    fig, (axes_0, axes) = plt.subplots(
        ncols=2, figsize=(8, 8), gridspec_kw={'width_ratios': [1, 7]})
    axes = sns.heatmap(df_heat, cmap="PuBu")
    axes_0.axhspan(
        pd.to_datetime('2021-01-01'),
        pd.to_datetime('2021-02-11'), facecolor='w', alpha=0.3, xmin=0)
    axes_0.axhspan(
        pd.to_datetime('2021-02-12'),
        pd.to_datetime('2021-02-17'), facecolor='r', alpha=0.3)
    axes_0.axhspan(
        pd.to_datetime('2021-02-18'),
        pd.to_datetime('2021-05-26'), facecolor='w', alpha=0.3)
    axes_0.axhspan(
        pd.to_datetime('2021-05-27'),
        pd.to_datetime('2021-06-10'), facecolor='r', alpha=0.3)
    axes_0.axhspan(
        pd.to_datetime('2021-06-11'),
        pd.to_datetime('2021-07-14'), facecolor='w', alpha=0.3)
    axes_0.axhspan(
        pd.to_datetime('2021-07-15'),
        pd.to_datetime('2021-07-20'), facecolor='r', alpha=0.3)
    axes_0.axhspan(
        pd.to_datetime('2021-07-21'),
        pd.to_datetime('2021-08-04'), facecolor='w', alpha=0.3)
    axes_0.axhspan(
        pd.to_datetime('2021-08-05'),
        pd.to_datetime('2021-10-21'), facecolor='r', alpha=0.3)
    axes_0.invert_yaxis()
    axes_0.margins(y=0)
    axes.axes.get_yaxis().set_visible(False)
    axes_0.axes.get_xaxis().set_visible(False)
    axes_0.set_xlabel('Lockdown (red)')
    axes_0.set_ylabel('Date')
    axes_0.set_xlabel('Hours in a day')
    axes.yaxis.tick_right()
    fig.tight_layout()
    axes.title.set_text("Hourly pedestrian counts from 2021-01 to 2021-10")
    axes_0.title.set_text("Lockdown")
    plt.savefig("lockdown_impace_heat_map.png", bbox_inches='tight')

    # time series plot
    df_time_series = dataframe
    df_time_series['Date_Time'] = pd.to_datetime(df_time_series['Date_Time'])
    df_time_series = \
        df_time_series[df_time_series['Date_Time'] < '2021-10-21'] \
        .groupby(df_time_series['Date_Time']).agg({'Hourly_Counts': 'sum'})
    df_time_series.reset_index(inplace=True)

    df_time_series.plot(
        x="Date_Time", y="Hourly_Counts",
        title="Covid-19 cases versus pedestrian " + \
        "(Red = Victoria under lockdown)", figsize=(12, 5))

    plt.savefig("lockdown_impace_time_series.png", bbox_inches='tight')
