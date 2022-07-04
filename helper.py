"""
Helper module for syndicate assignment.
"""

# import modules
import pandas as pd
import numpy as np

# define constant
WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
        "Sunday"]


def get_count_hourly(dataframe, year, hour):
    """
    Return the new dataframe of weekday pedestrian for the given year and
    hour, from the given dataframe.
    """
    # raise index error or get the correct dataframe
    if hour not in range(0, 24):
        raise IndexError
    else:
        df_new = dataframe[
            (dataframe['Year'] == year) &
            (dataframe['Time'] == hour) &
            (dataframe['Day'].isin(
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))]
        return df_new


def summary_hourly_count(dataframe, time):
    """
    Get the median, mean, maximum, minimum of 'Hourly_Counts' column for the
    given dataframe, for the time period given by time.
    """
    result = pd.DataFrame(columns=['time', 'median', 'mean', 'max', 'min'])
    dataframe = dataframe.groupby(
        ['Month', 'Mdate'], as_index=False)["Hourly_Counts"].sum()
    median = np.median(dataframe["Hourly_Counts"])
    mean = np.mean(dataframe["Hourly_Counts"])
    max_count = np.max(dataframe["Hourly_Counts"])
    min_count = np.min(dataframe["Hourly_Counts"])

    # store the data as a dict
    data = {
        'time': time,
        'median': median,
        'mean': mean,
        'max': max_count,
        'min': min_count}
    result = result.append(data, ignore_index=True)

    return result


def daily_count(dataframe):
    """
    Data cleansing helper for question 5, 7, 8.
    Sort and clean the data for the given dataframe dataframe by weekday.
    Return the sorted dataframe.
    """
    dataframe = pd.DataFrame(
        dataframe.to_dict()['Hourly_Counts'].items(),
        columns=["Day", "Hourly Counts"])
    dataframe.Day = dataframe.Day.astype("category")
    dataframe.Day = dataframe.Day.cat.set_categories(WEEK)
    dataframe = dataframe.sort_values("Day")
    dataframe = dataframe.set_index(dataframe["Day"], drop=True)
    return dataframe


def data_for_count(dataframe, nearby, sensor_id, start_time, end_time):
    """
    Function for preparing the train data or test data in Q10.
    Take a dataframe dataframe, a nearby sensor nearby, a chosen sensor with
    sensor_id. Generate and return the tuple of the required np.array for
    linear regression.
    """
    # rainfall of yesterday
    rain_prev = np.array(dataframe[
        (dataframe.Sensor_ID == sensor_id) &
        (dataframe.Time >= start_time) &
        (dataframe.Time <= end_time - 1) &
        (dataframe.Year == 2022) &
        (dataframe.Month.isin(["January", "February", "March", "April"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '04-30')
        ].sort_values(by=['Date_Time'])['Rainfall amount (millimetres)'])

    # solar of yesterday
    solar_prev = np.array(dataframe[
        (dataframe.Sensor_ID == sensor_id) &
        (dataframe.Time >= start_time) &
        (dataframe.Time <= end_time - 1) &
        (dataframe.Year == 2022) &
        (dataframe.Month.isin(["January", "February", "March", "April"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '04-30')
        ].sort_values(
            by=['Date_Time'])['Daily global solar exposure (MJ/m*m)'])

    # temperature of yesterday
    temp_prev = np.array(dataframe[
        (dataframe.Sensor_ID == sensor_id) &
        (dataframe.Time >= start_time) &
        (dataframe.Time <= end_time - 1) &
        (dataframe.Year == 2022) &
        (dataframe.Month.isin(["January", "February", "March", "April"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '04-30')
        ].sort_values(by=['Date_Time'])['Maximum temperature (Degree C)'])

    # sensor count of previous hour
    sensor3_past1 = np.array(dataframe[
        (dataframe.Sensor_ID == sensor_id) &
        (dataframe.Time >= start_time - 1) &
        (dataframe.Time <= end_time - 2) &
        (dataframe.Year == 2022) &
        (dataframe.Month.isin(["January", "February", "March", "April"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '01-01')
        ].sort_values(by=['Date_Time'])['Hourly_Counts'])

    # count of a nearby sensor in previous hour
    nearby_past1 = np.array(dataframe[
        (dataframe.Sensor_ID == nearby) &
        (dataframe.Time >= start_time - 1) &
        (dataframe.Time <= end_time - 2) &
        (dataframe.Year == 2022) &
        (dataframe.Month.isin(["January", "February", "March", "April"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '01-01')
        ].sort_values(by=['Date_Time'])['Hourly_Counts'])

    # sensor count of yesterday
    sensor3_pastday = np.array(dataframe[
        (dataframe.Sensor_ID == sensor_id) &
        (dataframe.Time >= start_time) &
        (dataframe.Time <= end_time - 1) &
        (dataframe.Year == 2022) &
        (dataframe.Month.isin(["January", "February", "March", "April"])) &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '04-30')
        ].sort_values(by=['Date_Time'])['Hourly_Counts'])

    return (rain_prev, solar_prev,
            temp_prev, sensor3_past1, nearby_past1, sensor3_pastday)


def test_data_for_count(dataframe, nearby, sensor_id, start_time, end_time,
                        day):
    """
    Function for preparing the test data in Q10.
    Take a dataframe, a nearby sensor nearby, a chosen sensor with sensor_id.
    Generate and return the tuple of the required np.array for linear
    regression for a given day of week 'day'.
    """
    # define yesterday
    dict_day = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7}
    if day == 'Monday':
        yesterday = 'Sunday'
    else:
        for key, value in dict_day.items():
            if value == dict_day[day] - 1:
                yesterday = key

    # compute array of required variables for training model
    rain_prev = np.array(dataframe[
        (dataframe.Day == yesterday) &
        (dataframe.Sensor_ID == sensor_id) &
        (dataframe.Time >= start_time) &
        (dataframe.Time <= end_time - 1) &
        (dataframe.Year == 2022) &
        (dataframe.Month == 'May') &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '05-31')
        ].sort_values(by=['Date_Time'])['Rainfall amount (millimetres)'])
    solar_prev = np.array(dataframe[
        (dataframe.Day == yesterday) &
        (dataframe.Sensor_ID == sensor_id) &
        (dataframe.Time >= start_time) &
        (dataframe.Time <= end_time - 1) &
        (dataframe.Year == 2022) &
        (dataframe.Month == 'May') &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '05-31')
        ].sort_values(by=['Date_Time'])[
            'Daily global solar exposure (MJ/m*m)'])
    temp_prev = np.array(dataframe[
        (dataframe.Day == yesterday) &
        (dataframe.Sensor_ID == sensor_id) &
        (dataframe.Time >= start_time) &
        (dataframe.Time <= end_time - 1) &
        (dataframe.Year == 2022) &
        (dataframe.Month == 'May') &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '05-31')
        ].sort_values(by=['Date_Time'])['Maximum temperature (Degree C)'])
    sensor3_past1 = np.array(dataframe[
        (dataframe.Day == day) &
        (dataframe.Sensor_ID == sensor_id) &
        (dataframe.Time >= start_time - 1) &
        (dataframe.Time <= end_time - 2) &
        (dataframe.Year == 2022) &
        (dataframe.Month == 'May') &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '05-01')
        ].sort_values(by=['Date_Time'])['Hourly_Counts'])
    nearby_past1 = np.array(dataframe[
        (dataframe.Day == day) &
        (dataframe.Sensor_ID == nearby) &
        (dataframe.Time >= start_time - 1) &
        (dataframe.Time <= end_time - 2) &
        (dataframe.Year == 2022) & (dataframe.Month == 'May') &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '05-01')
        ].sort_values(by=['Date_Time'])['Hourly_Counts'])
    sensor3_pastday = np.array(dataframe[
        (dataframe.Day == yesterday) &
        (dataframe.Sensor_ID == sensor_id) &
        (dataframe.Time >= start_time) &
        (dataframe.Time <= end_time - 1) &
        (dataframe.Year == 2022) &
        (dataframe.Month == 'May') &
        (dataframe.Date_Time.dt.strftime('%m-%d') != '05-31')
        ].sort_values(by=['Date_Time'])['Hourly_Counts'])

    return (rain_prev, solar_prev, temp_prev, sensor3_past1, nearby_past1,
            sensor3_pastday)


def compute_distance(distance, merge_df):
    """
    Compute and return the euclidean distance. Take an empty dictionary
    distance and compute the euclidean distance for the two Hourly_Counts
    in merge_df.
    """
    # get the data of distance for each day
    for day in merge_df['Date_Time'].unique():
        count_sensor1 = merge_df[merge_df.Date_Time == day]['Hourly_Counts_x']
        count_sensor2 = merge_df[merge_df.Date_Time == day]['Hourly_Counts_y']
        distance[day] = np.linalg.norm(
            np.array(count_sensor1) - np.array(count_sensor2))

    return distance


def pearson_distance(pearson_coef, compare_merged):
    """
    Compute and return the pearson correlation coefficient.
    Take an empty dictionary distance and compute the correlation for the two
    Hourly_Counts in compare_merged.
    """
    # get the person coefficient of each day of week
    for day in compare_merged['Date_Time'].unique():
        count_sensor1 = compare_merged[
            compare_merged.Date_Time == day]['Hourly_Counts_x']
        count_sensor2 = compare_merged[
            compare_merged.Date_Time == day]['Hourly_Counts_y']
        pearson_coef[day] = np.abs(np.corrcoef(
            np.array(count_sensor1), np.array(count_sensor2))[1, 0])

    return pearson_coef


def diff_conclusion(e_distance, measure):
    """
    Print the concluding result of euclidean distance e_distance.
    """
    # compute the maximum and minimum values
    max_day = max(e_distance, key=e_distance.get)
    max_change = e_distance[max_day]
    min_day = min(e_distance, key=e_distance.get)
    min_change = e_distance[min_day]

    # print the required output using the max and min info of data
    print(
        "Day with the greatest " + measure + " is " + str(max_day) +
        ", and the value is " + str(round(max_change, 2)) + ".")
    print(
        "Day with the least " + measure + " is " + str(min_day) +
        ", and the value is " + str(round(min_change, 2)) + ".")


def find_extreme_item(result):
    """
    Print the keys with maximum value and minimum value in dictionary.
    """
    if len(result.keys()) != 0:
        max_item = ''
        min_item = ''
        max_value = max(result.values())
        min_value = min(result.values())
        for key in result.keys():
            if result[key] == max_value:
                max_item = key
            if result[key] == min_value:
                min_item = key
        return (max_item, min_item)
    else:
        return('', '')
