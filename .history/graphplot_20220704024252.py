"""
Graph plotting module for syndicate assignment.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def bar_with_title(dataframe, title, x_axis, y_axis, fname):
    """
    Plot a histogram (bar plot) using data from dataframe df, with given name
    of x-aixs, y-axis, title. Save the plot to location given by fname.
    """
    dataframe.plot(kind = "bar")
    plt.xticks(horizontalalignment="center")
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')


def time_series_for_two(
    time_series, col1, col2, col1_name, col2_name, title, seg = ''):
    """
    Plot the time series data for two different columens col1 and col2 in the
    dataframe dataframe. Name the axis using col1_name and col2_name. Name the
    picture with title.
    Save the images as png image. If seg not empty, then the time series is
    segmented, else, it is the complete dataframe.
    """
    _, ax_left = plt.subplots(figsize=(17,5))
    ax_right = ax_left.twinx()

    ax_left.plot(
        time_series.Date_Time, time_series[col1], color = 'black',
        label = col1_name)
    ax_right.plot(
        time_series.Date_Time, time_series[col2], color = 'red',
        label = col2_name)

    plt.title(title)
    ax_left.set_ylabel(col1_name, color = 'black')
    ax_right.set_ylabel(col2_name, color = 'red')

    h_1, l_1 = ax_left.get_legend_handles_labels()
    h_2, l_2 = ax_right.get_legend_handles_labels()


    plt.legend(h_1 + h_2, l_1 + l_2, loc=2)
    plt.savefig(col1 + '_' + col2 + '_time_series' + str(seg) + '.png')



def unusual_day_plot(dataframe, info, index, sensor_id, nearby, model):
    """
    Plot the scatter plot of the pedestrian count of a sensor with 'sensor_id'.
    There is also a curve representing a predicted count of that sensor
    to show how unusual the real count is.
    Save the images as png image using the date value storeed in info.
    """

    month = info.iloc[index, 1]
    mdate = info.iloc[index, 2].astype(int)
    lastday = mdate - 1
    lastmonth = month

    if mdate == 1:
        if month in ['January', 'May']:
            lastday = 30
        elif month == 'February':
            lastday = 28
        else:
            lastday = 31

    rain_prev = np.array(dataframe[(dataframe.Sensor_ID == sensor_id) \
        & (dataframe.Year ==2022) & (dataframe.Month == lastmonth) \
        & (dataframe.Mdate == lastday)] \
        .sort_values(by = ['Date_Time'])['Rainfall amount (millimetres)'])
    solar_prev = np.array(dataframe[(dataframe.Sensor_ID == sensor_id) \
        & (dataframe.Year ==2022) & (dataframe.Month == lastmonth) \
        & (dataframe.Mdate == lastday)] \
        .sort_values(by = ['Date_Time'])['Daily global solar exposure (MJ/m*m)'])
    temp_prev = np.array(dataframe[(dataframe.Sensor_ID == sensor_id) \
        & (dataframe.Year ==2022) & (dataframe.Month == lastmonth) \
        & (dataframe.Mdate == lastday)] \
        .sort_values(by = ['Date_Time'])['Maximum temperature (Degree C)'])
    sensor2_pastday = np.array(dataframe[(dataframe.Sensor_ID == nearby) \
        & (dataframe.Year ==2022) & (dataframe.Month == lastmonth) \
        & (dataframe.Mdate == lastday)] \
        .sort_values(by = ['Date_Time'])['Hourly_Counts'])
    sensor3_pastday = np.array(dataframe[(dataframe.Sensor_ID == sensor_id) \
        & (dataframe.Year ==2022) & (dataframe.Month == lastmonth) \
        & (dataframe.Mdate == lastday)] \
        .sort_values(by = ['Date_Time'])['Hourly_Counts'])

    factors = np.concatenate((rain_prev.reshape(-1,1), solar_prev.reshape(-1,1), \
        temp_prev.reshape(-1,1), sensor2_pastday.reshape(-1,1), \
        sensor3_pastday.reshape(-1,1)), axis = 1)

    new_df = dataframe[(dataframe.Sensor_ID == sensor_id) & (dataframe.Year ==2022) \
        & (dataframe.Month == month) & (dataframe.Mdate == mdate)]
    new_df['predicted'] = model.predict(factors)
    new_df.plot.scatter(x = 'Time', y = 'Hourly_Counts', c = 'green')
    spline_1 = make_interp_spline(new_df['Time'], new_df['predicted'])
    value_range = np.linspace(new_df['Time'].min(), new_df['Time'].max(), 500)
    target = spline_1(value_range)
    plt.plot(value_range, target, c= 'lightblue')
    plt.title('Daily Pedestrian Counts of ' + month + ' ' + mdate)
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.savefig('unusual_daily_plot_' + str(index) + '.png')
