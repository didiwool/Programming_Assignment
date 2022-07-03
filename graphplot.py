"""
Graph plotting module for syndicate assignment.
"""
import matplotlib.pyplot as plt


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
