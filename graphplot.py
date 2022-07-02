import matplotlib.pyplot as plt

# plot a histogram with given name of x-aixs, y-axis, title, and filename for storage
def barWithTitle(df, title, x, y, fname):
    df.plot(kind = "bar")
    plt.xticks(horizontalalignment="center")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')


def timeSeriesForTwo(time_series, col1, col2, col1_name, col2_name, title):
    """
    Plot the time series data for two different columens col1 and col2 in the dataframe df. Name the axis using col1_name and col2_name. Name the picture with title.
    Save the images as png image.
    """
    fig, ax_left = plt.subplots(figsize=(17,5))
    ax_right = ax_left.twinx()

    ax_left.plot(time_series.Date_Time, time_series[col1], color='black', label=col1_name)
    ax_right.plot(time_series.Date_Time, time_series[col2], color='red', label=col2_name)
    plt.title(title)
    ax_left.set_ylabel(col1_name, color = 'black')
    ax_right.set_ylabel(col2_name, color = 'red')

    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()

    plt.legend(h1+h2, l1+l2, loc=2)
    plt.savefig(col1 + '_' + col2 + '_ time_series' + '.png')
