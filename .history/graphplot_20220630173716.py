import matplotlib.pyplot as plt


def barWithTitle(df, title, x, y):
    df.plot(kind = "bar")
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.title("Mean daily overall pedestrain count for each day of week in 2022")
    plt.xlabel("Day of week")
    plt.ylabel("mean daily overall pesdestrain count")
    plt.savefig('2021_busy_daily.png')