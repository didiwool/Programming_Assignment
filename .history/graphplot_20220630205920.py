import matplotlib.pyplot as plt

# plot a histogram with given name of x-aixs, y-axis, title, and filename for storage
def barWithTitle(df, title, x, y, fname):
    df.plot(kind = "bar")
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(fname)
