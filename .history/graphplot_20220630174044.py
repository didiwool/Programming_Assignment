import matplotlib.pyplot as plt


def barWithTitle(df, title, x, y, fname):
    df.plot(kind = "bar")
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(fname)
