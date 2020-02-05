import matplotlib.pyplot as plt
import seaborn as sns


def plot_line(x,y,title,xlabel,ylabel):
    sns.set_style("darkgrid")
    plt.plot(x, y, linewidth=0.1)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()