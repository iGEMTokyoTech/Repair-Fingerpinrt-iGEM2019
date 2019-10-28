import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_hist(width_list):
    # sns.set()
    fig, ax = plt.subplots()
    sns.distplot(
        width_list, bins=np.arange(min(width_list)-1, max(width_list)) + 0.5, color='#123456', label='data',
        kde=False,
        rug=False,
    )
    ax.set(ylabel='the number of circles', xlabel='radius of circle')
    # plt.savefig(file)
    plt.show()
