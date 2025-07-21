#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from IPython.display import display

from IPython.core.display import HTML

def pretty_print_df(df):
    """
    :param df: data frame

    return: None

    displays a dataframe
    """
    display(HTML(df.to_html()))

def print_tree_with_names(tree, column_names, indent):
    """Convert scikit learn GBT model tree into python code"""
    left = tree.children_left
    right = tree.children_right
    threshold = tree.threshold
    features = [column_names[i] for i in tree.feature]
    value = tree.value
    depth = 0

    def get_indent(depth, indent):
        s = ""
        for j in range(depth):
            s += indent
        return s

    def recurse(left, right, threshold, features, node, depth):
        depth += 1
        text = []
        line_indent = get_indent(depth, indent)
        if (threshold[node] != -2):
            text += [line_indent + 'if row["%s"] <= %f:' % (features[node], threshold[node])]
            if left[node] != -1:
                text += recurse(left, right, threshold, features, left[node], depth)
            text += [line_indent + "else:"]
            if right[node] != -1:
                text += recurse(left, right, threshold, features, right[node], depth)
        else:
            text += [line_indent + 'score += %f' % value[node][0][0]]
        return text

    return recurse(left, right, threshold, features, 0, 1)


def plot_distribution(df, cols, outlier=0.01, x=1, y=1, bins=100, logscale=False):
    tmp = df[cols].copy()
    tmp.dropna(inplace=True)

        # outlier=='stddev': remove all rows with outliers beyond 3 times stddev
    if outlier == 'stdddev':
        tmp = tmp[tmp.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]

    if (x * y < len(cols)):
        x = int(np.floor(np.sqrt(len(cols))))
        y = int(np.ceil(len(cols) / x))

    sns.set(font_scale=0.8)
    sns.set_style("dark")

    fig, axes = plt.subplots(y, x, figsize=(10, 3))
    if (x * y > 1):
        ax = axes.ravel()
    else:
        ax = []
        ax.append(axes)
    i = 0
    for column in tmp.columns:
        if outlier != 'stddev' and outlier == outlier and outlier > 0 and outlier < 1:
            # filter by fraction
            outlier = round(outlier, 2)
            xmin = tmp[column].describe(percentiles=[outlier])[str(int(100 * outlier)) + '%']
            xmax = tmp[column].describe(percentiles=[1 - outlier])[str(int(100 * (1 - outlier))) + '%']
            ttmp = tmp[(tmp[column] > xmin) & (tmp[column] < xmax)]
        else:
            ttmp = tmp

        _, _bins = np.histogram(ttmp[column], bins=bins)
        ax[i].hist(ttmp[column], bins=_bins)
        # ax[i].set_yticks(())
        ax[i].set_title(column)
        if logscale:
            ax[i].set_yscale('log')
        i += 1

    fig.tight_layout()
    plt.show()


def normalize(df, cols, verbose=False):
    for var in cols:
        low = df[var].describe(percentiles=[0.01])['1%']
        high = df[var].describe(percentiles=[0.99])['99%']
        diff = high - low
        low = max(low - diff * 0.05, df[var].describe()['min'])
        high = min(high + diff * 0.05, df[var].describe()['max'])
        if (verbose):
            print(var, low, high)
            print(df[var].describe(percentiles=[0.01, 0.99]))
        df[var + '_n'] = df[var].apply(lambda x: x if x != x else min(1, max(0, (x - low) / (high - low))))
    return [x + '_n' for x in cols]

def corrplot(df, outlier=True, verbose=False, fixedscale=True, annot=True):
    tmp = df.copy()
    if outlier == True:
        for var in df.columns:
            low = df[var].describe(percentiles=[0.01])['1%']
            high = df[var].describe(percentiles=[0.99])['99%']
            diff = high - low
            low = max(low - diff * 0.05, df[var].describe()['min'])
            high = min(high + diff * 0.05, df[var].describe()['max'])
            if (verbose):
                print(var, low, high)
                print(df[var].describe(percentiles=[0.01, 0.99]))
            tmp[var] = df[var].apply(lambda x: x if x != x else min(1, max(0, (x - low) / (high - low))))

    corr = df.corr()
    _, dim = df.shape
    print('Minimum correlation:', min(corr.min()), 'Maximum correlation:', max(corr.max()))
    plt.figure(figsize=(dim, dim / 2))
    cmap = sns.diverging_palette(220, 5, as_cmap=True)
    if (fixedscale):
        sns.heatmap(corr,
                    cmap=cmap,
                    annot=annot,
                    vmax=1, vmin=-1,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values)
    else:
        sns.heatmap(corr,
                    cmap=cmap,
                    annot=annot,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values)
    plt.show()


def scatterplot(df, col1, col2, outlier=True, hue=None, threshold=None, regressionline=False):
    plt.figure(figsize=(10, 10))
    if regressionline:
        sp = sns.regplot(x=col1, y=col2, data=df, fit_reg=True, line_kws={'color': 'r'})
    else:
        sp = sns.scatterplot(x=col1, y=col2, hue=hue, data=df)
    if outlier == True:
        sp.set(xlim=(df[col1].describe(percentiles=[0.01])['1%'], df[col1].describe(percentiles=[0.99])['99%']))
        sp.set(ylim=(df[col2].describe(percentiles=[0.01])['1%'], df[col2].describe(percentiles=[0.99])['99%']))
        print(col1, df[col1].describe(percentiles=[0.01])['1%'], df[col1].describe(percentiles=[0.99])['99%'])
        print(col2, df[col2].describe(percentiles=[0.01])['1%'], df[col2].describe(percentiles=[0.99])['99%'])
    if threshold != None:
        plt.vlines(threshold, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1])
    plt.show()
