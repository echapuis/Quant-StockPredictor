import indicators
import numpy as np
import pandas as pd
import datetime as dt
import math
import time
import util
import sys
import helpers
import matplotlib.pyplot as plt
import rule_based
import tester
from cycler import cycler
import ML_based

# pd.set_option('display.height',10)
if __name__ == "__main__":

    data = 'train' # test or train
    graph = True #makes a graph for manual and ml

    manual_vals, bench_vals = rule_based.test_Manual(data=data, graph=graph)
    ML_vals, bench_vals = ML_based.test_ML(data=data,graph=graph)

    plt.figure(0)
    df_temp = pd.concat([manual_vals, ML_vals, bench_vals], keys=['Manual', 'ML', 'Benchmark'], axis=1)
    df_temp.ix[0, 0] = 1
    df_temp = df_temp.fillna(method='ffill')
    # print df_temp
    plt.rc('axes', prop_cycle=(cycler('color', ['blue', 'green', 'black'])))
    plt.plot(df_temp)
    plt.legend(['Manual', 'ML', 'Benchmark'], loc='upper left')
    plt.xticks(rotation=45)
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.grid()
    plt.title('Comparison of Quantitative Methods')
    plt.show()





