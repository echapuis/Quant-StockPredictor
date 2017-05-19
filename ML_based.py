"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import os
import math
import RTLearner as rt
import BagLearner as bl
import sys
from matplotlib import pyplot as plt
import time
import datetime as dt
import pandas as pd
import helpers
import indicators
import util
from cycler import cycler
# pd.set_option('display.height',2000)

def build_orders (stockData,testData, ysell = -0.05, ybuy= 0.05, leaf_size = 50, bags = 10, majority = 0.3):
    # if len(sys.argv) != 2:
    #     print "Usage: python -m mc3_p1.testlearner <filename>"
    #     sys.exit(1)
    stockData = stockData.fillna(method = 'ffill')
    stockData = stockData.fillna(method = 'bfill')

    price = stockData['Price']['IBM']
    sma = stockData['SMA']['IBM']
    smaRatio = stockData['SMA Ratio']['IBM']
    bbp = stockData['BBP']['IBM']
    ema = stockData['EMA']['IBM']
    emaRatio = stockData['EMA Ratio']['IBM']
    momentum = stockData['Momentum']['IBM']

    YBUY = ybuy
    YSELL = ysell

    returns = price.shift(-10)/price - 1
    returns = returns.fillna(method='ffill')
    returns[returns > YBUY] = 1
    returns[returns < YSELL] = -1
    returns[(returns <=YBUY) & (returns >= YSELL)] = 0



    inf = open("data.csv")
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])
    # plt.figure(1)
    # df_temp = returns
    # df_temp = df_temp.fillna(method='ffill')
    # plt.plot(df_temp)
    # plt.legend(['Returns'], loc='upper left')
    # plt.xticks(rotation=45)
    # plt.ylabel('Returns')
    # plt.xlabel('Date')
    # plt.grid()
    # plt.title('Returns')
    # plt.show()




    testData = testData.fillna(method='ffill')
    testData = testData.fillna(method='bfill')

    tprice = testData['Price']['IBM']
    tsma = testData['SMA']['IBM']
    tsmaRatio = testData['SMA Ratio']['IBM']
    tbbp = testData['BBP']['IBM']
    tema = testData['EMA']['IBM']
    temaRatio = testData['EMA Ratio']['IBM']
    tmomentum = testData['Momentum']['IBM']

    data_file = open('tdata.csv', 'w+')
    for i in range(len(testData)):
        data_file.write(','.join(
            [str(tprice[i]), str(tsmaRatio[i]), str(tbbp[i]), str(temaRatio[i]), str(tmomentum[i])]))
        data_file.write('\n')
    data_file.close()

    inf = open("tdata.csv")
    tData = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])
    # separate out training and testing data
    trainX = data[:,0:-1]
    trainY = data[:,-1]
    testX = tData

    # create a learner and train it
    # var = 30
    # start = 40

    learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":leaf_size}, bags = bags, boost = False, verbose = False) # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it

    # evaluate in sample
    predY = learner.query(testX) # get the predictions
    predY[(predY<=majority) & (predY >= -majority)] = np.NaN
    # predY[predY<=0.5] = -1
    # print predY
    predY[0] = 0


    orders = pd.Series(predY, index = testData.index)
    # print orders.shape[0], orders.count()
    orders.ffill(inplace=True)
    orders.fillna(0, inplace=True)

    orderChange = orders.diff()
    orderChange.ix[0] = 0

    # And more importantly, drop all rows with no non-zero values (i.e. no orders).
    orderChange = orderChange.loc[(orderChange != 0)]
    order_list = []

    # order_list.append([start_date.date(), 'IBM', 'BUY',0])
    size = orders.shape[0]
    lastTrade = -10
    curPos = 0
    for day in orderChange.index:
        loc = orders.index.get_loc(day)
        if loc < lastTrade + 10: continue
        if loc >= size - 10:
            break
        if orderChange.ix[day] > 0:
            if curPos == 1: continue
            order_list.append([day.date(), 'IBM', 'BUY', 500])
            lastTrade = loc
            curPos += 1
            if orders.iloc[loc + 10] > curPos:
                order_list.append([day.date()+dt.timedelta(days=14), 'IBM', 'BUY', 500])
                curPos += 1
        elif orderChange.ix[day] < 0:
            if curPos == -1: continue
            order_list.append([day.date(), 'IBM', 'SELL', 500])
            lastTrade = loc
            curPos -= 1
            if orders.iloc[loc + 10] < curPos:
                order_list.append([day.date()+dt.timedelta(days=14), 'IBM', 'SELL', 500])
                curPos -= 1


    orders = pd.DataFrame(order_list, columns=['Date', 'Symbol', 'Order', 'Shares'])
    os.remove("data.csv")
    os.remove("tdata.csv")
    return orders

    # rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    #
    # c = np.corrcoef(predY, y=trainY)
    # results[i-1-start,0] = rmse
    # results[i-1-start,1] = c[0,1]
    #
    # # evaluate out of sample
    # predY = learner.query(testX) # get the predictions
    # rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    # results[i-1-start,2] =  rmse
    # c = np.corrcoef(predY, y=testY)
    # results[i-1-start,3] = c[0,1]

    # plt.subplot(311)
    # [a,b] = plt.plot(range(1,var+1), results[:,0], 'g-', range(1,var+1), results[:,2], 'b-')
    # plt.title('Change in RMSE with Increasing Leaf Size')
    # plt.xlabel('Leaf Size')
    # plt.ylabel('RMSE')
    # plt.legend([a,b], ['In Sample', 'Out of Sample'], loc = 4)
    #
    #
    # plt.subplot(313)
    # [c,d] = plt.plot(range(1, var+1), results[:, 1], 'g-', range(1, var+1), results[:, 3], 'b-')
    # plt.title('Change in Correlation with Increasing Leaf Size')
    # plt.xlabel('Leaf Size')
    # plt.ylabel('Correlation Coeff.')
    # plt.legend([c, d], ['In Sample', 'Out of Sample'], loc=1)
    #
    # plt.show()

def test_ML(data = 'test', graph=True):
    test = data=='test'
    start_date = dt.datetime(2006, 01, 01)
    end_date = dt.datetime(2009, 12, 31)
    if test:
        tstart_date = dt.datetime(2010, 01, 01)
        tend_date = dt.datetime(2010, 12, 31)
        file = 'test.csv'
    else:
        tstart_date = dt.datetime(2006, 01, 01)
        tend_date = dt.datetime(2009, 12, 31)
        file = 'train.csv'

    display = 'EMA Ratio'
    graph_result = graph
    graph_extra = False
    symbols = ['IBM']
    # symbols = ['HD']

    lookback = 14
    startval = 100000

    data = indicators.get_indicators(symbols, start_date, end_date, lookback)
    tdata = indicators.get_indicators(symbols, tstart_date, tend_date, lookback)

    ysell = -0.03
    ybuy = 0.02
    leaf_size = 5
    bags = 30
    majority = 0.4 #0.4

    orders = build_orders(data, tdata, ysell, ybuy, leaf_size, bags, majority);

    portvals = helpers.compute_portvals(orders, tstart_date, tend_date, startval)
    benchvals = helpers.compute_portvals2(file, startval)

    # print portvals

    prices = util.get_data(['IBM', 'SPY'], pd.date_range(start_date, end_date))

    norm_portvals = helpers.get_norm_data(portvals)
    norm_benchvals = helpers.get_norm_data(benchvals)
    # print prices
    # print norm_portvals
    # print norm_SPY

    if graph_result:
        plt.figure(0)
        df_temp = pd.concat([norm_portvals, norm_benchvals], keys=['Portfolio', 'Benchmark'], axis=1)
        df_temp.ix[0, 0] = 1
        df_temp['Portfolio'] = df_temp['Portfolio'].fillna(method='ffill')
        # print df_temp
        plt.rc('axes', prop_cycle=(cycler('color', ['blue', 'black'])))
        plt.plot(df_temp)
        plt.legend(['Portfolio', 'Benchmark'], loc='upper left')
        plt.xticks(rotation=45)
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.grid()
        plt.title('ML-based Portfolio vs. IBM')

        curPos = 'out'
        for order in orders.index:
            day = orders.ix[order, 'Date']
            ord = orders.ix[order, 'Order']
            if curPos == 'out' and ord == 'BUY':
                color = 'green'
                curPos = 'long'
            elif curPos == 'out' and ord == 'SELL':
                color = 'red'
                curPos = 'short'
            else:
                color = 'black'
                curPos = 'out'
            plt.axvline(day, color=color, linewidth=2)

        plt.show()
    return norm_portvals, norm_benchvals

if __name__ == "__main__":
    test_ML(data = 'train', graph=True)