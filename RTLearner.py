"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
from scipy import stats


class RTLearner(object):
    def __init__(self, leaf_size = 1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        if self.verbose: print "addEvidence"
        # build and save the model
        self.dataX = dataX
        self.dataY = dataY
        self.tree = self.build_tree(dataX,dataY)

        return self.tree

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        if self.verbose: print "query"
        values = np.zeros(points.shape[0])
        errorCount = 0
        # print self.tree.shape[1]
        # for i in range(4):
        #     if self.tree.shape[0] == 4:
        #         self.build_tree(self.dataX,self.dataY)
        for x in range(points.shape[0]):
            i = 0
            # # if self.tree.shape[0] == 4:
            # #     return self.tree[1]
            # else:
            while self.tree[i,0] != -1:
                factor = self.tree[i,0]
                splitVal = self.tree[i,1]
                if points[x,factor] <= splitVal:
                    i += self.tree[i,2]
                else:
                    i += self.tree[i,3]
                if i >= self.tree.shape[0]-1:
                    errorCount+=1
                    i = self.tree.shape[0]-1
                    break
            values[x] = self.tree[i,1]

        #print 'errorCount: ', errorCount, points.shape[0]
        return values

    def build_tree(self, dataX, dataY):
        if self.verbose: print "build_tree", self.leaf_size
        if self.verbose: print "data shape", dataX.shape

        #if no elements in subtree, return empty subtree
        if dataX.shape[0] == 0: return np.array([])

        #if there is only 1 item left or if fewer than leaf size, return mode of data
        if dataX.ndim == 1 or dataX.shape[0] <= self.leaf_size: return np.array([-1, stats.mode(dataY).mode[0], -1, -1])

        #if all of the data has the same value, return that value
        # if not np.all(dataY - dataY[0]):
        #     print 'all same'
        #     return np.array([-1, dataY[0],-1,-1])

        else:
            if self.verbose: print "passed conditions"
            i = np.random.randint(dataX.shape[1]-1)
            d = np.random.randint(dataX.shape[0],size=2)
            for j in range(11):
                if dataX[d[0],i] != dataX[d[1],i]: break
                else: d[1] = np.random.randint(dataX.shape[0])
                if j == 10: return np.array([-1, dataY[d[0]], -1, -1])

            splitVal = (dataX[d[0],i] + dataX[d[1],i])/2.0

            indices = dataX[:,i] <= splitVal
            leftTree = self.build_tree(dataX[indices,:], dataY[indices])
            indices = dataX[:, i] > splitVal
            rightTree = self.build_tree(dataX[indices,:], dataY[indices])

            leftTreeSize = leftTree.shape[0] if leftTree.ndim != 1 else 1
            if leftTree.shape[0] == 0 or rightTree.shape[0] == 0: leftTreeSize = 0

            root = [i, splitVal, 1, leftTreeSize + 1]
            if (leftTree.shape[0] != 0): root = np.vstack((root, leftTree))
            if (rightTree.shape[0] != 0): root = np.vstack((root, rightTree))
            return np.array(root)

if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
