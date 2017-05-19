"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
from scipy import stats
import RTLearner as rt
import time

class BagLearner(object):

    def __init__(self, learner = rt.RTLearner, kwargs = {"leaf_size":1, "verbose":False}, bags = 20, boost = False, verbose = False):
        self.learner = learner(**kwargs)
        self.learners = [learner(**kwargs) for _ in range(bags)]
        self.bags = bags

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        for learner in self.learners:
            sample = np.random.randint(dataX.shape[0], size=dataX.shape[0])
            learner.addEvidence(dataX[sample,:], dataY[sample])




    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        values = np.zeros((points.shape[0], self.bags))
        for i in range(len(self.learners)):
            values[:,i] = self.learners[i].query(points)

        return np.mean(values,axis=1)

    def build_tree(self, dataX, dataY):
        return self.learner.build_tree(self.learner, dataX, dataY)

if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
