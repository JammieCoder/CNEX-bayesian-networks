from causalnex.network import BayesianNetwork
from sklearn.model_selection import train_test_split


class BNetwork:
    __bn = None
    __sm = None
    __data = []

    # Data: discretised data
    def __init__(self, data, network):
        self.__data = data
        self.__sm = network.get_sm()
        self.setup()

    def setup(self):
        self.__bn = BayesianNetwork(self.__sm)

        # full dataset used to set states incase some are not used
        self.__bn = self.__bn.fit_node_states(self.__data)

        # split data into training and testing sets using
        # specified randomiser and train size of 0.9
        train, test = train_test_split(self.__data, train_size=0.9
                                       , random_state=7)
        self.__bn = self.__bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")
        # cpds dict
        print(self.__bn.cpds.get('G1'))
