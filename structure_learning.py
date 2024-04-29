import numpy as np  # numbers
import pandas as pd  # data sets
from causalnex.discretiser import Discretiser
from sklearn.preprocessing import LabelEncoder  # encoding continuous labels to discrete

from bayesian_network import BNetwork
from network import Network


class StructureLearner:
    __data = []
    __num_data = []
    network = None
    __discretised_data = []

    def __init__(self):
        # pd.DataFrame returned
        self.__data = pd.read_csv('student/student-por.csv', delimiter=';')
        self.drop_cols()
        self.encode_labels()
        self.network = Network(self.__num_data)
        self.discretise()
        self.bn = BNetwork(self.__discretised_data, self.network)

    def get_discretised(self):
        return f'Discretised:\n {list(self.__discretised_data.columns)} ' \
               f'\n {self.__discretised_data.head()}\n\n'

    def drop_cols(self):
        drop_col = ['sex', 'school', 'age', 'Mjob', 'Fjob', 'reason', 'guardian']
        self.__data = self.__data.drop(columns=drop_col)

    def get_numeric(self):
        return (f'Numeric:\n {list(self.__num_data.columns)} '
                f'\n {self.__num_data.head()}\n\n')

    # Label Encoding - Converting categorical data --> numerical data
    def encode_labels(self):
        self.__num_data = self.__data.copy()
        # Exclude non-numeric columns (np.number = base class/datatype for all numerical data)
        # Selects columns from the data with data types excluding numeric
        non_numeric = self.__num_data.select_dtypes(exclude=[np.number])

        le = LabelEncoder()
        for col in non_numeric.columns:
            # Transform the data in each non-numeric column
            self.__num_data[col] = le.fit_transform(self.__num_data[col])

    # Discretising - Converting numerical data --> categorical data
    def discretise(self):

        # == 0 --> 'no-failure',else 'have-failure'
        def failures_map(val):
            return 'no-failure' if val == 0 else 'have-failure'

        # > 2 --> 'long-studytime',else 'short-studytime'

        def studytime_map(val):
            return 'long-studytime' if val > 2 else 'short-studytime'

        def discretise_numeric_data():
            # discretise numeric data
            # buckets < 1, 1-9, >= 10
            self.__discretised_data['absences'] = (Discretiser(method='fixed', numeric_split_points=[0, 10])
                                                   .transform(self.__discretised_data['absences'].values))
            # buckets < 10, >= 10
            self.__discretised_data['G1'] = (Discretiser(method='fixed', numeric_split_points=[10])
                                             .transform(self.__discretised_data['G1'].values))
            self.__discretised_data['G2'] = (Discretiser(method='fixed', numeric_split_points=[10])
                                             .transform(self.__discretised_data['G2'].values))
            self.__discretised_data['G3'] = (Discretiser(method='fixed', numeric_split_points=[10])
                                             .transform(self.__discretised_data['G3'].values))
            # label categories for numeric data
            absences_map = {0: "No-absence", 1: "Low-absence", 2: "High-absence"}
            g1_map = {0: "Fail", 1: "Pass"}
            g2_map = {0: "Fail", 1: "Pass"}
            g3_map = {0: "Fail", 1: "Pass"}
            self.__discretised_data["absences"] = self.__discretised_data["absences"].map(absences_map)
            self.__discretised_data["G1"] = self.__discretised_data["G1"].map(g1_map)
            self.__discretised_data["G2"] = self.__discretised_data["G2"].map(g2_map)
            self.__discretised_data["G3"] = self.__discretised_data["G3"].map(g3_map)

        def discretise_category_data(data_vals):
            # map ndarrays
            vectorised_smap = np.vectorize(studytime_map)
            vectorised_fmap = np.vectorize(failures_map)
            print(vectorised_fmap(data_vals['failures']), vectorised_smap(data_vals['studytime'])
                  , '\n')

            # map data
            self.__discretised_data['failures'] = self.__num_data['failures'].map(failures_map)
            self.__discretised_data['studytime'] = self.__num_data['studytime'].map(studytime_map)

        def get_data_vals():
            # dict of columns to their set of possible values
            data_vals = {}
            for col in self.__data.columns:
                # set of possible values that the column can take
                # Indexed column --> pd.Series datatype
                # unique() --> np.ndarray datatype
                col_vals = self.__data[col].unique()
                # Sort array in-place with np.sort()
                col_vals.sort()
                data_vals[col] = col_vals
            return data_vals

        self.__discretised_data = self.__data.copy()
        data_vals = get_data_vals()

        # selecting just 'failures' and 'studytime' columns
        # returns pd.DataFrame
        data_in_question = self.__num_data[['failures', 'studytime']]
        print('\n', data_in_question, '\n')

        # dataset info - values meanings:
        # studytime: weekly study time (numeric: 1 - <2 hours,
        # 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
        # failures: number of past class failures (numeric: n if 1<=n<3,
        # else 4) - max 4 failures
        print('failures', 'studytime')
        print(data_vals['failures'], data_vals['studytime'])

        discretise_category_data(data_vals)
        discretise_numeric_data()
        print(self.__discretised_data[['failures', 'studytime', 'absences', 'G1', 'G2', 'G3']])

    def __str__(self):
        return (f'{list(self.__data.columns)} '
                f'\n {self.__data.head()}\n\n')


if __name__ == '__main__':
    sl = StructureLearner()
    # print(sl)
    # print(sl.get_numeric())
    # sl.network.plot()
    # sl.network.export()
