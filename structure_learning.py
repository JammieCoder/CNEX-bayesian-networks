import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from simple_plot import NetworkPlot


class StructureLearner:
    __data = []
    __struct_data = []
    network = None

    def __init__(self):
        self.__data = pd.read_csv('student/student-por.csv', delimiter=';')
        self.drop_cols()
        self.encode_labels()
        self.network = NetworkPlot(self.__struct_data)


    def drop_cols(self):
        drop_col = ['sex', 'school', 'age', 'Mjob', 'Fjob', 'reason', 'guardian']
        self.__data = self.__data.drop(columns=drop_col)

    def get_numeric(self):
        return (f'Numeric:\n {list(self.__struct_data.columns)} '
                f'\n {self.__struct_data.head()}\n\n')

    # Label Encoding - Converting categorical data --> numerical data
    def encode_labels(self):
        self.__struct_data = self.__data.copy()
        # Exclude non-numeric columns (np.number = base class/datatype for all numerical data)
        # Selects columns from the data with data types excluding numeric
        non_numeric = self.__struct_data.select_dtypes(exclude=[np.number])

        le = LabelEncoder()
        for col in non_numeric.columns:
            # Transform the data in each non-numeric column
            self.__struct_data[col] = le.fit_transform(self.__struct_data[col])

    def __str__(self):
        return (f'{list(self.__data.columns)} '
                f'\n {self.__data.head()}\n\n')


if __name__ == '__main__':
    sl = StructureLearner()
    print(sl)
    print(sl.get_numeric())
    #sl.network.plot()
    sl.network.export()
