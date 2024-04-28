# -*- coding: utf-8 -*-
import webbrowser
from datetime import time

from causalnex.plots import NODE_STYLE, EDGE_STYLE, plot_structure
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

class NetworkPlot:
    __data = None
    __fun = False
    __sm = StructureModel()
    __sm.add_edges_from(
        [('health', 'absences'), ('health', 'G1'), ('absences', 'G1'), ('absences', 'G3'), ('G1', 'G2')])

    def __init__(self, data=None, fun=False):
        self.__data = data
        self.fun = fun

        self.setup()

    # NOTEARS Learns the structure
    def setup(self):
        if self.__data is not None:
            print('Learning Structure...')
            self.__sm = from_pandas(self.__data, w_threshold=0.8)
        print(f'Edges\n{self.__sm.edges}')

    def plot(self):
        print('Plotting...')
        viz = plot_structure(self.__sm, all_node_attributes=NODE_STYLE.WEAK, all_edge_attributes=EDGE_STYLE.WEAK)
        viz.toggle_physics(self.__fun)

        html = viz.generate_html(notebook='01_simple_plot.html')
        file = 'fully_connected.html'
        with open(file, 'w', encoding='utf-8') as f:
            f.write(html)

        print('Opening...')
        print(webbrowser.open(file, 1))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    network = NetworkPlot()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
