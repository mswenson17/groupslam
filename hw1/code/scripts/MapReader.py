import numpy as np

from matplotlib import pyplot as plt
# from matplotlib import figure as fig


class MapReader:

    def __init__(self, src_path_map):

        self._occupancy_map = np.genfromtxt(src_path_map, skip_header=7)
        self._occupancy_map[self._occupancy_map < 0] = -1
        self._occupancy_map[self._occupancy_map > 0] = 1 - self._occupancy_map[self._occupancy_map > 0]
        self._occupancy_map = np.flipud(self._occupancy_map)

        self._resolution = 10  # each cell has a 10cm resolution in x,y axes
        self._size_x = self._occupancy_map.shape[0] * self._resolution
        self._size_y = self._occupancy_map.shape[1] * self._resolution

        print 'Finished reading 2D map of size : ' + '(' + str(self._size_x) + ',' + str(self._size_y) + ')'

    def visualize_map(self):
        # fig = plt.figure()
        # plt.switch_backend('TkAgg')
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.ion()
        plt.imshow(self._occupancy_map, cmap='Greys')
        plt.axis([0, self._size_x / 10, 0, self._size_y / 10])
        plt.draw()
        plt.pause(10)

    def get_map(self):
        return self._occupancy_map

    def get_map_size_x(self):  # in cm
        return self._size_x

    def get_map_size_y(self):  # in cm
        return self._size_y

    def raytrace(self, x_t1, theta):
        length = np.floor(self.get_map_size_x() / self._resolution / 2)

        points = np.linspace(1, length, num=length, endpoint=False)
        x_vals = np.floor(points * np.cos(theta + x_t1[2]) + x_t1[0])
        y_vals = np.floor(points * np.sin(theta + x_t1[2]) + x_t1[1])

        x_vals[x_vals < 0] = 0
        y_vals[y_vals < 0] = 0
        x_vals[x_vals >= self._size_x / self._resolution] = self._size_x / self._resolution - 1
        y_vals[y_vals >= self._size_y / self._resolution] = self._size_y / self._resolution - 1
        # fig = plt.figure()
        # plt.switch_backend('TkAgg')
        wall = self._occupancy_map[y_vals.astype(int), x_vals.astype(int)]
        edge = np.where(wall > .9)[0]
        point = (x_vals[edge[0]], y_vals[edge[0]])
        # print("raytrace: " + str(point))

        # print("points")
        # print(wall.shape)
        # print(wall)
        # print(edge[0])
        # print("wall point")
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        # plt.ion()
        # plt.imshow(self._occupancy_map, cmap='Greys')
        # plt.axis([0, self._size_x / 10, 0, self._size_y / 10])
        # plt.plot(x_vals, y_vals, 'o')

        # # test = np.where(self._occupancy_map != -1.000)
        # # plt.plot(test[0], test[1], '.g')
        # # plt.plot(x_vals, y_vals, 'o')

        # plt.draw()
        # plt.pause(9)
        return point

if __name__ == "__main__":

    src_path_map = '../data/map/wean.dat'
    map1 = MapReader(src_path_map)
    # map1.raytrace((400, 400, 0), 1.6)
    map1.visualize_map()
