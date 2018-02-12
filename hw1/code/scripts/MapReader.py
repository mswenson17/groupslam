import numpy as np
from matplotlib import pyplot as plt
# from matplotlib import figure as fig


class MapReader:

    def __init__(self, src_path_map):

        self._occupancy_map = np.genfromtxt(src_path_map, skip_header=7)
        self._occupancy_map[self._occupancy_map < 0] = -1
        self._occupancy_map[self._occupancy_map > 0] = 1 - self._occupancy_map[self._occupancy_map > 0]
        self._occupancy_map = np.flipud(self._occupancy_map)
        # self._occupancy_map = np.transpose(self._occupancy_map)

        self._resolution = 10  # each cell has a 10cm resolution in x,y axes
        self._size_x = self._occupancy_map.shape[0] * self._resolution
        self._size_y = self._occupancy_map.shape[1] * self._resolution

        length = 700.
        scale = 2
        points = np.tile(np.arange(0., length, scale), [181, 1])
        self._sines = np.empty(np.shape(points))
        self._cosines = np.empty(np.shape(points))
        for idx, point in np.ndenumerate(points):
            self._sines[idx] = np.sin(idx[0] * np.pi / 180.) * point
            self._cosines[idx] = np.cos(idx[0] * np.pi / 180.) * point
            # print(str(idx) + ": " + str(point) + " val: "+ str(np.sin(idx[0] * np.pi / 180.) * point) + " sin: "+str(self._sines[idx]))

        # print(self._sines[1,...])
        # print(self._cosines[1,...])
        # print(np.shape(self._sines))

        print('Finished reading 2D map of size : ' + '(' + str(self._size_x) + ',' + str(self._size_y) + ')')

    def visualize_map(self):
        # fig = plt.figure()
        # plt.switch_backend('TkAgg')
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.ion()
        plt.imshow(np.transpose(self._occupancy_map), cmap='Greys')
        plt.axis([0, self._size_x / 10, 0, self._size_y / 10])
        plt.draw()
        plt.pause(10)

    def get_map(self):
        return self._occupancy_map

    def get_map_size_x(self):  # in cm
        return self._size_x

    def get_map_size_y(self):  # in cm
        return self._size_y

    def raytrace(self, x_t1, angle, debug=False):

        # points = np.linspace(1, length, num=length, endpoint=False)
        # x_vals = np.floor(points * np.cos(theta + x_t1[2]) + x_t1[0])
        # y_vals = np.floor(points * np.sin(theta + x_t1[2]) + x_t1[1])
        theta = x_t1[2]  # angle * np.pi / 180
        sines = self._sines[angle, ...] * np.cos(theta) + self._cosines[angle, ...] * np.sin(theta)
        cosines = self._cosines[angle, ...] * np.cos(theta) + self._sines[angle, ...] * np.sin(theta)

        x_vals = np.round(cosines + x_t1[0])
        y_vals = np.round(sines + x_t1[1])

        x_vals[x_vals < 0] = 0
        y_vals[y_vals < 0] = 0
        x_vals[x_vals >= self._size_x / self._resolution] = self._size_x / self._resolution - 1
        y_vals[y_vals >= self._size_y / self._resolution] = self._size_y / self._resolution - 1
        # fig = plt.figure()
        # plt.switch_backend('TkAgg')
        wall = self._occupancy_map[x_vals.astype(int), y_vals.astype(int)]
        edge = np.where(wall > .9)[0]
        if len(edge) > 0:
            point = (x_vals[edge[0]], y_vals[edge[0]])
        else:
            point = (0., 0.)

        # print(x_t1)
        # print(cosines)
        # print(x_vals)
        if debug:
            print("raytrace: " + str(point))

            # print("points")
            # print(wall.shape)
            # print(wall)
            # print(edge[0])
            # print("wall point")
            # mng = plt.get_current_fig_manager()
            # mng.resize(*mng.window.maxsize())
            plt.ion()
            plt.imshow(np.transpose(self._occupancy_map), cmap='Greys')
            plt.axis([0, self._size_x / 10, 0, self._size_y / 10])
            plt.plot(x_vals, y_vals)

            # test = np.where(self._occupancy_map != -1.000)
            # plt.plot(test[0], test[1], '.g')
            plt.plot(x_vals, y_vals, 'o')
            plt.draw()
            plt.pause(10)
        return point


if __name__ == "__main__":

    src_path_map = '../data/map/wean.dat'
    map1 = MapReader(src_path_map)
    map1.raytrace((90, 570, 0), 40)
    # map1.visualize_map()
