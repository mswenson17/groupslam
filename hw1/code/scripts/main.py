import numpy as np
# import sys
# import pdb
import math

from MapReader import MapReader
from MotionModel import MotionModel
from SensorModel import SensorModel
from Resampling import Resampling
from matplotlib import pyplot as plt
# from matplotlib import figure as fig
from functools import partial
from multiprocessing import Pool


def visualize_map(occupancy_map):
    # plt.switch_backend('TkAgg')
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    plt.ion()
    # plt.imshow(np.transpose(occupancy_map), cmap='Greys')
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep):
    x_locs = X_bar[:, 1] / 10.0
    y_locs = X_bar[:, 0] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    plt.savefig("../../../../log/" + '{:03d}'.format(tstep) + ".png")
    plt.pause(0.00001)
    scat.remove()  # comment this out for a quick'n'dirty trjactory visualizer


def visualize_lasers(pos, z_t, time_idx, fig):
    ax = fig.add_subplot(111)
    del ax.lines[:]  # refresh

    x_locs = pos[0] / 10.0
    y_locs = pos[1] / 10.0
    theta = pos[2]

    lines = []
    for i in range(0, len(z_t), 10):  # show every 10th measurement
        beamAngle = i * math.pi / 180 + (theta - math.pi / 2)  # radians

        x_laser = math.cos(beamAngle) * (z_t[i] / 10.0) + x_locs
        y_laser = math.sin(beamAngle) * (z_t[i] / 10.0) + y_locs

        lines.append(ax.plot([x_locs, x_laser], [y_locs, y_laser], 'b'))


def init_particles_random(num_particles):

    # initialize [x, y, theta] positions in world_frame for all particles
    # (randomly across the map)
    x0_vals = np.random.uniform(4000, 4001, (num_particles, 1))
    y0_vals = np.random.uniform(4000, 4001, (num_particles, 1))
    theta0_vals = np.random.uniform(3.14, 3.6, (num_particles, 1))
    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    # pdb.set_trace()

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    # (in free space areas of the map)

    freeSpaceThreshold = .1
    X_bar_init = np.empty([0, 4])

    while len(X_bar_init) < num_particles:

        y = np.random.uniform(0, 8000)
        x = np.random.uniform(3000, 7000)
        # y = np.random.uniform(4000, 8000)
        # x = np.random.uniform(4000, 8000)
        theta = np.random.uniform(0, 2 * np.pi)

        #x=4000
        #y=4000
        theta =260/360*math.pi*2

        result = occupancy_map[int(y / 10), int(x / 10)]
        if abs(result) <= freeSpaceThreshold:  # we're good!
            X_bar_init = np.vstack((X_bar_init, [y, x, theta, 1 / float(num_particles)]))

    return X_bar_init


def get_laser_odom(robot_odom):
    return ((robot_odom[0] + 25 * math.cos(robot_odom[2])) / 10.,
            (robot_odom[1] + 25 * math.sin(robot_odom[2])) / 10.,
            robot_odom[2])


def particle_update(meas_type, u_t0, u_t1, ranges, time_idx, particle):
    global motion_model, sensor_model
    x_t1 = np.zeros((1, 3), dtype=np.float64)

    x_t0 = particle[0:3]
    """
    MOTION MODEL
    """
    if ~(u_t0[0:3] == u_t1[0:3]).all():
        # x_t0 = X_bar[m, 0:3]
        x_t1 = motion_model.update(u_t0, u_t1, x_t0)
    else:
        # x_t0 = X_bar[m, 0:3]
        x_t1 = x_t0

    """ SENSOR MODEL """
    if (meas_type == "L"):
        # x_t0 = X_bar[m, 0:3]
        x_t1 = motion_model.update(u_t0, u_t1, x_t0)
        odometry_laser = get_laser_odom(x_t1)

        z_t = ranges
        # print("odom laser: " + str(odometry_laser))
        w_t = sensor_model.beam_range_finder_model(z_t, odometry_laser)
        # w_t = 1/num_particles
        particle_update = np.hstack((x_t1, w_t))

    else:
        particle_update = np.hstack((x_t1, particle[3]))

    return particle_update


def pool_init():
    global map_obj, occupancy_map, motion_model, sensor_model
    """
    Initialize Parameters
    """
    src_path_map = '../data/map/wean.dat'

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()

    motion_model = MotionModel()
    sensor_model = SensorModel(map_obj)


def main():
    """
    Initialize Parameters
    """
    src_path_map = '../data/map/wean.dat'

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()

    src_path_log = '../data/log/robotdata1.log'
    logfile = open(src_path_log, 'r')

    resampler = Resampling()

    num_particles = 1000
    vis_flag = 1

    if vis_flag:
        visualize_map(occupancy_map)

    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """  # pf = ParticleFilter()

    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    # X_bar = init_particles_random(num_particles)
    X_bar = init_particles_freespace(num_particles, occupancy_map)

    pool = Pool(8, pool_init)

    first_time_idx = True
    last_time_stamp = 0
    plot_index = 0
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        meas_type = line[0]  # L : laser scan measurement, O : odometry measurement
        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        u_t1 = meas_vals[0:3]  # odometry reading [x, y, theta] in odometry frame
        time_stamp = meas_vals[-1]

        # if ((time_stamp <= 0.0) | (meas_type == "O")): # ignore pure odometry measurements for now (faster debugging)
        # continue

        print("Processing time step " + str(time_idx) + " at time " + str(time_stamp) + "s measurement: " + meas_type)
        if (meas_type == "L"):
            # odometry_laser = meas_vals[3:6]  # [x, y, theta] coordinates of laser in odometry frame
            ranges = meas_vals[6:-1]  # 180 range measurement values from single laser scan
            # if num_particles < 5:
            # for x in X_bar:
            # visualize_lasers(x, ranges, time_idx, map_obj)

        if (first_time_idx):
            u_t0 = u_t1
            first_time_idx = False
            continue
        X_bar_new = np.empty((0, 4), dtype=np.float64)

        p_up = partial(particle_update, meas_type, u_t0, u_t1, ranges, time_stamp)
        results = pool.map(p_up, X_bar)
        X_bar_new = np.squeeze(results)

        # """
        # RESAMPLING
        # # """
        # if np.dot(u_t0,u_t0)!=np.dot(u_t1,u_t1):
        # if ~(u_t0[0:3] == u_t1[0:3]).all():
        if (meas_type == "L"):
            X_bar_new = resampler.low_variance_sampler(X_bar_new)

        X_bar = X_bar_new
        u_t0 = u_t1

        if vis_flag and time_stamp - last_time_stamp > .1:
            visualize_timestep(X_bar, plot_index)
            last_time_stamp = time_stamp
            plot_index+=1


if __name__ == "__main__":
    main()
