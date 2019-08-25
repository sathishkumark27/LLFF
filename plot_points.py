import sys
import os
import numpy as np
import math, random

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt    
from mpl_toolkits.mplot3d import axes3d
# mpl.use('Agg')
import mpl_toolkits.mplot3d 

def plot_points_2d(pts_wld, pts_cam, scaling_factor=1., savepath=None):
    
    plt.figure(figsize=(12,4))
    print("pts_wld.shape = ", pts_wld.shape)
    print("pts_cam.shape = ", pts_cam.shape)

    plt.subplot(121)    
    plt.plot(pts_cam[:,1], -pts_cam[:,0])
    plt.plot(pts_wld[:,0], pts_wld[:,1],c='r')    
    plt.scatter(pts_cam[:,1], -pts_cam[:,0])
    plt.scatter(pts_wld[:,0], pts_wld[:,1])

    plt.axis('equal')

    plt.subplot(122)    
    plt.plot(pts_cam[:,1], pts_cam[:,2])
    plt.plot(pts_wld[:,0], pts_wld[:,2],c='r')  
    plt.scatter(pts_cam[:,1], pts_cam[:,2])
    plt.scatter(pts_wld[:,0], pts_wld[:,2])
    plt.axis('equal')
    if savepath is not None:
        plt.savefig(os.path.join(savepath, 'points_2d.png'))
        plt.close()

def plot_points_3d(pts_wld, pts_cam, scaling_factor=1., savepath=None):
    
    x_w = pts_wld[:,0]
    y_w = pts_wld[:,1]
    z_w = pts_wld[:,2]
    x_c = pts_cam[:,1]
    y_c = -pts_cam[:,0]
    z_c = pts_cam[:,2]
    #print([xi])
    #fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    #ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    ax.scatter(x_c, y_c, z_c, s=100, c='b', zorder=10)
    ax.scatter(x_w, y_w, z_w, s=100, c='r', zorder=10)
    if savepath is not None:
        plt.savefig(os.path.join(savepath, 'points_3d.png'))
        plt.close()

pts_wld = np.array([[-2.98419106, -3.87355404, 0.99440323],
[-3.93910212, -0.1695384, 1.52158497],
[-0.82782498, -0.2301885, 1.99226294],
[1.22367178, 0.16056286, 0.57588912],
[3.84738677, 0.51595765, 0.24046618],
[2.3610052, 3.29575044, -1.07639388],
[0.77849219, 3.34961145, -1.49014218],
[0.179419, 2.89678789, -0.72758554],
[-1.26191318, 2.99112423, -1.3642171],
[-2.55860445, 2.96718112, -1.71064796],
[-1.54586359, 3.70809827, -0.85397246],
[-1.32707106, -4.13820683, 1.1872404],
[-0.24270711, 3.49277061, -1.09063448],
[1.30828567, 3.45523068, -1.03548848],
[3.98747717, 3.37851225, -1.3876157],
[0.02569516, -4.13151108, 0.98213431],
[2.2664466, -3.83673998, 0.51085049],
[3.89171251, -3.69511558, 0.34002146],
[3.33440774, -2.7356131, 0.36812083],
[0.81510065, -2.15788379, 0.98891873],
[-1.54434833, -1.70221788, 1.347863],
[-4.06692515, -1.15463683, 1.60125631]])
 
pts_cam = np.array([[-3.90419092, -3.18123416, -1.1226185],
[-0.17157576, -4.1448317, -1.34300777],
[-0.19433322, -1.04706909, -1.89853504],
[0.0896256, 1.04142145, -0.51108594],
[0.42044437, 3.67350073, -0.21883272],
[3.09170296, 2.22583619, 1.34300706],
[3.11304364, 0.65480541, 1.80109323],
[2.71875644, 0.03532458, 1.02249732],
[2.76370289, -1.38866855, 1.70210083],
[2.71274762, -2.67584554, 2.07969164],
[3.51700584, -1.68498536, 1.25515539],
[-4.15237032, -1.5300958, -1.37839781],
[3.28529627, -0.3763453, 1.44052952],
[3.25309705, 1.17261426, 1.34195755],
[3.15181457, 3.86002639, 1.61673798],
[-4.16027221, -0.17240245, -1.20899854],
[-3.9004378, 2.08033435, -0.77583755],
[-3.77101884, 3.70970892, -0.63756768],
[-2.81252062, 3.15310936, -0.57839604],
[-2.19123934, 0.61912405, -1.08731391],
[-1.71135926, -1.74834127, -1.34864703],
[-1.14791896, -4.27598649, -1.49353198]])

plot_points_2d(pts_wld, pts_cam, scaling_factor=1., savepath="./")
plot_points_3d(pts_wld, pts_cam, scaling_factor=1., savepath="./")