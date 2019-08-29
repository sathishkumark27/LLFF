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

def make_cylinder(radius, length, nlength, alpha, nalpha, center, orientation):

    #Create the length array
    I = np.linspace(0, length, nlength)

    #Create alpha array avoid duplication of endpoints
    #Conditional should be changed to meet your requirements
    if int(alpha) == 360:
        A = np.linspace(0, alpha, num=nalpha, endpoint=False)/180*np.pi
    else:
        A = np.linspace(0, alpha, num=nalpha)/180*np.pi

    #Calculate X and Y
    X = radius * np.cos(A)
    Y = radius * np.sin(A)

    #Tile/repeat indices so all unique pairs are present
    pz = np.tile(I, nalpha)
    px = np.repeat(X, nlength)
    py = np.repeat(Y, nlength)

    points = np.vstack(( pz, px, py )).T

    #Shift to center
    shift = np.array(center) - np.mean(points, axis=0)
    points += shift

    #Orient tube to new vector

    #Grabbed from an old unutbu answer
    def rotation_matrix(axis,theta):
        a = np.cos(theta/2)
        b,c,d = -axis*np.sin(theta/2)
        return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                         [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                         [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

    ovec = orientation / np.linalg.norm(orientation)
    cylvec = np.array([1,0,0])

    if np.allclose(cylvec, ovec):
        return points

    #Get orthogonal axis and rotation
    oaxis = np.cross(ovec, cylvec)
    rot = np.arccos(np.dot(ovec, cylvec))

    R = rotation_matrix(oaxis, rot)
    return points.dot(R)


def plot_points_3d(pts_wld, scaling_factor=1., savepath=None, filename=None):
    
    #pts_wld.shape should be n X 3
    #pts_cam.shape should be n X 3

    x_w = pts_wld[:,0]
    y_w = pts_wld[:,1]
    z_w = pts_wld[:,2]
    # x_c = pts_cam[:,1]
    # y_c = -pts_cam[:,0]
    # z_c = pts_cam[:,2]
    #print([xi])
    #fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    #ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    #ax.scatter(x_c, y_c, z_c, s=100, c='b', zorder=10)
    ax.scatter(x_w, y_w, z_w, s=100, c='r', zorder=10)
    if savepath is not None:
        plt.savefig(os.path.join(savepath, filename+'.png'))
        plt.close()
        
points = make_cylinder(3, 5, 5, 360, 10, [0,2,0], [1,0,0])
print(points)
plot_points_3d(points, scaling_factor=1., savepath="./", filename="cyl_points")

