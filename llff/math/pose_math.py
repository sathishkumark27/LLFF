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



def normalize(x):
    return x / np.linalg.norm(x)

# https://www.3dgep.com/understanding-the-view-matrix/
def viewmatrix(z, up, pos):
    vec2 = normalize(z) #z-axis
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg)) # x-axis
    vec0 = normalize(np.cross(vec1, vec2)) # y-axis
    m = np.stack([vec0, vec1, vec2, pos], 1) # adding the rotaion info i.e where to look from the center point 'pos'
    return m

def poses_avg(poses):

    hwf = poses[:3, -1:, 0]
    #print("poses.shape = ", poses.shape)
    #print("poses = ", poses)

    center = poses[:3, 3, :].mean(-1)
    vec2 = normalize(poses[:3, 2, :].sum(-1))
    vec0_avg = poses[:3, 0, :].sum(-1)
    c2w = np.concatenate([viewmatrix(vec2, vec0_avg, center), hwf], 1)
    
    return c2w

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt


def nearest_pose(p, poses):
    dists = np.sum(np.square(p[:3, 3:4] - poses[:3, 3, :]), 0)
    return np.argsort(dists)



def render_path_axis(c2w, up, ax, rad, focal, N):
    render_poses = []
    #print("c2w = ", c2w)
    center = c2w[:,3]
    hwf = c2w[:,4:5]
    v = c2w[:,ax] * rad    
    for t in np.linspace(-10.,10.,N+1)[:-1]:
        c = center + t * v
        z = normalize(c - (center - focal * c2w[:,2]))
        #print(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        view_matrix = viewmatrix(z, up, c) # view matrix for the point 'c' which add the rotation info i.e where to look from the point 'c' in the world
        render_poses.append(np.concatenate([view_matrix, hwf], 1))     
    return render_poses

def render_cylinder(c2w, up, rads, focal, N, r, theta1=0.0, theta2=np.pi):
    render_poses = []
    #camera_poses = []
    center = c2w[:,3] # avg center in world co-ord sys
    hwf = c2w[:,4:5]
    rads = np.array(list(rads) + [1.])    
    #heights = [i for i in range(-10., 10., 1)]  
    heights = list(np.linspace(-3., 3., 7)[:-1]) # -10 to 10 artifacts
    #heights = [0.0]  
    # pts_cam = []
    # pts_wld = []  
    for h in heights :
        for theta in np.linspace(theta1, theta2, N+1)[:-1]: # 0 to pi is the correct range           
            c = np.dot(c2w[:3,:4], np.array([-h, r * np.cos(theta), r * np.sin(theta), 1.]) * rads) # camera centre in world co-ord sys
            # I need to move the camera to location 'c'  and the orientation is inverse of view matrix at this location          
            z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
            # render poses are view matrices (pose of the camera in the world which is placed at the point c) which has the rotation info for the world point 'c'
            view_matrix = viewmatrix(z, up, c) # view of the world from camera postion 'c' which is nothing but w2c transform             
            render_poses.append(np.concatenate([view_matrix, hwf], 1))
    return render_poses

def render_path_grid(c2w, up, ax, rad_x, rad_y, focal, N, len):
    render_poses = []
    center = c2w[:,3]
    hwf = c2w[:,4:5]
    v_x = c2w[:,1] * rad_x
    v_y = c2w[:,0] * rad_y
    for t_y in np.linspace(-len, len, N+1)[:-1]:        
        tt_y = t_y * v_y      #plot_points_2d
        for t_x in np.linspace(-len, len, N+1)[:-1]:
            c = center + t_x * v_x + tt_y
            z = normalize(c - (center - focal * c2w[:,2]))
            render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))        
    return render_poses    
    
        
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]: # generates points on unit circle i.e r==1
        c = np.dot(c2w[:3,:4], np.array([-np.sin(theta), np.cos(theta), -np.sin(theta*zrate), 1.]) * rads) # camera centre in world co-ord sys     
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

 
    
def generate_render_path(poses, bds, comps=None, N=30, scenedir="./"):
    if comps is None:
        comps = [True]*7
    
    close_depth, inf_depth = bds[0, :].min()*.9, bds[1, :].max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz
    
    shrink_factor = .8
    zdelta = close_depth * .2

    c2w = poses_avg(poses)
    up = normalize(poses[:3, 0, :].sum(-1))

    tt = ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, -1)
    render_poses = [] 

    if comps[0]:
        render_poses += render_path_axis(c2w, up, 1, shrink_factor*rads[1], focal, N*50)
        #render_poses += render_path_axis(c2w, up, 1, rads, focal, N)
        name = "x_axis_path"
    if comps[1]:
        render_poses += render_path_axis(c2w, up, 0, shrink_factor*rads[0], focal, N*50)
        name = "y_axis_path"
    if comps[2]:
        render_poses += render_path_axis(c2w, up, 2, shrink_factor*zdelta, focal, N*50)
        name = "z_axis_path"    
    if comps[3]:
        rads[2] = zdelta
        render_poses += render_path_spiral(c2w, up, rads, focal, zdelta, 0., 1, N*2)
        name = "circle_axis_path"
    if comps[4]:
        rads[2] = zdelta
        render_poses += render_path_spiral(c2w, up, rads, focal, zdelta, .5, 2, N*4)
        name = "spiral_axis_path"
    if comps[5]:
        rads[2] = zdelta        
        render_poses += render_cylinder(c2w, up, rads, focal, N*4, 1.0, theta1=0.0, theta2=np.pi)
        name = "cylinder_path_1"
    if comps[6]:
        rads[2] = zdelta 
        #theta1=np.pi/6., theta2=5*np.pi/6.       
        render_poses += render_cylinder(c2w, up, rads, focal, N*4, 1.5, theta1=0., theta2=np.pi)
        name = "cylinder_path_1_5"        
    if comps[7]:
        rads[2] = zdelta        
        render_poses += render_cylinder(c2w, up, rads, focal, N*4, 2.0, theta1=np.pi/4., theta2=3*np.pi/4.)
        name = "cylinder_path_2"        
    if comps[8]:
        render_poses = render_path_grid(c2w, up, 0, shrink_factor*rads[1], shrink_factor*rads[0], focal, N, 1.)
        name = "grid_path_1" 
    if comps[9]:
        render_poses = render_path_grid(c2w, up, 0, shrink_factor*rads[1], shrink_factor*rads[0], focal, N, 1.5)
        name = "grid_path_1_5" 
    if comps[10]:
        render_poses = render_path_grid(c2w, up, 0, shrink_factor*rads[1], shrink_factor*rads[0], focal, N, 2.)
        name = "grid_path_2"                        
    
    render_poses = np.array(render_poses)
    path = scenedir + "/" + name + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    render_path_fig(poses, render_poses, scaling_factor=1., savepath=path)
    return render_poses



def render_path_fig(poses, render_poses, scaling_factor=1., savepath=None):
    c2w = poses_avg(poses)
    #tt = pts2cam(poses, c2w)


    plt.figure(figsize=(16,8))

    plt.subplot(2,2,1)
    tt = ptstocam(render_poses[:,:3,3], c2w) * scaling_factor # transform rendered points to cam points
    render_pts_cam = tt
    plt.title(label="Novel-Views(Blue) & Basis-View(Red) in X-Y plane of Camera co-ord system")
    plt.plot(tt[:,1], -tt[:,0], c='b')
    
    tt = ptstocam(poses[:3,3,:].T, c2w) * scaling_factor #transform world points to cam points
    basis_pts_cam = tt    
    plt.scatter(tt[:,1], -tt[:,0], c='r')
    plt.axis('equal')    

    plt.subplot(2,2,2)
    tt = ptstocam(render_poses[:,:3,3], c2w) * scaling_factor
    plt.title(label="Novel-Views(Blue) & Basis-Views(Red) in X-Z plane of Camera co-ord system")
    plt.plot(tt[:,1], tt[:,2], c='b')
    tt = ptstocam(poses[:3,3,:].T, c2w) * scaling_factor
    plt.scatter(tt[:,1], tt[:,2], c='r')
    plt.axis('equal')

    plt.subplot(2,2,3)
    tt = render_poses[:,:3,3]
    render_pts_wld = tt
    plt.title(label="Novel-Views(Blue) & Basis-Views(Red) in X-Y plane of World co-ord system")
    plt.plot(tt[:,1], -tt[:,0], c='b')
    tt = poses[:3,3,:].T
    basis_pts_wld = tt
    plt.scatter(tt[:,1], -tt[:,0], c='r')
    plt.axis('equal')

    plt.subplot(2,2,4)
    tt = render_poses[:,:3,3]
    plt.title(label="Novel-Views(Blue) & Basis-Views(Red) in X-Z plane of World co-ord system")
    plt.plot(tt[:,1], tt[:,2], c='b')
    tt = poses[:3,3,:].T
    plt.scatter(tt[:,1], tt[:,2], c='r')
    plt.axis('equal')

    plot_points_3d(render_pts_wld, render_pts_cam, centre_wld = None, centre_cam = None, scaling_factor=1., savepath=savepath, filename="render_3d") 
    plot_points_3d(basis_pts_wld, basis_pts_cam, centre_wld = None, centre_cam = None, scaling_factor=1., savepath=savepath, filename="basis_3d")

    if savepath is not None:
        plt.savefig(os.path.join(savepath, 'path_slices.png'))
        plt.close()

def render_semisphere_fig(poses, render_poses, scaling_factor=1., savepath=None):

    c2w = poses_avg(poses)
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))#* scaling_factor
    y = np.outer(np.sin(theta), np.sin(phi))#* scaling_factor
    z = np.outer(np.cos(theta), np.ones_like(phi))#* scaling_factor

    tt = ptstocam(render_poses[:,:3,3], c2w) #* scaling_factor # -y, x, z
    xi = tt[:,1]
    yi = -tt[:,0]
    zi = tt[:,2]
    #print([xi])
    #fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    #ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)
    if savepath is not None:
        plt.savefig(os.path.join(savepath, 'path_slices_3d.png'))
        plt.close()

def plot_points_2d(pts_wld, pts_cam, centre_wld = None, centre_cam = None, scaling_factor=1., savepath=None, filename=None):
    
    #pts_wld.shape should be n X 3
    #pts_cam.shape should be n X 3

    plt.figure(figsize=(12,4))
    #print("pts_wld.shape = ", pts_wld.shape)
    #print("pts_cam.shape = ", pts_cam.shape)

    plt.subplot(121)    
    plt.plot(pts_cam[:,1], -pts_cam[:,0], c='b')
    #plt.plot(pts_wld[:,0], pts_wld[:,1],c='r')
    plt.plot(pts_wld[:,1], -pts_wld[:,0],c='r')
    if centre_cam is not None:
        #plt.scatter(centre[0], centre[1], c='g') 
        plt.scatter(centre_cam[1], -centre_cam[0], c='b')
    if centre_wld is not None:        
        plt.scatter(centre_wld[1], -centre_wld[0], c='r')        

    

    plt.axis('equal')

    plt.subplot(122)    
    plt.plot(pts_cam[:,1], pts_cam[:,2])
    #plt.plot(pts_wld[:,0], pts_wld[:,2],c='r') 
    plt.plot(pts_wld[:,1], pts_wld[:,2],c='r') 
    if centre_cam is not None:
        #plt.scatter(centre[0], centre[2], c='g')  
        plt.scatter(centre_cam[1], centre_cam[2], c='b')
    if centre_wld is not None:
        #plt.scatter(centre[0], centre[2], c='g')  
        plt.scatter(centre_wld[1], centre_wld[2], c='r')        

    plt.axis('equal')
    if savepath is not None:
        plt.savefig(os.path.join(savepath, filename+'.png'))
        plt.close()

def plot_points_3d(pts_wld, pts_cam,  centre_wld = None, centre_cam = None, scaling_factor=1., centre= None, savepath=None, filename=None):
    
    #pts_wld.shape should be n X 3
    #pts_cam.shape should be n X 3

    x_w = pts_wld[:,1]
    y_w = -pts_wld[:,0]
    z_w = pts_wld[:,2]
    x_c = pts_cam[:,1]
    y_c = -pts_cam[:,0]
    z_c = pts_cam[:,2]
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    plt.title(label="Novel-Views & Basis-Views in World(Red) & Camera(Blue) co-ord system")
    ax.scatter(x_c, z_c, y_c, s=100, c='b', zorder=10)
    ax.scatter(x_w, z_w, y_w, s=100, c='r', zorder=10)
    if centre_cam is not None: 
        ax.scatter(centre_cam[1], -centre_cam[0], centre_cam[2], s=100, c='b', zorder=10)
    if centre_wld is not None:
        ax.scatter(centre_wld[1], -centre_wld[0], centre_wld[2], s=100, c='r', zorder=10)            
    if savepath is not None:
        plt.savefig(os.path.join(savepath, filename+'.png'))
        plt.close()