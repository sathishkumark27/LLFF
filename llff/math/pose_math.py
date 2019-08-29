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

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
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
    for t in np.linspace(-3.,3.,N+1)[:-1]:
        c = center + t * v
        z = normalize(c - (center - focal * c2w[:,2]))
        #print(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))        
    return render_poses

def render_cylinder(c2w, up, rads, focal, N):
    render_poses = []
    #print("rad = ", rad)
    center = c2w[:,3] # avg center in world co-ord sys
    hwf = c2w[:,4:5]
    rads = np.array(list(rads) + [1.])
    z_c = [-0.08, -0.06, -0.04, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08]
    for h in z_c :
        for theta in np.linspace(-np.pi , 0, N+1)[:-1]:        
            c = np.dot(c2w[:3,:4], np.array([-np.sin(theta), np.cos(theta), h, 1.]) * rads) # camera centre in world co-ord sys
            #c = np.dot(c2w[:3,:4], np.array([-np.sin(theta), np.cos(theta), -np.sin(theta*zrate), 1.]) * rads)
            print("theta = ", theta)
            print("c = ", c)        
            z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
            print("z = ", z)
            render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def render_path_grid(c2w, up, ax, rad, focal, N):
    render_poses = []
    #print("c2w = ", c2w)
    center = c2w[:,3]
    hwf = c2w[:,4:5]
    v_x = c2w[:,1] * rad
    v_z = c2w[:,2] * rad
    for t_z in np.linspace(-3.,10.,13)[:-1]:  
        tt_z = t_z * v_z      
        for t_x in np.linspace(-5.,5.,N+1)[:-1]:
            c = center + t_x * v_x + tt_z
            print("c, center, t_x, v_x, t_x*v_x ==", (c, center, t_x, v_x, t_x*v_x))
            z = normalize(c - (center - focal * c2w[:,2]))
            #print(np.concatenate([viewmatrix(z, up, c), hwf], 1))
            render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))        
    return render_poses    
    
        
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]: # generates points on unit circle i.e r==1
        c = np.dot(c2w[:3,:4], np.array([-np.sin(theta), np.cos(theta), -np.sin(theta*zrate), 1.]) * rads) # camera centre in world co-ord sys
        print("theta = ", theta)
        print("c = ", c)        
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        print("z = ", z)
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

 
    
def generate_render_path(poses, bds, comps=None, N=30):
    if comps is None:
        comps = [True]*7
    
    close_depth, inf_depth = bds[0, :].min()*.9, bds[1, :].max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz
    
    shrink_factor = .8
    zdelta = close_depth * .2

    c2w = poses_avg(poses)
    #print("pose_avg c2w.shape = ", c2w.shape)
    #print("pose_avg c2w = ", c2w)
    up = normalize(poses[:3, 0, :].sum(-1))
    #print("up.shape = ", up.shape)
    #print("up = ", up)
    
    tt = ptstocam(poses[:3,3,:].T, c2w).T
    #print("tt.shape = ", tt.shape) #shape = 3 X 22
    #print("tt = ", tt)
    rads = np.percentile(np.abs(tt), 90, -1)
    print("rads.shape = ", rads.shape)
    print("rads = ", rads)
    pts_cam = tt.T
    pts_wld = poses[:3,3,:].T    
    print("centre_wld = ", c2w[:3,3])
    centre_cam = ptstocam(c2w[:3,3], c2w)
    print("centre_cam = ", centre_cam)
    origin_wld = np.array([0,0,0])
    print("origin_wld = ", origin_wld)
    origin_wld_in_cam = ptstocam(origin_wld, c2w)
    print("origin_wld_in_cam = ", origin_wld_in_cam)
    plot_points_2d(pts_wld, pts_cam, centre_wld = c2w[:,3], centre_cam = centre_cam, scaling_factor=1., savepath="./",filename="basis_2d") # pts_wld and pts_cam == n X 3 shape
    plot_points_3d(pts_wld, pts_cam, centre_wld = c2w[:,3], centre_cam = centre_cam, scaling_factor=1., savepath="./", filename="basis_3d") # pts_wld and pts_cam == n X 3 shape

    render_poses = []    
    
    if comps[0]:
        render_poses += render_path_axis(c2w, up, 1, shrink_factor*rads[1], focal, N*10)
        #render_poses += render_path_axis(c2w, up, 1, rads, focal, N)
        name = "x_axis_path"
    if comps[1]:
        render_poses += render_path_axis(c2w, up, 0, shrink_factor*rads[0], focal, N)
        name = "y_axis_path"
    if comps[2]:
        render_poses += render_path_axis(c2w, up, 2, shrink_factor*zdelta, focal, N)
        name = "z_axis_path"
    
    rads[2] = zdelta
    if comps[3]:
        render_poses += render_path_spiral(c2w, up, rads, focal, zdelta, 0., 1, N*2)
        name = "circle_axis_path"
    if comps[4]:
        render_poses += render_path_spiral(c2w, up, rads, focal, zdelta, .5, 2, N*4)
        name = "spiral_axis_path"
    if comps[5]:
        render_poses += render_cylinder(c2w, up, rads, focal, N*4)
        name = "cylinder_path"
    if comps[6]:
        render_poses += render_path_grid(c2w, up, 0, shrink_factor*rads[0], focal, N*10)
        name = "grid_path"        
    
    render_poses = np.array(render_poses)
    path = "./" + name + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    render_path_fig(poses, render_poses, scaling_factor=1., savepath=path)
    #render_semisphere_fig(poses, render_poses, scaling_factor=1., savepath=path)  
    tt = ptstocam(render_poses[:,:3,3], c2w)
    pts_wld = render_poses[:,:3,3]
    pts_cam = tt 
    plot_points_2d(pts_wld, pts_cam, centre_wld = c2w[:,3], centre_cam = centre_cam, scaling_factor=1., savepath="./", filename="render_2d"+"_"+name) # pts_wld and pts_cam == n X 3 shape
    plot_points_3d(pts_wld, pts_cam, centre_wld = c2w[:,3], centre_cam = centre_cam, scaling_factor=1., savepath="./", filename="render_3d"+"_"+name) # pts_wld and pts_cam == n X 3 shape 
    return render_poses



def render_path_fig(poses, render_poses, scaling_factor=1., savepath=None):
    c2w = poses_avg(poses)
    #tt = pts2cam(poses, c2w)


    plt.figure(figsize=(12,4))

    plt.subplot(121)
    tt = ptstocam(render_poses[:,:3,3], c2w) * scaling_factor
    plt.plot(tt[:,1], -tt[:,0])
    #print("tt 1 = ", tt)
    tt = ptstocam(poses[:3,3,:].T, c2w) * scaling_factor
    plt.scatter(tt[:,1], -tt[:,0])
    #print("tt 2 = ", tt)
    plt.axis('equal')

    plt.subplot(122)
    tt = ptstocam(render_poses[:,:3,3], c2w) * scaling_factor
    plt.plot(tt[:,1], tt[:,2])
    tt = ptstocam(poses[:3,3,:].T, c2w) * scaling_factor
    plt.scatter(tt[:,1], tt[:,2])
    plt.axis('equal')

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
    #print([xi])
    #fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    #ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    ax.scatter(x_c, y_c, z_c, s=100, c='b', zorder=10)
    ax.scatter(y_w, -x_w, z_w, s=100, c='r', zorder=10)
    if centre_cam is not None:
        #ax.scatter(centre[0], centre[1], centre[2], s=100, c='g', zorder=10)   
        ax.scatter(centre_cam[1], -centre_cam[0], centre_cam[2], s=100, c='b', zorder=10)
    if centre_wld is not None:
        #ax.scatter(centre[0], centre[1], centre[2], s=100, c='g', zorder=10)   
        ax.scatter(centre_wld[1], -centre_wld[0], centre_wld[2], s=100, c='r', zorder=10)            
    if savepath is not None:
        plt.savefig(os.path.join(savepath, filename+'.png'))
        plt.close()