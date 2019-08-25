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
    print("pts_wld = ", pts)
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    # print("tt_in.shape = ", tt.shape)
    # print("tt_in = ", tt)
    # c2w_hm = np.concatenate((c2w[:3,:4], np.array([[0., 0., 0., 1.]])), axis=0)
    # w2c_hm = np.linalg.inv(c2w_hm)
    # tmp = np.array([[1.]*pts.shape[0]])
    # pts_hm = np.concatenate((pts.T, tmp), axis=0)
    # tt_hm = np.matmul(w2c_hm, pts_hm)
    # print("tt_hm.shape =", tt_hm.shape)
    # print("tt_hm.T =", tt_hm.T[:,0:3])
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
    for t in np.linspace(-1.,1.,N+1)[:-1]:
        #print("t =", t)
        c = center + t * v
        #print("c =", c)
        z = normalize(c - (center - focal * c2w[:,2]))
        #print("z =", z)
        #print(np.concatenate([viewmatrix(z, up, c), hwf], 1))
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

# def render_path_semisphere(c2w, up, rads, focal, zdelta, rots, N):
#     render_poses = []
#     rads = np.array(list(rads) + [1.])
#     hwf = c2w[:,4:5]
#     vec = np.random.randn(3, N) # (3, N)    
#     vec /= np.linalg.norm(vec, axis=0) #https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
#     #return vec
#     for col in range(N):
#         c_cam = vec[:,col] # x, y, z shape (3, )
#         c = np.dot(c2w[:3,:4], np.array([-c_cam[1], c_cam[0], -c_cam[2], 1.]) * rads)
#         z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
#         render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
#     return render_poses

### https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(samples=1,randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));
    print("increment = ", increment)

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2)) / 3
        #r = 10

        phi = ((i + rnd) % samples) * increment
        print("phi = ", phi)

        x = math.cos(phi) * r
        z = math.sin(phi) * r
        print("i = ", i)
        print("y = ", y)
        print("r = ", r)
        print("x = ", x)
        print("z = ", z)

        points.append(np.array([x,y,z]))  # list of np arrays    
    return points

def render_path_semisphere_fib(c2w, up, rads, focal, zdelta, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    points = fibonacci_sphere(N,False)    
    #vec /= np.linalg.norm(vec, axis=0) #https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
    #return vec
    for point in range(len(points)):
        c_cam = points[point] # x, y, z shape (3, )
        c = np.dot(c2w[:3,:4], np.array([-c_cam[1], c_cam[0], -c_cam[2], 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def render_path_semisphere(c2w, up, rads, focal, zdelta, rots, N):
    num_pts = N
    rads = np.array(list(rads) + [1.])
    render_poses = []
    hwf = c2w[:,4:5]
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    #phi = np.arccos(1 - 2*indices/num_pts)
    phi = [1.5714] * len(indices)
    theta = np.pi * (1 + 5**0.5) * indices
    #print("theta = ", theta)
    #print("theta = ", theta/3.14)
    
    r = 1
    x_c, y_c, z_c = r * np.cos(theta) * np.sin(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(phi);
    c2w_hm = np.concatenate((c2w[:3,:4], np.array([[0., 0., 0., 1.]])), axis=0)
    w2c_hm = np.linalg.inv(c2w_hm)
    #print("w2c_hm = ", w2c_hm)
    #print("c2w_hm = ", c2w_hm)
    cam_points = []
    for i in range(N):
        cam_points.append(np.dot(w2c_hm[:3,:4], np.array([-y_c[i], x_c[i], -z_c[i], 1.])))
    #print("cam_points = ", cam_points)
    #print("cam_points.shape = ", np.array(cam_points).shape)
    cam_points = np.array(cam_points)
    plt.figure().add_subplot(111, projection='3d').scatter(cam_points[:, 0], cam_points[:,1], cam_points[:,2]);
    plt.savefig(os.path.join("./", 'cam_points_3d.png'))
    for i in range(N):        
        c = np.dot(c2w[:3,:4], np.array([-cam_points[i][1], cam_points[i][0], -cam_points[i][2], 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses  
    
def generate_render_path(poses, bds, comps=None, N=30):
    if comps is None:
        comps = [True]*6
    
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
    plot_points_2d(pts_wld, pts_cam, scaling_factor=1., savepath="./",filename="basis_2d") # pts_wld and pts_cam == n X 3 shape
    plot_points_3d(pts_wld, pts_cam, scaling_factor=1., savepath="./", filename="basis_3d") # pts_wld and pts_cam == n X 3 shape

    render_poses = []    
    
    if comps[0]:
        render_poses += render_path_axis(c2w, up, 1, shrink_factor*rads[1], focal, N)
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
        render_poses += render_path_semisphere(c2w, up, rads, focal, zdelta, 1, N*10)
        name = "semisphere_path"
    
    render_poses = np.array(render_poses)
    path = "./" + name + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    render_path_fig(poses, render_poses, scaling_factor=1., savepath=path)
    #render_semisphere_fig(poses, render_poses, scaling_factor=1., savepath=path)  
    tt = ptstocam(render_poses[:,:3,3], c2w)
    pts_wld = render_poses[:,:3,3]
    pts_cam = tt 
    plot_points_2d(pts_wld, pts_cam, scaling_factor=1., savepath="./", filename="render_2d"+"_"+name) # pts_wld and pts_cam == n X 3 shape
    plot_points_3d(pts_wld, pts_cam, scaling_factor=1., savepath="./", filename="render_3d"+"_"+name) # pts_wld and pts_cam == n X 3 shape 
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

def plot_points_2d(pts_wld, pts_cam, scaling_factor=1., savepath=None, filename=None):
    
    #pts_wld.shape should be n X 3
    #pts_cam.shape should be n X 3

    plt.figure(figsize=(12,4))
    print("pts_wld.shape = ", pts_wld.shape)
    print("pts_cam.shape = ", pts_cam.shape)

    plt.subplot(121)    
    plt.plot(pts_cam[:,1], -pts_cam[:,0])
    plt.plot(pts_wld[:,0], pts_wld[:,1],c='r')    

    

    plt.axis('equal')

    plt.subplot(122)    
    plt.plot(pts_cam[:,1], pts_cam[:,2])
    plt.plot(pts_wld[:,0], pts_wld[:,2],c='r')   

    plt.axis('equal')
    if savepath is not None:
        plt.savefig(os.path.join(savepath, filename+'.png'))
        plt.close()

def plot_points_3d(pts_wld, pts_cam, scaling_factor=1., savepath=None, filename=None):
    
    #pts_wld.shape should be n X 3
    #pts_cam.shape should be n X 3

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
        plt.savefig(os.path.join(savepath, filename+'.png'))
        plt.close()