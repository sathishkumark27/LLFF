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