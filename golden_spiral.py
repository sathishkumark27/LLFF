# from numpy import pi, cos, sin, sqrt, arange
# import matplotlib.pyplot as pp

# num_pts = 1000
# indices = arange(0, num_pts, dtype=float) + 0.5
# print(indices)

# r = sqrt(indices/num_pts)
# theta = pi * (1 + 5**0.5) * indices

# pp.scatter(r*cos(theta), r*sin(theta))
# pp.show()

from numpy import pi, cos, sin, arccos, arange
import mpl_toolkits.mplot3d
import matplotlib.pyplot as pp

num_pts = 1000
indices = arange(0, num_pts, dtype=float) + 0.5
#phi = arccos(1 - 2*indices/num_pts) # o to pi latitude
phi = arccos(1 - indices/num_pts) # 0 to pi/2 latitude
print("phi = ", phi)
theta = pi * (1 + 5**0.5) * indices
print("theta = ", theta)
r = 2
x, y, z = r * cos(theta) * sin(phi), r * sin(theta) * sin(phi), r * cos(phi);

pp.figure().add_subplot(111, projection='3d').scatter(x, y, z);
pp.show()