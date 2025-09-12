import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan, sqrt
import matplotlib.pyplot as plt

def Ell_define(l1, inc, l2):
    sita1 = arctan(l2*sin(2.*inc) / (l1+l2*cos(2.*inc)))

    a_ell = (l1+l2)/2.
    b_ell = sqrt(l1 * l2 * sin(inc) ** 2)
    # b_ell = sqrt((l1+l2-l1*cos(sita1)-l2*cos(sita1-2.*inc)) * (l1+l2+l1*cos(sita1)+l2*cos(sita1-2*inc))) /2.

    sita3 = arcsin(l1*sin(sita1) / l2)

    return a_ell, b_ell, sita1, sita3
def calcEll_Yvalue(a, b, x):
    return sqrt(b**2. - (b*(x - sqrt(a**2. - b**2.)) /a)**2.)

def mirr_ray_intersection(coeffs, ray, source, negative=False):
    a, b, c, d, e, f, g, h, i, j = coeffs
    l, m, n = ray
    p, q, r = source

    A = a * l**2 + b * m**2 + c * n**2 + d * m * l + e * n * l + f * m * n
    B = (2 * a * p * l + 2 * b * q * m + 2 * c * r * n +
        d * (p * m + q * l) + e * (p * n + r * l) + f * (r * m + q * n) +
        g * l + h * m + i * n)
    C = (a * p**2 + b * q**2 + c * r**2 + d * p * q + e * p * r + f * q * r +
        g * p + h * q + i * r + j)

    D = B**2 - 4 * A * C
    if not all(x > 0 for x in D):
        point = np.full(source.shape, np.nan)
        return point

    if negative:
        t = (-B - np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    else:
        t = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)

    point = np.zeros_like(source)
    point[0, :] = t * l + p
    point[1, :] = t * m + q
    point[2, :] = t * n + r

    return point

def reflect_ray(ray, N):
    l, m, n = ray
    nx, ny, nz = N

    A = l * nx + m * ny + n * nz
    phai = ray - 2 * A * N

    phai = normalize_vector(phai)
    return phai

def normalize_vector(vector):
    norm = np.linalg.norm(vector, axis=0)
    return vector / norm if np.all(norm != 0) else vector

def norm_vector(coeffs, point):
    a, b, c, d, e, f, g, h, i, j = coeffs
    x, y, z = point

    N = np.zeros_like(point)
    N[0, :] = 2 * a * x + d * y + e * z + g
    N[1, :] = 2 * b * y + d * x + f * z + h
    N[2, :] = 2 * c * z + e * x + f * y + i

    N = normalize_vector(N)
    return N

def shift_x(coeffs, s):
    a, b, c, d, e, f, g, h, i, j = coeffs
    g2 = g - 2 * a * s
    h = h - d * s
    i = i - e * s
    j = j + a * s**2 - g * s
    return [a, b, c, d, e, f, g2, h, i, j]

def shift_y(coeffs, s):
    a, b, c, d, e, f, g, h, i, j = coeffs
    g = g - d * s
    h2 = h - 2 * b * s
    i = i - f * s
    j = j + b * s**2 - h * s
    return [a, b, c, d, e, f, g, h2, i, j]

def shift_z(coeffs, s):
    a, b, c, d, e, f, g, h, i, j = coeffs
    g = g - e * s
    h2 = h - f * s
    i2 = i - 2 * c * s
    j = j + c * s**2 - i * s
    return [a, b, c, d, e, f, g, h, i2, j]

def plane_ray_intersection(coeffs, ray, source):
    g, h, i, j = coeffs[6:10]
    l, m, n = ray
    p, q, r = source

    t = -(g * p + h * q + i * r + j) / (g * l + h * m + i * n)

    point = np.zeros_like(source)
    point[0, :] = t * l + p
    point[1, :] = t * m + q
    point[2, :] = t * n + r

    return point
def point_rotate_z(point, theta, center):
    x, y, z = point
    x, y, z = x - center[0], y - center[1], z - center[2]

    Cos, Sin = np.cos(theta), np.sin(theta)
    x1, y1, z1 = x * Cos + y * Sin, y * Cos - x * Sin, z

    x1, y1, z1 = x1 + center[0], y1 + center[1], z1 + center[2]
    return np.array([x1, y1, z1])
if __name__ == "__main__":
    num_points = 20
    source = np.zeros((3, num_points*num_points))
    div_s = 0.01 # Divergence angle in radians
    angle_z = np.linspace(-div_s, div_s, num_points)
    l1 = 4.0  # Distance from source to mirror 1
    l2 = 1  # Distance from mirror 1 to mirror 2
    inc = 0.2  # Grazing incidence angle in radians
    a_ell, b_ell, sita1, sita3 = Ell_define(l1, inc, l2)
    f_ell = sqrt(a_ell**2 - b_ell**2)
    x_center = l1* cos(sita1)
    y_center = l1* sin(sita1)
    mirr_length = 0.1  # Mirror length
    x1 = x_center - mirr_length/2
    x2 = x_center + mirr_length/2
    y1 = calcEll_Yvalue(a_ell, b_ell, x1)
    y2 = calcEll_Yvalue(a_ell, b_ell, x2)
    sita1_1 = arctan(y1/x1)
    sita1_2 = arctan(y2/x2)
    
    angle_y = np.linspace(sita1_1, sita1_2, num_points)
    
    angle_YY, angle_ZZ = np.meshgrid(angle_y, angle_z)

    vector = np.zeros((3, num_points, num_points))
    vector[0, :, :] = 1
    vector[1, :, :] = tan(angle_YY)
    vector[2, :, :] = tan(angle_ZZ)
    vector = vector.reshape(3, -1)
    vector = normalize_vector(vector)

    coeffs_ell = np.zeros(10)
    coeffs_ell[0] = 1./a_ell**2
    coeffs_ell[1] = 1./b_ell**2
    coeffs_ell[9] = -1.
    coeffs_ell = shift_x(coeffs_ell, f_ell)

    points_on_ell = mirr_ray_intersection(coeffs_ell, vector, source)
    N_ell = norm_vector(coeffs_ell, points_on_ell)
    reflected_ell = reflect_ray(vector, N_ell)

    dist_s_f = f_ell*2
    coeffs_det = np.zeros(10)
    coeffs_det[6] = 1.
    coeffs_det[9] = -dist_s_f
    points_on_det = plane_ray_intersection(coeffs_det, reflected_ell, points_on_ell)

    coeffs_det1 = np.zeros(10)
    coeffs_det1[6] = 1.
    coeffs_det1[9] = -dist_s_f +1e-8
    points_on_det1 = plane_ray_intersection(coeffs_det1, reflected_ell, points_on_ell)

    coeffs_det2 = np.zeros(10)
    coeffs_det2[6] = 1.
    coeffs_det2[9] = -dist_s_f -1e-8
    points_on_det2 = plane_ray_intersection(coeffs_det2, reflected_ell, points_on_ell)

    coeffs_source1 = np.zeros(10)
    coeffs_source1[6] = 1.
    coeffs_source1[9] = -0.5
    points_on_source1 = plane_ray_intersection(coeffs_source1, reflected_ell, points_on_ell)
    coeffs_source2 = np.zeros(10)
    coeffs_source2[6] = 1.
    coeffs_source2[9] = 0.5
    points_on_source2 = plane_ray_intersection(coeffs_source2, reflected_ell, points_on_ell)


    # plt.figure(figsize=(6, 6))
    # vector_det1todet2 = points_on_det2 - points_on_det1
    # mean_angle = arctan(np.mean(vector_det1todet2[1, :]/vector_det1todet2[0, :]))
    # ### dist_s_f, 0中心に回転
    # points_on_det1_rot = point_rotate_z(points_on_det1, mean_angle, [dist_s_f, 0, 0])
    # points_on_det2_rot = point_rotate_z(points_on_det2, mean_angle, [dist_s_f, 0, 0])
    # plt.quiver(points_on_det1_rot[0, :], points_on_det1_rot[1, :], points_on_det2_rot[0, :] - points_on_det1_rot[0, :], points_on_det2_rot[1, :] - points_on_det1_rot[1, :], angles='xy', scale_units='xy', scale=1, color='b', width=0.002)
    # plt.xlabel('X (m)')
    # plt.ylabel('Y (m)')
    
    points_on_source1 = points_on_source1.reshape(3, num_points, num_points)
    points_on_source2 = points_on_source2.reshape(3, num_points, num_points)


    plt.figure(figsize=(6, 6))
    # plt.quiver(points_on_source1[0, :, :].flatten(), points_on_source1[2, :, :].flatten(), points_on_source2[0, :, :].flatten() - points_on_source1[0, :, :].flatten(), points_on_source2[2, :, :].flatten() - points_on_source1[2, :, :].flatten(), angles='xy', scale_units='xy', scale=1, color='b', width=0.0002)
    # plt.quiver(points_on_source1[0, 0, :], points_on_source1[2, 0, :], points_on_source2[0, 0, :] - points_on_source1[0, 0, :], points_on_source2[2, 0, :] - points_on_source1[2, 0, :], angles='xy', scale_units='xy', scale=1, color='r', width=0.002)
    # plt.quiver(points_on_source1[0, -1, :], points_on_source1[2, -1, :], points_on_source2[0, -1, :] - points_on_source1[0, -1, :], points_on_source2[2, -1, :] - points_on_source1[2, -1, :], angles='xy', scale_units='xy', scale=1, color='r', width=0.002)
    plt.quiver(points_on_source1[0, :, 0], points_on_source1[2, :, 0], points_on_source2[0, :, 0] - points_on_source1[0, :, 0], points_on_source2[2, :, 0] - points_on_source1[2, :, 0], angles='xy', scale_units='xy', scale=1, color='g', width=0.002)
    plt.quiver(points_on_source1[0, :, -1], points_on_source1[2, :, -1], points_on_source2[0, :, -1] - points_on_source1[0, :, -1], points_on_source2[2, :, -1] - points_on_source1[2, :, -1], angles='xy', scale_units='xy', scale=1, color='g', width=0.002)
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')

    # plt.figure(figsize=(6, 6))
    # plt.quiver(points_on_source1[0, :, :].flatten(), points_on_source1[1, :, :].flatten(), points_on_source2[0, :, :].flatten() - points_on_source1[0, :, :].flatten(), points_on_source2[1, :, :].flatten() - points_on_source1[1, :, :].flatten(), angles='xy', scale_units='xy', scale=1, color='b', width=0.0002)
    # plt.quiver(points_on_source1[0, 0, :], points_on_source1[1, 0, :], points_on_source2[0, 0, :] - points_on_source1[0, 0, :], points_on_source2[1, 0, :] - points_on_source1[1, 0, :], angles='xy', scale_units='xy', scale=1, color='r', width=0.002)
    # plt.quiver(points_on_source1[0, -1, :], points_on_source1[1, -1, :], points_on_source2[0, -1, :] - points_on_source1[0, -1, :], points_on_source2[1, -1, :] - points_on_source1[1, -1, :], angles='xy', scale_units='xy', scale=1, color='r', width=0.002)
    # plt.quiver(points_on_source1[0, :, 0], points_on_source1[1, :, 0], points_on_source2[0, :, 0] - points_on_source1[0, :, 0], points_on_source2[1, :, 0] - points_on_source1[1, :, 0], angles='xy', scale_units='xy', scale=1, color='g', width=0.002)
    # plt.quiver(points_on_source1[0, :, -1], points_on_source1[1, :, -1], points_on_source2[0, :, -1] - points_on_source1[0, :, -1], points_on_source2[1, :, -1] - points_on_source1[1, :, -1], angles='xy', scale_units='xy', scale=1, color='g', width=0.002)
    # plt.xlabel('X (m)')
    # plt.ylabel('Y (m)')

    plt.show()

    
    points_on_ell = points_on_ell.reshape(3, num_points, num_points)
    points_on_det = points_on_det.reshape(3, num_points, num_points)
    
    plt.figure(figsize=(6, 6))
    plt.plot(points_on_det[1, :, :], points_on_det[2, :, :], marker='o', linestyle='', color='b')
    plt.xlabel('Y (m)')
    plt.ylabel('Z (m)')
    
    ### side view
    plt.figure(figsize=(6, 6))
    plt.plot(points_on_det[0, :, :], points_on_det[2, :, :], marker='o', linestyle='', color='b')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')

    ### top view
    plt.figure(figsize=(6, 6))
    plt.plot(points_on_det[0, :, :], points_on_det[1, :, :], marker='o', linestyle='', color='b')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    ### side view ray ell と detectorをつなぐ線を描画
    plt.figure(figsize=(6, 6))
    plt.quiver(points_on_ell[0, :, :], points_on_ell[1, :, :], points_on_det[0, :, :] - points_on_ell[0, :, :], points_on_det[1, :, :] - points_on_ell[1, :, :], angles='xy', scale_units='xy', scale=1, color='b', width=0.002)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.show()