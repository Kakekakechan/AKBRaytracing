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

def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)  # 単位ベクトルに正規化
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    ux, uy, uz = axis

    # クロス積行列
    cross_product_matrix = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])

    # ロドリゲスの回転行列
    R = np.eye(3) * cos_theta + (1 - cos_theta) * np.outer(axis, axis) + cross_product_matrix * sin_theta
    return R

def rotate_general_axis(coeffs, axis, theta, center):
    # 移動と回転を行う
    coeffs = shift_x(coeffs, -center[0])
    coeffs = shift_y(coeffs, -center[1])
    coeffs = shift_z(coeffs, -center[2])

    # 係数の分解
    a, b, c, d, e, f, g, h, i, j = coeffs

    # 回転行列を取得
    R = rotation_matrix(axis, theta).T

    a1 = a * R[0,0]**2    + b * R[1,0]**2    + c * R[2,0]**2    + d * R[0,0] * R[1,0] + e * R[2,0] * R[0,0] + f * R[1,0] * R[2,0]
    b1 = a * R[0,1]**2    + b * R[1,1]**2    + c * R[2,1]**2    + d * R[0,1] * R[1,1] + e * R[2,1] * R[0,1] + f * R[1,1] * R[2,1]
    c1 = a * R[0,2]**2    + b * R[1,2]**2    + c * R[2,2]**2    + d * R[0,2] * R[1,2] + e * R[2,2] * R[0,2] + f * R[1,2] * R[2,2]
    d1 = 2*a*R[0,0]*R[0,1]+ 2*b*R[1,0]*R[1,1]+ 2*c*R[2,0]*R[2,1]+ d*(R[0,1]*R[1,0]+R[0,0]*R[1,1]) + e*(R[2,1]*R[0,0]+R[2,0]*R[0,1]) + f*(R[1,1]*R[2,0]+R[1,0]*R[2,1])
    e1 = 2*a*R[0,0]*R[0,2]+ 2*b*R[1,0]*R[1,2]+ 2*c*R[2,0]*R[2,2]+ d*(R[0,2]*R[1,0]+R[0,0]*R[1,2]) + e*(R[2,2]*R[0,0]+R[2,0]*R[0,2]) + f*(R[1,2]*R[2,0]+R[1,0]*R[2,2])
    f1 = 2*a*R[0,1]*R[0,2]+ 2*b*R[1,1]*R[1,2]+ 2*c*R[2,1]*R[2,2]+ d*(R[0,1]*R[1,2]+R[0,2]*R[1,1]) + e*(R[2,1]*R[0,2]+R[2,2]*R[0,1]) + f*(R[1,1]*R[2,2]+R[1,2]*R[2,1])
    g1 = g * R[0, 0] + h * R[1, 0] + i * R[2, 0]
    h1 = g * R[0, 1] + h * R[1, 1] + i * R[2, 1]
    i1 = g * R[0, 2] + h * R[1, 2] + i * R[2, 2]
    j1 = j
    coeffs_new = [a1, b1, c1, d1, e1, f1, g1, h1, i1, j1]

    # 元の位置に戻す
    coeffs_new = shift_x(coeffs_new, center[0])
    coeffs_new = shift_y(coeffs_new, center[1])
    coeffs_new = shift_z(coeffs_new, center[2])

    return coeffs_new, rotation_matrix(axis, theta)

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
def KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v , gapf=0.):

    a_h, b_h, sita1h, sita3h = Ell_define(l1h, inc_h, l2h)
    s2f_h = sqrt(a_h**2. - b_h**2.) * 2.    ###horizontal source-focus

    xh_s = l1h * cos(sita1h) - mlen_h/2.    ###Hmirror start
    xh_e = l1h * cos(sita1h) + mlen_h/2.    ###Hmirror end
    yh_s = calcEll_Yvalue(a_h,b_h,xh_s)
    yh_e = calcEll_Yvalue(a_h,b_h,xh_e)
    accept_h = abs(yh_e - yh_s)
    NA_h = sin(abs(arctan(yh_e/(s2f_h-xh_e)) - arctan(yh_s/(s2f_h-xh_s))))/2.

    l1v = l1h + (l2h - wd_v - mlen_v/2.)-gapf         ###1st guess of l1v
    l2v = wd_v + mlen_v/2.                  ###1st guess of l2v

    while True:
        a_v, b_v, sita1v, sita3v = Ell_define(l1v, inc_v, l2v)
        s2f_v = sqrt(a_v**2. - b_v**2.) * 2.    ###horizontal source-focus

        diff = s2f_h - s2f_v - gapf
        if abs(diff) < 1e-9:
            break
        else:
            l1v += diff*0.9

    xv_s = l1v * cos(sita1v) - mlen_v/2.    ###Vmirror start
    xv_e = l1v * cos(sita1v) + mlen_v/2.    ###Vmirror end
    yv_s = calcEll_Yvalue(a_v,b_v,xv_s)
    yv_e = calcEll_Yvalue(a_v,b_v,xv_e)
    accept_v = abs(yv_e - yv_s)
    NA_v = sin(abs(arctan(yv_e/(s2f_v-xv_e)) - arctan(yv_s/(s2f_v-xv_s))))/2.

    gap = xv_s - xh_e
    if gap < 0.:
        print(' ')
        print('!!!!!!!!!!!!')
        print('gap = '+str(gap)+' , mirrors interferes')
        print('!!!!!!!!!!!!')

    return a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap]
class ell():
    def __init__(self,l1,l2,inc,mirr_length):
        print('set')
        self.a_ell, self.b_ell, self.sita1, self.sita3 = Ell_define(l1, inc, l2)
        self.f_ell = sqrt(self.a_ell**2 - self.b_ell**2)
        self.x_center = l1* cos(self.sita1)
        self.y_center = l1* sin(self.sita1)
        self.x1 = self.x_center - mirr_length/2
        self.x2 = self.x_center + mirr_length/2
        self.y1 = calcEll_Yvalue(self.a_ell, self.b_ell, self.x1)
        self.y2 = calcEll_Yvalue(self.a_ell, self.b_ell, self.x2)
        self.p1 = sqrt(self.x1**2+self.y1**2)
        self.p2 = sqrt(self.x2**2+self.y2**2)
        self.sita1_1 = arctan(self.y1/self.x1)
        self.sita1_2 = arctan(self.y2/self.x2)
        self.sita3_1 = arctan(self.y1/(2*self.f_ell-self.x1))
        self.sita3_2 = arctan(self.y2/(2*self.f_ell-self.x2))

        self.s0_prime_x1 = self.p1*(cos(self.sita1_1)-cos(self.sita3_1))
        self.s0_prime_x2 = self.p2*(cos(self.sita1_2)-cos(self.sita3_2))
        self.dist_s_f = self.f_ell*2
        print('s0_prime_x1',self.s0_prime_x1)
        print('s0_prime_x2',self.s0_prime_x2)
        return
    def coeffs(self,option):
        self.coeffs = np.zeros(10)
        self.coeffs[0] = 1./self.a_ell**2
        if option == 'y':
            self.coeffs[1] = 1./self.b_ell**2
        else:
            self.coeffs[2] = 1./self.b_ell**2
        self.coeffs[9] = -1.
        self.coeffs = shift_x(self.coeffs, self.f_ell)
        return
    def calc_reflect(self,inc_vector,inc_points):
        self.points = mirr_ray_intersection(self.coeffs, inc_vector,inc_points)
        self.N_ell = norm_vector(self.coeffs, self.points)
        self.reflect = reflect_ray(inc_vector, self.N_ell)
        return
class PlanePoints:
    def __init__(self,position,delta,inc_ray,inc_points):
        coeffs_det = np.zeros(10)
        coeffs_det[6] = 1.
        coeffs_det[9] = -position

        self.points0 = plane_ray_intersection(coeffs_det, inc_ray, inc_points)

        coeffs_det1 = np.zeros(10)
        coeffs_det1[6] = 1.
        coeffs_det1[9] = -position +delta
        self.points1 = plane_ray_intersection(coeffs_det1, inc_ray, inc_points)

        coeffs_det2 = np.zeros(10)
        coeffs_det2[6] = 1.
        coeffs_det2[9] = -position -delta
        self.points2 = plane_ray_intersection(coeffs_det2, inc_ray, inc_points)
        return
def plot_3_view(points,title):
    fig, axs = plt.subplots(2,2)
    plt.suptitle(title)
    axs = axs.ravel()
    axs[0].plot(points[1, :, :], points[2, :, :], marker='o', linestyle='', color='b')
    axs[0].set_xlabel('Y (m)')
    axs[0].set_ylabel('Z (m)')

    ### side view
    axs[1].plot(points[0, :, :], points[2, :, :], marker='o', linestyle='', color='b')
    axs[1].set_xlabel('X (m)')
    axs[1].set_ylabel('Z (m)')

    ### top view
    axs[2].plot(points[0, :, :], points[1, :, :], marker='o', linestyle='', color='b')
    axs[2].set_xlabel('X (m)')
    axs[2].set_ylabel('Y (m)')
    plt.tight_layout()
    return
def plot_side_topview(list):
    fig, axs = plt.subplots(2,1)
    axs = axs.ravel()
    print('len(list)',len(list))
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    for i in range(len(list)):
        points = list[i]
        ### side view
        axs[0].plot(points[0, :, :], points[2, :, :], marker='o', linestyle='', color=colors[i])
        axs[0].set_xlabel('X (m)')
        axs[0].set_ylabel('Z (m)')

        ### top view
        axs[1].plot(points[0, :, :], points[1, :, :], marker='o', linestyle='', color=colors[i])
        axs[1].set_xlabel('X (m)')
        axs[1].set_ylabel('Y (m)')
    plt.tight_layout()
    plt.axis('equal')
    return
if __name__ == "__main__":
    num_points = 20
    source = np.zeros((3, num_points*num_points))
    div_s = 0.01 # Divergence angle in radians
    angle_z = np.linspace(-div_s, div_s, num_points)


    l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v = [np.float64(146.), np.float64(0.086),  np.float64(0.214), np.float64(0.060), np.float64(0.0211), np.float64(0.21), np.float64(0.0232)] ### 最適化の例
    inc_h /= 20
    inc_v /= 20
    a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap] = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v)


    # l1 = 146.0  # Distance from source to mirror 1
    # l2 = 0.1  # Distance from mirror 1 to mirror 2
    # inc = 0.2  # Grazing incidence angle in radians
    # mirr_length = 0.1  # Mirror length

    ell_v = ell(l1h, l2h, inc_h, mlen_h)

    vir_slen_h = (ell_v.s0_prime_x1 + ell_v.s0_prime_x2)/2.
    print('vir_slen_h',vir_slen_h)

    a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap] = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v, gapf=vir_slen_h)


    ell_h = ell(l1v, l2v, inc_v, mlen_v)

    ### v
    angle_y = np.linspace(ell_v.sita1_1, ell_v.sita1_2, num_points)
    angle_z = np.linspace(ell_h.sita1_1, ell_h.sita1_2, num_points)
    mean_angle_z = np.mean(angle_z)
    angle_z -= mean_angle_z
    angle_YY, angle_ZZ = np.meshgrid(angle_y, angle_z)
    print('angle_YY',np.min(angle_YY),np.max(angle_YY))
    print('angle_ZZ',np.min(angle_ZZ),np.max(angle_ZZ))

    ### rays definition
    vector = np.zeros((3, num_points, num_points))
    vector[0, :, :] = 1
    vector[1, :, :] = tan(angle_YY)
    vector[2, :, :] = tan(angle_ZZ)
    vector = vector.reshape(3, -1)
    vector = normalize_vector(vector)

    ### ell_v definition
    ell_v.coeffs('y')
    ell_v.calc_reflect(vector,source)

    ell_h.coeffs('z')
    axis_x = [1, 0, 0]
    axis_y = [0, 1, 0]
    axis_z = [0, 0, 1]
    theta = (ell_v.sita3_1 + ell_v.sita3_2)/2
    print('theta',theta)
    ell_h.coeffs, r_matrix = rotate_general_axis(ell_h.coeffs, axis_y, mean_angle_z, [0, 0, 0])
    ell_h.coeffs, r_matrix = rotate_general_axis(ell_h.coeffs, axis_z, -theta, [ell_v.dist_s_f, 0, 0])

    ell_h.calc_reflect(ell_v.reflect,ell_v.points)

    # ell_h.calc_reflect(vector,source)
    # focus_h_solo = PlanePoints(ell_h.dist_s_f*np.cos(mean_angle_z),1e-8,ell_h.reflect,ell_h.points)
    # focus_h_solo.points0 = focus_h_solo.points0.reshape(3, num_points, num_points)
    # ell_h_matrix = ell_h.points.reshape(3, num_points, num_points)
    # plot_3_view(focus_h_solo.points0,'focus_h_solo')
    # plt.figure()
    # plt.plot(focus_h_solo.points0[0, :, :], focus_h_solo.points0[2, :, :], marker='o', linestyle='', color='b')
    # plt.plot(ell_h_matrix[0, :, :], ell_h_matrix[2, :, :], marker='o', linestyle='', color='r')
    # plt.xlabel('X (m)')
    # plt.ylabel('Z (m)')
    # plt.figure()
    # plt.plot(focus_h_solo.points0[0, :, :], focus_h_solo.points0[1, :, :], marker='o', linestyle='', color='b')
    # plt.plot(ell_h_matrix[0, :, :], ell_h_matrix[1, :, :], marker='o', linestyle='', color='r')
    # plt.xlabel('X (m)')
    # plt.ylabel('Y (m)')
    # plt.show()

    focus = PlanePoints(ell_v.dist_s_f,1e-4,ell_h.reflect,ell_h.points)

    # ell_v.calc_points_plane(ell_v.dist_s_f,1e-8)
    focus_v_solo = PlanePoints(ell_v.dist_s_f,1e-8,ell_v.reflect,ell_v.points)
    virtual_source_h = PlanePoints(0.,ell_v.s0_prime_x2*2,ell_v.reflect,ell_v.points)

    virtual_source_h.points1 = virtual_source_h.points1.reshape(3, num_points, num_points)
    virtual_source_h.points2 = virtual_source_h.points2.reshape(3, num_points, num_points)
    ell_v.points = ell_v.points.reshape(3, num_points, num_points)
    ell_h.points = ell_h.points.reshape(3, num_points, num_points)
    focus_v_solo.points0 = focus_v_solo.points0.reshape(3, num_points, num_points)
    focus.points0 = focus.points0.reshape(3, num_points, num_points)

    list = [ell_v.points, ell_h.points, focus.points0]
    plot_side_topview(list)
    # plt.show()

    plt.figure(figsize=(6, 6))
    plt.quiver(virtual_source_h.points1[0, :, 0], virtual_source_h.points1[2, :, 0], virtual_source_h.points2[0, :, 0] - virtual_source_h.points1[0, :, 0], virtual_source_h.points2[2, :, 0] - virtual_source_h.points1[2, :, 0], angles='xy', scale_units='xy', scale=1, color='g', width=0.002)
    plt.quiver(virtual_source_h.points1[0, :, -1], virtual_source_h.points1[2, :, -1], virtual_source_h.points2[0, :, -1] - virtual_source_h.points1[0, :, -1], virtual_source_h.points2[2, :, -1] - virtual_source_h.points1[2, :, -1], angles='xy', scale_units='xy', scale=1, color='g', width=0.002)
    plt.scatter(ell_v.s0_prime_x1,0)
    plt.scatter(ell_v.s0_prime_x2,0)
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.xlim([0,ell_v.s0_prime_x2*2])
    plt.title('virtual_source_h')
    
    focus.points1 = focus.points1.reshape(3, num_points, num_points)
    focus.points2 = focus.points2.reshape(3, num_points, num_points)
    plt.figure(figsize=(6, 6))
    plt.quiver(focus.points1[0, :, 0], focus.points1[2, :, 0], focus.points2[0, :, 0] - focus.points1[0, :, 0], focus.points2[2, :, 0] - focus.points1[2, :, 0], angles='xy', scale_units='xy', scale=1, color='g', width=0.002)
    plt.quiver(focus.points1[0, :, num_points//2], focus.points1[2, :, num_points//2], focus.points2[0, :, num_points//2] - focus.points1[0, :, num_points//2], focus.points2[2, :, num_points//2] - focus.points1[2, :, num_points//2], angles='xy', scale_units='xy', scale=1, color='g', width=0.002)
    plt.quiver(focus.points1[0, :, -1], focus.points1[2, :, -1], focus.points2[0, :, -1] - focus.points1[0, :, -1], focus.points2[2, :, -1] - focus.points1[2, :, -1], angles='xy', scale_units='xy', scale=1, color='r', width=0.002)
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.xlim([focus.points1[0, :].min(),focus.points2[0, :].max()])
    plt.title('focus')
    plt.figure(figsize=(6, 6))
    # plt.quiver(focus.points1[0, :, 0], focus.points1[1, :, 0], focus.points2[0, :, 0] - focus.points1[0, :, 0], focus.points2[1, :, 0] - focus.points1[1, :, 0], angles='xy', scale_units='xy', scale=1, color='g', width=0.002)
    # plt.quiver(focus.points1[0, :, num_points//2], focus.points1[1, :, num_points//2], focus.points2[0, :, num_points//2] - focus.points1[0, :, num_points//2], focus.points2[1, :, num_points//2] - focus.points1[1, :, num_points//2], angles='xy', scale_units='xy', scale=1, color='g', width=0.002)
    # plt.quiver(focus.points1[0, :, -1], focus.points1[1, :, -1], focus.points2[0, :, -1] - focus.points1[0, :, -1], focus.points2[1, :, -1] - focus.points1[1, :, -1], angles='xy', scale_units='xy', scale=1, color='r', width=0.002)
    plt.quiver(focus.points1[0, 0, :], focus.points1[1, 0, :], focus.points2[0, 0, :] - focus.points1[0, 0, :], focus.points2[1, 0, :] - focus.points1[1, 0, :], angles='xy', scale_units='xy', scale=1, color='g', width=0.002)
    plt.quiver(focus.points1[0, num_points//2, :], focus.points1[1, num_points//2, :], focus.points2[0, num_points//2, :] - focus.points1[0, num_points//2, :], focus.points2[1, num_points//2, :] - focus.points1[1, num_points//2, :], angles='xy', scale_units='xy', scale=1, color='g', width=0.002)
    plt.quiver(focus.points1[0, -1, :], focus.points1[1, -1, :], focus.points2[0, -1, :] - focus.points1[0, -1, :], focus.points2[1, -1, :] - focus.points1[1, -1, :], angles='xy', scale_units='xy', scale=1, color='r', width=0.002)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.xlim([focus.points1[0, :].min(),focus.points2[0, :].max()])
    plt.title('focus')
    plt.show()



    plot_3_view(focus_v_solo.points0,'focus_v_solo')
    plot_3_view(focus.points0,'focus')


    ### side view ray ell と detectorをつなぐ線を描画
    plt.figure(figsize=(6, 6))
    plt.quiver(ell_v.points[0, :, :], ell_v.points[1, :, :], focus_v_solo.points0[0, :, :] - ell_v.points[0, :, :], focus_v_solo.points0[1, :, :] - ell_v.points[1, :, :], angles='xy', scale_units='xy', scale=1, color='b', width=0.002)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.show()
