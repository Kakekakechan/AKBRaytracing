import os
os.environ['NUMBA_NUM_THREADS'] = '10'
os.environ['OMP_NUM_THREADS'] = '10'
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import numba
from numba import njit, prange
import matplotlib.cm as cm
from numpy import abs, sin, cos,tan, arcsin,arccos,arctan, sqrt, pi
from mpmath import mp, mpf, sin, cos, tan, sqrt, pi, fabs, asin, acos, atan, isnan, nan, matrix, nstr
mp.dps = 20
from matplotlib.backends.backend_pdf import PdfPages
import KB_design_NAbased
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from PIL import Image
# from zernike import RZern
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider
# import cupy as cp
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import legendre_fit as lf
# # コア数を設定
import tifffile
import cv2
global option_AKB
global wave_num_H
global wave_num_V
global downsample_h1
global downsample_v1
global downsample_h2
global downsample_v2
global downsample_h_f
global downsample_v_f
global option_2mirror
global option_rotate
global option_HighNA
global defocusForWave
global option_avrgsplt
global optKBdesign
global directory_name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
directory_name = f"output_{timestamp}"
# 新しいフォルダを作成
os.makedirs(directory_name, exist_ok=True)
optKBdesign=True
option_2mirror =True
option_rotate = True
option_avrgsplt = False
# （0ならそのまま、2なら半分、4なら1/4,、6なら1/8）
downsample_h1 = 0
downsample_v1 = 0
downsample_h2 = 0
downsample_v2 = 0
downsample_h_f = 0
downsample_v_f = 0
unit = 129
wave_num_H=unit
wave_num_V=unit
# option_AKB = True
option_AKB = True
option_wolter_3_1 = True
option_wolter_3_3_tandem = False
option_HighNA = True
global LowNAratio
LowNAratio = 1.
defocusForWave = 1e-3
def crop(start,end,mabiki):
    # 元の配列（例として 0 から end までの範囲）
    original_array = list(range(end + 1))
    # 間引き率を設定 (例: 2 つおきに取り出す場合)
    skip_rate = mabiki
    # 範囲内の値を間引く
    thinned_array = original_array[start:end:skip_rate]
    return thinned_array


def shift_x(coeffs, s):
    a, b, c, d, e, f, g, h, i, j = coeffs
    g2 = g - 2 * a * s
    h = h - d * s
    i = i - e * s
    j = j + a * s**2 - g * s
    return matrix([a, b, c, d, e, f, g2, h, i, j])

def shift_y(coeffs, s):
    a, b, c, d, e, f, g, h, i, j = coeffs
    g = g - d * s
    h2 = h - 2 * b * s
    i = i - f * s
    j = j + b * s**2 - h * s
    return matrix([a, b, c, d, e, f, g, h2, i, j])

def shift_z(coeffs, s):
    a, b, c, d, e, f, g, h, i, j = coeffs
    g = g - e * s
    h2 = h - f * s
    i2 = i - 2 * c * s
    j = j + c * s**2 - i * s
    return matrix([a, b, c, d, e, f, g, h, i2, j])

def rotate_x(coeffs, theta, center):
    coeffs = shift_x(coeffs, -center[0])
    coeffs = shift_y(coeffs, -center[1])
    coeffs = shift_z(coeffs, -center[2])

    a, b, c, d, e, f, g, h, i, j = coeffs
    Cos = cos(theta)
    Sin = sin(theta)

    a1 = a
    b1 = b * Cos**2 + c * Sin**2 - f * Sin * Cos
    c1 = b * Sin**2 + c * Cos**2 + f * Sin * Cos
    d1 = d * Cos - e * Sin
    e1 = d * Sin + e * Cos
    f1 = b * sin(2 * theta) - c * sin(2 * theta) + f * cos(2 * theta)
    g1 = g
    h1 = h * Cos - i * Sin
    i1 = h * Sin + i * Cos
    j1 = j

    coeffs = matrix([a1, b1, c1, d1, e1, f1, g1, h1, i1, j1])
    coeffs = shift_x(coeffs, center[0])
    coeffs = shift_y(coeffs, center[1])
    coeffs = shift_z(coeffs, center[2])
    return coeffs

def rotate_y(coeffs, theta, center):
    coeffs = shift_x(coeffs, -center[0])
    coeffs = shift_y(coeffs, -center[1])
    coeffs = shift_z(coeffs, -center[2])

    a, b, c, d, e, f, g, h, i, j = coeffs
    Cos = cos(theta)
    Sin = sin(theta)

    a1 = a * Cos**2 + c * Sin**2 + e * Sin * Cos
    b1 = b
    c1 = a * Sin**2 + c * Cos**2 - e * Sin * Cos
    d1 = d * Cos + f * Sin
    e1 = -a * sin(2 * theta) + c * sin(2 * theta) + e * cos(2 * theta)
    f1 = -d * Sin + f * Cos
    g1 = g * Cos + i * Sin
    h1 = h
    i1 = i * Cos - g * Sin
    j1 = j

    coeffs = matrix([a1, b1, c1, d1, e1, f1, g1, h1, i1, j1])
    coeffs = shift_x(coeffs, center[0])
    coeffs = shift_y(coeffs, center[1])
    coeffs = shift_z(coeffs, center[2])
    return coeffs

def rotate_z(coeffs, theta, center):
    coeffs = shift_x(coeffs, -center[0])
    coeffs = shift_y(coeffs, -center[1])
    coeffs = shift_z(coeffs, -center[2])

    a, b, c, d, e, f, g, h, i, j = coeffs
    Cos = cos(theta)
    Sin = sin(theta)

    a1 = a * Cos**2 + b * Sin**2 - d * Sin * Cos
    b1 = b * Cos**2 + a * Sin**2 + d * Sin * Cos
    c1 = c
    d1 = a * sin(2 * theta) - b * sin(2 * theta) + d * cos(2 * theta)
    e1 = e * Cos - f * Sin
    f1 = e * Sin + f * Cos
    g1 = g * Cos - h * Sin
    h1 = g * Sin + h * Cos
    i1 = i
    j1 = j

    coeffs = matrix([a1, b1, c1, d1, e1, f1, g1, h1, i1, j1])
    coeffs = shift_x(coeffs, center[0])
    coeffs = shift_y(coeffs, center[1])
    coeffs = shift_z(coeffs, center[2])
    return coeffs

def rotation_matrix(axis, theta):
    axis = axis / mpmath_norm(axis)  # 単位ベクトルに正規化
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    ux, uy, uz = axis
    


    # クロス積行列
    cross_product_matrix = matrix(
        [
            [mpf('0.0'), -uz, uy],
            [uz, mpf('0.0'), -ux],
            [-uy, ux, mpf('0.0')]
        ]
    )
    # ロドリゲスの回転行列
    # R = np.eye(3) * cos_theta + (1 - cos_theta) * np.outer(axis, axis) + cross_product_matrix * sin_theta
    # 単位行列 I (3x3)
    I = matrix(3, 3)
    for i in range(3):
        I[i, i] = mpf('1.0')

    # 外積軸×軸行列 (outer product)
    outer = matrix(3, 3)
    for i in range(3):
        for j in range(3):
            outer[i, j] = axis[i, 0] * axis[j, 0] if axis.rows == 3 else axis[0, i] * axis[0, j]

    # R = I*cosθ + (1 - cosθ) * outer + cross_product_matrix * sinθ
    R = matrix(3, 3)
    for i in range(3):
        for j in range(3):
            R[i, j] = I[i, j]*cos_theta + (1 - cos_theta) * outer[i, j] + cross_product_matrix[i, j]*sin_theta
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
    coeffs_new = matrix([a1, b1, c1, d1, e1, f1, g1, h1, i1, j1])

    # 元の位置に戻す
    coeffs_new = shift_x(coeffs_new, center[0])
    coeffs_new = shift_y(coeffs_new, center[1])
    coeffs_new = shift_z(coeffs_new, center[2])

    return coeffs_new, rotation_matrix(axis, theta)

# 1. 係数計算の関数
def compute_coeffs(R, coeffs):
    # coeffs: [a, b, c, d, e, f, g, h, i, j] という順番で
    a, b, c, d, e, f, g, h, i, j = coeffs

    # 各係数を計算
    a1 = a * R[0, 0]**2 + b * R[1, 0]**2 + c * R[2, 0]**2 + d * R[0, 0] * R[1, 0] + e * R[2, 0] * R[0, 0] + f * R[1, 0] * R[2, 0]
    b1 = a * R[0, 1]**2 + b * R[1, 1]**2 + c * R[2, 1]**2 + d * R[0, 1] * R[1, 1] + e * R[2, 1] * R[0, 1] + f * R[1, 1] * R[2, 1]
    c1 = a * R[0, 2]**2 + b * R[1, 2]**2 + c * R[2, 2]**2 + d * R[0, 2] * R[1, 2] + e * R[2, 2] * R[0, 2] + f * R[1, 2] * R[2, 2]
    d1 = 2*a*R[0,0]*R[0,1] + 2*b*R[1,0]*R[1,1] + 2*c*R[2,0]*R[2,1] + d*(R[0,1]*R[1,0] + R[0,0]*R[1,1]) + e*(R[2,1]*R[0,0] + R[2,0]*R[0,1]) + f*(R[1,1]*R[2,0] + R[1,0]*R[2,1])
    e1 = 2*a*R[0,0]*R[0,2] + 2*b*R[1,0]*R[1,2] + 2*c*R[2,0]*R[2,2] + d*(R[0,2]*R[1,0] + R[0,0]*R[1,2]) + e*(R[2,2]*R[0,0] + R[2,0]*R[0,2]) + f*(R[1,2]*R[2,0] + R[1,0]*R[2,2])
    f1 = 2*a*R[0,1]*R[0,2] + 2*b*R[1,1]*R[1,2] + 2*c*R[2,1]*R[2,2] + d*(R[0,1]*R[1,2] + R[0,2]*R[1,1]) + e*(R[2,1]*R[0,2] + R[2,2]*R[0,1]) + f*(R[1,1]*R[2,2] + R[1,2]*R[2,1])
    g1 = g * R[0, 0] + h * R[1, 0] + i * R[2, 0]
    h1 = g * R[0, 1] + h * R[1, 1] + i * R[2, 1]
    i1 = g * R[0, 2] + h * R[1, 2] + i * R[2, 2]
    j1 = j

    return matrix([a1, b1, c1, d1, e1, f1, g1, h1, i1, j1])


def mirr_ray_intersection(coeffs, ray, source, negative=False):
    coeffs = [mpf(x) for x in coeffs]
    a, b, c, d, e, f_, g, h, i, j = coeffs

    # mpmath matrix (3×n)
    assert ray.rows == 3 and source.rows == 3
    n_cols = ray.cols

    point = matrix(3, n_cols)

    for col in range(n_cols):
        l, m, n = ray[0, col], ray[1, col], ray[2, col]
        p, q, r = source[0, col], source[1, col], source[2, col]

        A = a*l**2 + b*m**2 + c*n**2 + d*m*l + e*n*l + f_*m*n
        B = (2*a*p*l + 2*b*q*m + 2*c*r*n +
             d*(p*m + q*l) + e*(p*n + r*l) + f_*(r*m + q*n) +
             g*l + h*m + i*n)
        C = (a*p**2 + b*q**2 + c*r**2 + d*p*q + e*p*r + f_*q*r +
             g*p + h*q + i*r + j)

        D = B**2 - 4*A*C
        if D < 0:
            point[0, col] = nan
            point[1, col] = nan
            point[2, col] = nan
            continue

        sqrtD = sqrt(D)
        if negative:
            t = (-B - sqrtD) / (2*A)
        else:
            t = (-B + sqrtD) / (2*A)

        point[0, col] = t*l + p
        point[1, col] = t*m + q
        point[2, col] = t*n + r

    return point


def calcEll_Yvalue(a, b, x):
    return sqrt(b**2. - (b*(x - sqrt(a**2. - b**2.)) /a)**2.)


def calc_Y_hyp(a, b, x):
    y = np.sqrt(-b ** 2 + (b * (x - np.sqrt(a ** 2 + b ** 2)) / a) ** 2)
    return y


def norm_vector(coeffs, point):
    a, b, c, d, e, f, g, h, i, j = coeffs

    if point.rows == 1 and point.cols == 3:
        # 1x3行ベクトル
        x, y, z = point[0, 0], point[0, 1], point[0, 2]
        N = matrix(1, 3)
        N[0, 0] = 2 * a * x + d * y + e * z + g
        N[0, 1] = 2 * b * y + d * x + f * z + h
        N[0, 2] = 2 * c * z + e * x + f * y + i

    elif point.rows == 3:
        # 3×n 行列
        N = matrix(3, point.cols)
        for col in range(point.cols):
            x, y, z = point[0, col], point[1, col], point[2, col]
            N[0, col] = 2 * a * x + d * y + e * z + g
            N[1, col] = 2 * b * y + d * x + f * z + h
            N[2, col] = 2 * c * z + e * x + f * y + i

    else:
        raise ValueError("point の形状は 1x3 または 3xn の matrix である必要があります")

    N = normalize_vector(N)
    return N


# 接線ベクトルを計算する関数
def tang_vector(coeffs, ray, points):
    N = norm_vector(coeffs, points)  # 法線ベクトルを取得
    dot_product = np.einsum('ij,ij->j', ray, N)  # rayと法線の内積
    T = ray - dot_product * N  # 法線方向の成分を引く
    return normalize_vector(T)

def normalize_vector(vector):
    norm = mpmath_norm(vector, axis=0)

    # normがスカラーならリスト化して統一処理（例）
    if not isinstance(norm, (list, tuple)):
        norm = [norm]

    # 全てゼロ判定
    if all(n == 0 for n in norm):
        return vector

    # 要素ごとに割る（ベクトル割り）
    for j, n in enumerate(norm):
        if n == 0:
            continue  # 0割り回避
        for i in range(vector.rows):
            vector[i, j] /= n

    return vector

def mpmath_mean(x, axis=None):
    rows, cols = x.rows, x.cols

    if axis is None:
        # 全要素平均
        total = mpf(0)
        for i in range(rows):
            for j in range(cols):
                total += x[i, j]
        return total / (rows * cols)

    elif axis == 0:
        # 列ごとの平均: 1×cols
        means = matrix(1, cols)
        for j in range(cols):
            s = mpf(0)
            for i in range(rows):
                s += x[i, j]
            means[0, j] = s / rows
        return means

    elif axis == 1:
        # 行ごとの平均: rows×1
        means = matrix(rows, 1)
        for i in range(rows):
            s = mpf(0)
            for j in range(cols):
                s += x[i, j]
            means[i, 0] = s / cols
        return means

    else:
        raise ValueError("axis must be None, 0, or 1")

def reflect_ray(ray, N):
    # 入力: ray と N は 3x1 または 3xn の行列を想定
    cols = ray.cols
    phai = matrix(3, cols)

    for j in range(cols):
        # 各列を取り出す
        l = ray[0, j]
        m = ray[1, j]
        n = ray[2, j]
        nx = N[0, j]
        ny = N[1, j]
        nz = N[2, j]

        # スカラー積
        A = l * nx + m * ny + n * nz

        # phi = ray - 2*A*N
        phai[0, j] = l - 2 * A * nx
        phai[1, j] = m - 2 * A * ny
        phai[2, j] = n - 2 * A * nz

    phai = normalize_vector(phai)
    return phai


def plane_ray_intersection(coeffs, ray, source):
    g, h, i, j = coeffs[6:10]
    ncols = ray.cols  # ベクトルの本数

    # 出力: 3×n の点
    point = matrix(3, ncols)

    for col in range(ncols):
        l = ray[0, col]
        m = ray[1, col]
        n = ray[2, col]

        p = source[0, col]
        q = source[1, col]
        r = source[2, col]

        denom = g * l + h * m + i * n
        num = -(g * p + h * q + i * r + j)

        if denom == 0:
            t = mpf('nan')  # 交点が存在しない場合
        else:
            t = num / denom

        point[0, col] = t * l + p
        point[1, col] = t * m + q
        point[2, col] = t * n + r

    return point

def rotatematrix(rotation_matrix, axis_x, axis_y, axis_z):
    axis_x_new = mpmath_dot(rotation_matrix, axis_x)
    axis_y_new = mpmath_dot(rotation_matrix, axis_y)
    axis_z_new = mpmath_dot(rotation_matrix, axis_z)
    return axis_x_new, axis_y_new, axis_z_new

def rotate_vectors(vector, theta_y, theta_z):
    # 回転行列 (y軸中心)
    R_y = matrix([
        [cos(theta_y), mpf('0.0'), sin(theta_y)],
        [mpf('0.0'), mpf('1.0'), mpf('0.0')],
        [-sin(theta_y), mpf('0.0'), cos(theta_y)]
    ])

    # 回転行列 (z軸中心)
    R_z = matrix([
        [cos(theta_z), -sin(theta_z), mpf('0.0')],
        [sin(theta_z), cos(theta_z), mpf('0.0')],
        [mpf('0.0'), mpf('0.0'), mpf('1.0')]
    ])

    # z軸 → y軸の順に回転させる
    reflect_rotated = R_y @ (R_z @ vector)

    return reflect_rotated

# def rotate_points(points, focus_apprx, theta_y, theta_z):
#     # focus_apprx を原点に移動
#     points_shifted = points - focus_apprx[:, np.newaxis]

#     # y軸とz軸で回転
#     points_rotated = rotate_vectors(points_shifted, theta_y, theta_z)

#     # 元の位置に戻す
#     points_rotated += focus_apprx[:, np.newaxis]

#     return points_rotated
def rotate_points(points, focus_apprx, theta_y, theta_z):
    """
    points: mpmath.matrix (3 x N)
    focus_apprx: mpmath.matrix (3 x 1) or (3,)
    theta_y, theta_z: mpf
    """
    cols = points.cols

    # focus_apprx を列ベクトル (3 x 1) にする
    if isinstance(focus_apprx, matrix) and focus_apprx.cols == 1:
        focus_col = focus_apprx
    else:
        focus_col = matrix(focus_apprx).transpose()  # 3x1 にする

    # focus_apprx を N 列に複製
    focus_mat = matrix(3, cols)
    for j in range(cols):
        for i in range(3):
            focus_mat[i, j] = focus_col[i]

    # 原点に移動
    points_shifted = points - focus_mat

    # y,z回転
    points_rotated = rotate_vectors(points_shifted, theta_y, theta_z)

    # 元の位置に戻す
    points_rotated += focus_mat

    return points_rotated

def plane_correction_with_nan_and_outlier_filter(data, sigma_threshold=3):
    """
    NaNを無視＆3σ外れ値も無視して平面補正を行う

    Parameters
    ----------
    data : ndarray (2D)
        平面補正対象の2D行列データ（NaNを含んでも良い）
    sigma_threshold : float, optional
        何σを外れ値とみなすか（デフォルト=3）

    Returns
    -------
    corrected_data : ndarray (2D)
        平面補正後の2Dデータ（元のNaN位置はNaNのまま）
    """

    # 座標系生成（単純にインデックスを使う）
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

    # 有効データマスク（NaN除外）
    mask = ~np.isnan(data)

    # 一旦データ抽出
    x_fit = x[mask]
    y_fit = y[mask]
    z_fit = data[mask]

    # まず全データで平面フィット（初期推定用）
    def plane(X, a, b, c, d, e):
        x, y = X
        return a * x + b * y + c + d * x **2 + e* y **2
    
    def plane2(X, a, b, c):
        x, y = X
        return a * x + b * y + c
    
    

    initial_params, _ = curve_fit(plane, (x_fit, y_fit), z_fit)

    # 推定平面からの残差
    initial_plane = plane((x_fit, y_fit), *initial_params)
    residual = z_fit - initial_plane

    # 残差の3σを計算して外れ値除去
    sigma = np.std(residual)
    valid_mask = np.abs(residual) < sigma_threshold * sigma

    # 外れ値除去後のデータで再度フィット
    x_fit = x_fit[valid_mask]
    y_fit = y_fit[valid_mask]
    z_fit = z_fit[valid_mask]

    final_params, _ = curve_fit(plane2, (x_fit, y_fit), z_fit)

    # 最終平面補正
    plane_surface = plane2((x, y), *final_params)
    corrected_data = data - plane_surface

    # 元のNaNは維持
    corrected_data[~mask] = np.nan

    return corrected_data

def extract_affine_square_region(img: np.ndarray, target_size: int = None) -> np.ndarray:
    """
    NaNで囲まれた実数値2Dデータから、斜め矩形領域を検出して正方形にアフィン変換で切り出す。

    Parameters:
        img: np.ndarray
            2次元の実数値データ（NaN含む）
        target_size: int or None
            出力する正方形の一辺のサイズ（Noneなら最長辺を使用）

    Returns:
        rectified_img: np.ndarray
            アフィン変換された正方形画像（NaN維持）
    """
    assert img.ndim == 2, "2次元配列を入力してください"

    # --- 1. NaN以外のマスクを作成 ---
    valid_mask = ~np.isnan(img)
    valid_mask_uint8 = valid_mask.astype(np.uint8) * 255

    # --- 2. 輪郭検出 ---
    contours, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("有効領域が見つかりませんでした")

    contour = max(contours, key=cv2.contourArea)

    # --- 3. 輪郭を4点近似 ---
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) != 4:
        raise ValueError(f"矩形が4点で検出できませんでした（点数: {len(approx)}）")

    pts_src = approx[:, 0, :].astype(np.float32)

    # --- 4. 並び替え（左上 → 右上 → 左下）で3点を使う ---
    def order_points_affine(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        top_left = pts[np.argmin(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]
        return np.array([top_left, top_right, bottom_left], dtype=np.float32)

    pts_src_ordered = order_points_affine(pts_src)

    # --- 5. 出力サイズ決定 ---
    if target_size is None:
        width = mpmath_norm(pts_src_ordered[0] - pts_src_ordered[1])
        height = mpmath_norm(pts_src_ordered[0] - pts_src_ordered[2])
        side = int(max(width, height))
    else:
        side = int(target_size)

    # 出力先3点（正方形）：左上→右上→左下
    pts_dst = np.array([
        [0, 0],
        [side - 1, 0],
        [0, side - 1]
    ], dtype=np.float32)

    # --- 6. アフィン変換行列作成 ---
    M = cv2.getAffineTransform(pts_src_ordered, pts_dst)

    # --- 7. NaN→0で warp ---
    img_filled = np.nan_to_num(img, nan=0.0)
    warped = cv2.warpAffine(img_filled, M, (side, side), flags=cv2.INTER_LINEAR)

    # --- 8. NaNマスクも変換して復元 ---
    mask_warped = cv2.warpAffine(valid_mask_uint8, M, (side, side), flags=cv2.INTER_NEAREST)
    warped[mask_warped == 0] = np.nan

    return warped
def mpmath_cross(u, v):
    """3次元ベクトルのクロス積"""
    return matrix([
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    ])
def mpmath_inner(u, v):
    """内積"""
    return sum(u[k]*v[k] for k in range(3))
def angle_between_2vector(ray1,ray2):
    # 列ごとの結果を保存するリスト
    angles_between_rad = []  # ray1とray2のなす角度
    angles_yx_rad = []       # y/x の角度
    angles_zx_rad = []       # z/x の角度

    # 各列に対して処理を行う
    for i in range(ray1.cols):
        # 各列ごとにベクトルを取得
        v1 = matrix([mpf(ray1[0, i]), mpf(ray1[1, i]), mpf(ray1[2, i])])
        v2 = matrix([mpf(ray2[0, i]), mpf(ray2[1, i]), mpf(ray2[2, i])])

        # ray1 と ray2 のなす角度を計算
        dot_product = mpmath_inner(v1, v2)
        norm_v1 = mpmath_norm(v1)
        norm_v2 = mpmath_norm(v2)
        cross_product = mpmath_cross(v1, v2)
        norm_cross = mpmath_norm(cross_product)
        angle_between_rad = mp.atan2(norm_cross, dot_product)
        angles_between_rad.append(angle_between_rad)

        # y/x, z/x の角度
        x_val = ray1[0, i]
        if x_val != 0:
            angle_yx_rad = mp.atan2(ray1[1, i], x_val)
            angle_zx_rad = mp.atan2(ray1[2, i], x_val)
        else:
            angle_yx_rad = nan
            angle_zx_rad = nan

        angles_yx_rad.append(angle_yx_rad)
        angles_zx_rad.append(angle_zx_rad)
    return angles_between_rad, angles_yx_rad, angles_zx_rad

def mpmath_norm(mat, axis=None):
    """
    mpmath.matrix に対してノルムを計算する。
    
    - mat: mpmath.matrix, shape = (3, n) または (3, 1)
    - axis=None … 全体のノルム（3×1の場合のみ）
    - axis=0 … 列ごとのノルム
    """
    if axis is None:
        # 3x1の場合: 全体のノルム
        return sqrt(mat[0]**2 + mat[1]**2 + mat[2]**2)
    elif axis == 0:
        # 列ごとのノルム
        n_cols = mat.cols
        result = []
        for j in range(n_cols):
            norm_j = sqrt(mat[0,j]**2 + mat[1,j]**2 + mat[2,j]**2)
            result.append(norm_j)
        return result
    else:
        raise ValueError(f"axis={axis} は未サポートです")
def mpmath_zeros(*shape):
    """
    mpmath.matrix でゼロ行列を作る。
    1次元の場合は 1×n の行ベクトルを返す。

    例：
    mpmath_zeros(10)      → 1×10
    mpmath_zeros(3, 5)    → 3×5
    mpmath_zeros((3,5))   → 3×5  # 追加対応
    """
    # shapeが1個でかつその中身がタプルなら展開
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]

    if len(shape) == 1:
        rows = 1
        cols = shape[0]
    elif len(shape) == 2:
        rows, cols = shape
    else:
        raise ValueError("1次元または2次元のみサポート")

    m = matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            m[i, j] = mpf(0)
    return m


def mpmath_dot(A, B):
    # 次元チェック
    if A.cols != B.rows:
        raise ValueError(f"行列積の次元不一致: A.cols={A.cols} != B.rows={B.rows}")

    result = matrix(A.rows, B.cols)

    for i in range(A.rows):
        for j in range(B.cols):
            s = 0
            for k in range(A.cols):
                s += A[i, k] * B[k, j]
            result[i, j] = s

    return result

def mpmath_linspace(start, stop, num=50, endpoint=True):
    """
    mpmath 互換 linspace
    np.linspace(0, 1, 5) と同じ呼び方が可能
    戻り値は 1×num の行ベクトル
    """
    start = mpf(start)
    stop = mpf(stop)

    if num < 2:
        return matrix([[start]])

    if endpoint:
        step = (stop - start) / (num - 1)
    else:
        step = (stop - start) / num

    result = matrix(1, num)
    for i in range(num):
        result[0, i] = start + i * step

    return result
def elementwise_tan(vec):
    """
    mpmath 行/列ベクトルに対して要素ごとに tan を適用。
    """
    rows, cols = vec.rows, vec.cols
    result = matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i, j] = tan(vec[i, j])
    return result

def elementwise_atan(vec):
    """
    mpmath 行/列ベクトルに対して要素ごとに tan を適用。
    """
    rows, cols = vec.rows, vec.cols
    result = matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i, j] = atan(vec[i, j])
    return result

def elementwise_div(numer, denom):
    """
    mpmath.matrix 同士を要素ごとに割る。

    Parameters
    ----------
    numer : mpmath.matrix
        分子（同じ形の行列）
    denom : mpmath.matrix
        分母（同じ形の行列）

    Returns
    -------
    result : mpmath.matrix
        要素ごとの除算結果
    """
    if numer.rows != denom.rows or numer.cols != denom.cols:
        raise ValueError("numer と denom は同じ形である必要があります")

    result = matrix(numer.rows, numer.cols)

    for i in range(numer.rows):
        for j in range(numer.cols):
            result[i, j] = numer[i, j] / denom[i, j]

    return result

def select_columns_matrix(m, col_indices):
    """
    mpmath.matrix の特定の列だけ抜き出した行列を作る。
    """
    result = matrix(m.rows, len(col_indices))
    for idx, col in enumerate(col_indices):
        if col >= m.cols:
            raise IndexError(f"列番号 {col} は範囲外")
        for row in range(m.rows):
            result[row, idx] = m[row, col]
    return result

def mpmath_interp1d_wrapper_to_matrix(x_mpmath, y_mpmath, x_new_mpmath, kind='linear'):
    """
    mpmath.matrix型の x, y と x_new を受け取り、
    scipy.interpolate.interp1d で補間し、
    補間結果を mpmath.matrix で返す関数。

    引数:
    - x_mpmath: 1×n の mpmath.matrix
    - y_mpmath: 1×n の mpmath.matrix
    - x_new_mpmath: 1×m の mpmath.matrix
    - kind: 補間の種類（デフォルト 'linear'）

    戻り値:
    - 補間結果の 1×m の mpmath.matrix
    """
    # print('x_mpmath:', nstr(x_mpmath,5))
    # print('y_mpmath:', nstr(y_mpmath,5))
    # print('x_new_mpmath:', nstr(x_new_mpmath,5))
    # mpmath.matrix → numpy 1D array に変換
    x_np = np.array([float(x_mpmath[0, i]) for i in range(x_mpmath.cols)])
    y_np = np.array([float(y_mpmath[0, i]) for i in range(y_mpmath.cols)])
    x_new_np = np.array([float(x_new_mpmath[0, i]) for i in range(x_new_mpmath.cols)])

    # 補間関数作成
    interp_func = interp1d(x_np, y_np, kind=kind)
    y_new_np = interp_func(x_new_np)

    # numpy配列 → mpmath.matrixに戻す (1×m 行列)
    y_new_mp = matrix(1, len(y_new_np))
    for i, val in enumerate(y_new_np):
        y_new_mp[0, i] = mpf(val)

    return y_new_mp

                # output_equal_v = np.linspace(angle_v_sep_y[0],angle_v_sep_y[-1],len(angle_v_sep_y))
                # output_equal_h = np.linspace(angle_h_sep_y[0],angle_h_sep_y[-1],len(angle_h_sep_y))

                # interp_func_v = interp1d(angle_v_sep_y, rand_p0v, kind='linear')
                # interp_func_h = interp1d(angle_h_sep_y, rand_p0h, kind='linear')

                # rand_p0v_new = interp_func_v(output_equal_v)
                # rand_p0h_new = interp_func_h(output_equal_h)

def mpmath_min(m):
    # mpmath.matrix の全要素から最小値を返す
    vals = [m[i, j] for i in range(m.rows) for j in range(m.cols)]
    return min(vals)

def mpmath_max(m):
    # mpmath.matrix の全要素から最大値を返す
    vals = [m[i, j] for i in range(m.rows) for j in range(m.cols)]
    return max(vals)

import numpy as np
from mpmath import matrix

def mpmath_matrix_to_numpy(mat):
    """
    mpmath.matrix を np.ndarray (dtype=np.float64) に変換する

    Parameters
    ----------
    mat : mpmath.matrix
        変換したい mpmath の行列

    Returns
    -------
    np.ndarray
        numpy の float64 配列
    """
    arr = np.zeros((mat.rows, mat.cols), dtype=np.float64)
    for i in range(mat.rows):
        for j in range(mat.cols):
            arr[i, j] = float(mat[i, j])
    return arr


if option_wolter_3_1:
    def plot_result_debug(params,option,source_shift=[0.,0.,0.],option_tilt = True,option_legendre=False):
        defocus, astigH, \
        pitch_hyp_v, roll_hyp_v, yaw_hyp_v, decenterX_hyp_v, decenterY_hyp_v, decenterZ_hyp_v,\
        pitch_hyp_h, roll_hyp_h, yaw_hyp_h, decenterX_hyp_h, decenterY_hyp_h, decenterZ_hyp_h,\
        pitch_ell_v, roll_ell_v, yaw_ell_v, decenterX_ell_v, decenterY_ell_v, decenterZ_ell_v,\
        pitch_ell_h, roll_ell_h, yaw_ell_h, decenterX_ell_h, decenterY_ell_h, decenterZ_ell_h  = params ### 3型 1st 1型 4th 3型 2nd 1型 3rd

        if True:  # 初期設定 2025/06/12
            ### 3型 Setting12
            a_hyp_v  =   mpf(72.9825)
            b_hyp_v  =   mpf(0.263879113520857)
            a_ell_v  =   mpf(0.1175)
            b_ell_v  =   mpf(0.0283168369674688)
            hyp_length_v  =   mpf(0.043)
            ell_length_v  =   mpf(0.0809220387326922)
            theta1_v  =   mpf(5.55983241203018E-05)
            theta2_v  =   mpf(0.117)
            theta3_v  =   mpf(0.23405559832412)
            theta4_v  =   mpf(0.243572583671924)
            theta5_v  =   mpf(-0.253089569019727)
            phai_hyp_v  =   mpf(-0.11705559832412)
            phai_ell_v  =   mpf(0.0172109421954116)
            F_eff_v  =   mpf(0.0314202608037988)
            Mag_v  =   mpf(4651.85850857562)
            NA_v  =   mpf(0.0819227398493535)
            Aperture_v  =   mpf(0.0195160723325698)
            F0_F2_v  =   mpf(146.19402782262)




            ### 1型 setting11
            a_ell_h  =   mpf(73.1076714403445)
            b_ell_h  =   mpf(0.517019631143022)
            a_hyp_h  =   mpf(0.0077)
            b_hyp_h  =   mpf(0.00432051448679384)
            hyp_length_h  =   mpf(0.01380360633)
            ell_length_h  =   mpf(0.030)
            theta1_h  =   mpf(0.000145746388538841)
            theta2_h  =   mpf(0.17)
            theta3_h  =   mpf(0.339854253611461)
            theta4_h  =   mpf(0.182330449161024)
            theta5_h  =   mpf(0.757889356272919)
            phai_ell_h  =   mpf(-0.169854253611461)
            phai_hyp_h  =   mpf(-0.540136320213965)
            F_eff_h  =   mpf(0.0225095323759206)
            Mag_h  =   mpf(6493.76076981313)
            NA_h  =   mpf(0.0819088461351015)
            Aperture_h  =   mpf(0.00507547047200988)
            F0_F2_h  =   mpf(146.194027821967)




            length_hyp_v = hyp_length_v
            length_ell_v = ell_length_v
            length_hyp_h = hyp_length_h
            length_ell_h = ell_length_h

        omega_v = theta1_v - theta5_v
        if option == 'ray':
            omega_v = theta1_v - theta5_v
            print('omega_v',omega_v)
            omega_h = theta1_h + theta5_h
            print('omega_h',omega_h)
        omega_v = 0.

        org_hyp_v = np.sqrt(a_hyp_v**2 + b_hyp_v**2)
        org_hyp_h = np.sqrt(a_hyp_h**2 + b_hyp_h**2)

        org_ell_v = np.sqrt(a_ell_v**2 - b_ell_v**2)
        org_ell_h = np.sqrt(a_ell_h**2 - b_ell_h**2)

        n = 20
        # param_ = mpmath_linspace(-1,1,n)
        std_v = np.full(n, np.nan)
        std_h = np.full(n, np.nan)
        # Input parameters
        ray_num = 25

        ray_num_H = 25
        ray_num_V = 25
        ray_num = 25
        if option == 'ray':
            ray_num_H = 25
            ray_num_V = 25
            # ray_num = 1
            print('omega_v',omega_v)
        if option == 'wave' or option == 'ray_wave':
            ray_num_H = wave_num_H
            ray_num_V = wave_num_V
            ray_num = wave_num_H
        bool_draw = True
        bool_point_source = True
        bool_imaging = False
        option_axial = True
        option_alignment = True
        optin_axialrotation = True
        option_rotateLocal = True
        # Set mirror parameters directly
        c_v = mpmath_zeros(10)
        c_v[0] = 1 / a_hyp_v**2
        c_v[2] = -1 / b_hyp_v**2
        c_v[9] = -1.
        
        org_v = np.sqrt(a_hyp_v**2 + b_hyp_v**2)
        c_v = shift_x(c_v, org_v)

        center_v = mirr_ray_intersection(c_v, matrix([[cos(theta1_v)], [mpf('0.0')], [sin(theta1_v)]]), matrix([[mpf('0.0')], [mpf('0.0')], [mpf('0.0')]]))
        if not np.isreal(center_v).all():
            return np.inf
        x1_v = center_v[0, 0] - length_hyp_v / 2
        x2_v = center_v[0, 0] + length_hyp_v / 2

        y1_v = calc_Y_hyp(a_hyp_v, b_hyp_v, x1_v)
        y2_v = calc_Y_hyp(a_hyp_v, b_hyp_v, x2_v)

        accept_v = fabs(y2_v - y1_v)
        l1v = mpmath_norm(c_v)

        ### 1 ell
        c_h = mpmath_zeros(10)
        c_h[0] = 1 / a_ell_h**2
        c_h[1] = 1 / b_ell_h**2
        c_h[9] = -1.
        org_h = np.sqrt(a_ell_h**2 - b_ell_h**2)
        c_h = shift_x(c_h, org_h)
        center_h = mirr_ray_intersection(c_h, matrix([[cos(theta1_h)], [sin(theta1_h)], [mpf('0.0')]]), matrix([[mpf('0.0')], [mpf('0.0')], [mpf('0.0')]]))
        if not np.isreal(center_h).all():
            return np.inf
        x1_h = center_h[0, 0] - length_ell_h / 2
        x2_h = center_h[0, 0] + length_ell_h / 2
        y1_h = calcEll_Yvalue(a_ell_h, b_ell_h, x1_h)
        y2_h = calcEll_Yvalue(a_ell_h, b_ell_h, x2_h)
        # print(atan(y1_h/x1_h))
        # print(atan(y2_h/x2_h))
        accept_h = fabs(y2_h - y1_h)
        l1h = mpmath_norm(c_h)

        if option == 'ray':
            print('x1_v',x1_v)
            print('y1_v',y1_v)
            print('x2_v',x2_v)
            print('y2_v',y2_v)
            print('x1_h',x1_h)
            print('y1_h',y1_h)
            print('x2_h',x2_h)
            print('y2_h',y2_h)

        # Raytrace (X = x-ray direction)

        # V hyp mirror set (1st)
        axis_x = matrix([mpf('1.0'), mpf('0.0'), mpf('0.0')])
        axis_y = matrix([mpf('0.0'), mpf('1.0'), mpf('0.0')])
        axis_z = matrix([mpf('0.0'), mpf('0.0'), mpf('1.0')])
        coeffs_hyp_v = mpmath_zeros(10)
        coeffs_hyp_v[0] = 1 / a_hyp_v**2
        coeffs_hyp_v[2] = -1 / b_hyp_v**2
        coeffs_hyp_v[9] = -1.
        coeffs_hyp_v = shift_x(coeffs_hyp_v, org_hyp_v)
        if option_axial:
            # coeffs_hyp_v = rotate_y(coeffs_hyp_v, theta1_v, [0, 0, 0])
            coeffs_hyp_v, rotation_matrix = rotate_general_axis(coeffs_hyp_v, axis_y, theta1_v, [0, 0, 0])
            axis_x, axis_y, axis_z = rotatematrix(rotation_matrix, axis_x, axis_y, axis_z)

        if option_alignment and option_axial:
            bufray = mpmath_zeros((3, 3))
            ### 4隅の光線
            # theta_cntr_h = (atan(y2_h / x2_h) + atan(y1_h / x1_h))/2.
            theta_cntr_v = (atan(y2_v / x2_v) + atan(y1_v / x1_v))/2.
            def print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h):
                l2_h = (4*a_hyp_h**2 + (org_hyp_h*2)**2 -4*a_hyp_h*(org_hyp_h*2)*cos(theta1_h))/(4*org_hyp_h - 4*a_hyp_h)
                # l2_v = (4*a_hyp_v**2 + (org_hyp_v*2)**2 -4*a_hyp_v*(org_hyp_v*2)*cos(theta1_v))/(4*org_hyp_v - 4*a_hyp_v)

                l1_h = 2*a_hyp_h +l2_h
                # l1_v = 2*a_hyp_v +l2_v

                theta2_h = asin(org_hyp_h*2*sin(theta1_h)/l2_h)/2
                # theta2_v = asin(org_hyp_v*2*sin(theta1_v)/l2_v)/2

                theta3_h = asin(l1_h*sin(theta1_h)/l2_h)
                # theta3_v = asin(l1_v*sin(theta1_v)/l2_v)

                l4_h = ((org_ell_h)**2 - 2*org_ell_h*a_ell_h*cos(theta3_h) + a_ell_h**2)/(a_ell_h - org_ell_h*cos(theta3_h))
                # l4_v = ((org_ell_v)**2 - 2*org_ell_v*a_ell_v*cos(theta3_v) + a_ell_v**2)/(a_ell_v - org_ell_v*cos(theta3_v))

                l3_h = 2*a_ell_h - l2_h - l4_h
                # l3_v = 2*a_ell_v - l2_v - l4_v

                theta5_h = asin((2*a_ell_h - l4_h)*sin(theta3_h)/l4_h)
                # theta5_v = asin((2*a_ell_v - l4_v)*sin(theta3_v)/l4_v)

                theta4_h = (theta5_h+theta3_h)/2.
                # theta4_v = (theta5_v+theta3_v)/2.
                # print('theta4_v incidence ell',theta4_v)
                theta4_h = asin(2*org_ell_h*sin(theta3_h)/l4_h)/2
                # theta4_v = asin(2*org_ell_v*sin(theta3_v)/l4_v)/2
                return theta2_h, theta3_h, theta4_h, theta5_h, l1_h, l2_h, l3_h, l4_h
            theta2_v, theta3_v, theta4_v, theta5_v, l1_v, l2_v, l3_v, l4_v = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,theta1_v)
            theta2_v1, theta3_v1, theta4_v1, theta5_v1, l1_v1, l2_v1, l3_v1, l4_v1 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,atan(y1_v / x1_v))
            theta2_v2, theta3_v2, theta4_v2, theta5_v2, l1_v2, l2_v2, l3_v2, l4_v2 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,atan(y2_v / x2_v))
            omega_v = (theta5_v1 + theta5_v2 + atan(y1_v / x1_v) + atan(y2_v / x2_v))/2

            if option == 'ray':
                print('===== ===== =====')
                print('l1_v',l1_v)
                print('l2_v',l2_v)
                print('l3_v',l3_v)
                print('l4_v',l4_v)
                print('l1_v1',l1_v1)
                print('l2_v1',l2_v1)
                print('l3_v1',l3_v1)
                print('l4_v1',l4_v1)
                print('l1_v2',l1_v2)
                print('l2_v2',l2_v2)
                print('l3_v2',l3_v2)
                print('l4_v2',l4_v2)


                print('theta1_v',theta1_v)
                print('theta1_v1',atan(y1_v / x1_v))
                print('theta1_v2',atan(y2_v / x2_v))
                print('theta2_v',theta2_v)
                print('theta2_v1',theta2_v1)
                print('theta2_v2',theta2_v2)
                print('theta3_v',theta3_v)
                print('theta3_v1',theta3_v1)
                print('theta3_v2',theta3_v2)
                print('theta4_v',theta4_v)
                print('theta4_v1',theta4_v1)
                print('theta4_v2',theta4_v2)
                print('theta5_v',theta5_v)
                print('theta5_v1',theta5_v1)
                print('theta5_v2',theta5_v2)
                print('omega_v',omega_v)
                # print('na_h',sin((theta5_h1 - theta5_h2))/2.)
                print('na_v',sin((theta5_v1 - theta5_v2))/2.)
                # print('div_h',(theta5_h1 - theta5_h2))
                print('div_v',(theta5_v1 - theta5_v2))
                print('div0_h',atan(y1_h / x1_h) - atan(y2_h / x2_h))
                print('div0_v',atan(y1_v / x1_v) - atan(y2_v / x2_v))
                print('===== ===== =====')

            theta_source_1_v = atan(y1_v / x1_v) - theta_cntr_v
            theta_source_2_v = atan(y2_v / x2_v) - theta_cntr_v

            bufray[0, 0] = 1.

            bufray[0, 1] = 1.
            bufray[2, 1] = tan(theta_source_1_v)

            bufray[0, 2] = 1.
            bufray[2, 2] = tan(theta_source_2_v)
            source = mpmath_zeros((3, 3))

        bufray = normalize_vector(bufray)

        center_hyp_v = mirr_ray_intersection(coeffs_hyp_v, bufray, source)
        if not np.isreal(center_hyp_v).all():
            return np.inf
        bufreflect1 = reflect_ray(bufray, norm_vector(coeffs_hyp_v, center_hyp_v))

        bufreflangle1_y = atan(mpmath_mean(bufreflect1[2, 1:]) / mpmath_mean(bufreflect1[0, 1:]))

        if option == 'ray':
            print('coeffs_hyp_v',coeffs_hyp_v)
            print('center_hyp_v',center_hyp_v)
            print('bufray',bufray)
            print('mpmath_mean(bufreflect1[2, 1:])',mpmath_mean(bufreflect1[2, 1:]))
            print('mpmath_mean(bufreflect1[0, 1:])',mpmath_mean(bufreflect1[0, 1:]))
            print('angle_y 1st to 2nd',bufreflangle1_y)


        # V ell mirror set (2nd)
        axis2_x = matrix([mpf('1.0'), mpf('0.0'), mpf('0.0')])
        axis2_y = matrix([mpf('0.0'), mpf('1.0'), mpf('0.0')])
        axis2_z = matrix([mpf('0.0'), mpf('0.0'), mpf('1.0')])
        coeffs_ell_v = mpmath_zeros(10)
        coeffs_ell_v[0] = 1 / a_ell_v**2
        coeffs_ell_v[2] = 1 / b_ell_v**2
        coeffs_ell_v[9] = -1.
        org_ell_v1 = org_ell_v
        if option == 'ray':
            print('coeffs_ell_v',coeffs_ell_v)
        coeffs_ell_v = shift_x(coeffs_ell_v, 2 * org_hyp_v + org_ell_v)
        if option_axial:
            coeffs_ell_v, rotation_matrix = rotate_general_axis(coeffs_ell_v, axis2_y, theta1_v, [0, 0, 0])
            # coeffs_ell_v, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis2_z, -theta1_h, [0, 0, 0])
            axis2_x, axis2_y, axis2_z = rotatematrix(rotation_matrix, axis2_x, axis2_y, axis2_z)
        print('bufreflect1',bufreflect1)
        mean_bufreflect1 = mpmath_mean(bufreflect1[:, 1:],axis = 1)

        center_ell_v = mirr_ray_intersection(coeffs_ell_v, bufreflect1, center_hyp_v)
        if not np.isreal(center_ell_v).all():
            return np.inf


        bufreflect2 = reflect_ray(bufreflect1, norm_vector(coeffs_ell_v, center_ell_v))
        bufreflangle2_z = atan(mpmath_mean(bufreflect2[1, 1:]) / mpmath_mean(bufreflect2[0, 1:]))
        bufreflangle2_y = atan(mpmath_mean(bufreflect2[2, 1:]) / mpmath_mean(bufreflect2[0, 1:]))

        if option == 'ray':
            print('angle_y 2nd to 3rd',bufreflangle2_y)
            print('angle_z 2nd to 3rd',bufreflangle2_z)


        # Set H ellipse mirror in the vert set (3rd)
        axis3_x = matrix([mpf('1.0'), mpf('0.0'), mpf('0.0')])
        axis3_y = matrix([mpf('0.0'), mpf('1.0'), mpf('0.0')])
        axis3_z = matrix([mpf('0.0'), mpf('0.0'), mpf('1.0')])
        coeffs_ell_h = mpmath_zeros(10)
        coeffs_ell_h[0] = 1 / a_ell_h**2
        coeffs_ell_h[1] = 1 / b_ell_h**2
        coeffs_ell_h[9] = -1.

        coeffs_ell_h = shift_x(coeffs_ell_h, org_ell_h + astigH)

        if option_axial:
            coeffs_ell_h, rotation_matrix = rotate_general_axis(coeffs_ell_h, axis3_z, -theta1_h, [0, 0, 0])
            axis3_x, axis3_y, axis3_z = rotatematrix(rotation_matrix, axis3_x, axis3_y, axis3_z)

        center_ell_h = mirr_ray_intersection(coeffs_ell_h, bufreflect2, center_ell_v)
        if not np.isreal(center_ell_h).all():
            return np.inf
        if option_alignment:
            if not optin_axialrotation:
                coeffs_ell_h = rotate_y(coeffs_ell_h, -bufreflangle2_y, mpmath_mean(center_ell_v[:, 1:],axis=1))
            if optin_axialrotation:
                if option_rotateLocal:
                    ### 正確に言うと　omega_h
                    coeffs_ell_h, rotation_matrix = rotate_general_axis(coeffs_ell_h, axis3_y, omega_v, mpmath_mean(center_ell_v[:, 1:],axis=1))
                    axis3_x, axis3_y, axis3_z = rotatematrix(rotation_matrix, axis3_x, axis3_y, axis3_z)
                else:
                    coeffs_ell_h = rotate_y(coeffs_ell_h, -bufreflangle2_y, center_ell_h[:, 0])
                center_ell_h = mirr_ray_intersection(coeffs_ell_h, bufreflect2, center_ell_v)
        bufreflect3 = reflect_ray(bufreflect2, norm_vector(coeffs_ell_h, center_ell_h))
        bufreflangle3_y = atan(mpmath_mean(bufreflect3[2, 1:]) / mpmath_mean(bufreflect3[0, 1:]))
        bufreflangle3_z = atan(mpmath_mean(bufreflect3[1, 1:]) / mpmath_mean(bufreflect3[0, 1:]))

        if option == 'ray':
            print('angle_y 3rd to 4th',bufreflangle3_y)
            print('angle_z 3rd to 4th',bufreflangle3_z)

        # Set ellipse mirror in the horiz set (4th)
        axis4_x = matrix([mpf('1.0'), mpf('0.0'), mpf('0.0')])
        axis4_y = matrix([mpf('0.0'), mpf('1.0'), mpf('0.0')])
        axis4_z = matrix([mpf('0.0'), mpf('0.0'), mpf('1.0')])
        coeffs_hyp_h = mpmath_zeros(10)
        coeffs_hyp_h[0] = 1 / a_hyp_h**2
        coeffs_hyp_h[1] = -1 / b_hyp_h**2
        coeffs_hyp_h[9] = -1.

        coeffs_hyp_h = shift_x(coeffs_hyp_h, -org_hyp_h + 2 * org_ell_h + astigH)
        if option_axial:
            # coeffs_ell_h = rotate_y(coeffs_ell_h, bufreflangle1_y, center_hyp_v[:, 0])
            coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis4_z, -theta1_h, [0, 0, 0])
            axis4_x, axis4_y, axis4_z = rotatematrix(rotation_matrix, axis4_x, axis4_y, axis4_z)

        center_hyp_h = mirr_ray_intersection(coeffs_hyp_h, bufreflect3, center_ell_h,negative=True)
        if not np.isreal(center_hyp_h).all():
            return np.inf
        if option_alignment:
            if not optin_axialrotation:
                coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle3_y, mpmath_mean(center_hyp_v[:, 1:],axis=1))
            if optin_axialrotation:
                if option_rotateLocal:
                    coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis4_y, omega_v, mpmath_mean(center_ell_v[:, 1:],axis=1))
                    axis4_x, axis4_y, axis4_z = rotatematrix(rotation_matrix, axis4_x, axis4_y, axis4_z)
                else:
                    coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle3_y, mpmath_mean(center_hyp_h[:, 1:],axis=1))
                center_hyp_h = mirr_ray_intersection(coeffs_hyp_h, bufreflect3, center_ell_h,negative=True)
        bufreflect4 = reflect_ray(bufreflect3, norm_vector(coeffs_hyp_h, center_hyp_h))
        bufreflangle4_z = atan(mpmath_mean(bufreflect4[1, 1:]) / mpmath_mean(bufreflect4[0, 1:]))
        bufreflangle4_y = atan(mpmath_mean(bufreflect4[2, 1:]) / mpmath_mean(bufreflect4[0, 1:]))

        s2f_H = -2 * org_hyp_h + 2 * org_ell_h
        s2f_V = 2 * org_hyp_v + 2 * org_ell_v
        # print(s2f_H)
        # print(s2f_V)
        s2f_middle = (s2f_H + s2f_V) / 2
        coeffs_det = mpmath_zeros(10)
        coeffs_det[6] = 1.
        coeffs_det[9] = -(s2f_middle + defocus)

        if option == 'ray':
            print('center_hyp_v',center_hyp_v)
            print('center_ell_v',center_ell_v)
            print('center_ell_h',center_ell_h)
            print('center_hyp_h',center_hyp_h)
            print('s2f_H',s2f_H)
            print('s2f_V',s2f_V)
            print('s2f_middle + defocus',s2f_middle + defocus)
        if center_ell_v[0,0]<center_hyp_v[0,0]:
            print('conflict 1st 2nd')
            return np.inf
        if center_ell_h[0,0]<center_ell_v[0,0]:
            print('conflict 2nd 3rd')
            return np.inf
        if center_hyp_h[0,0]<center_ell_h[0,0]:
            print('conflict 3rd 4th')
            return np.inf

        detcenter = plane_ray_intersection(coeffs_det, bufreflect4, center_hyp_h)
        detcenter = detcenter[:, 0]
        if option_rotateLocal:
            if True:
                center_wolter_h = (mpmath_mean(center_ell_h[:, 1:],axis=1) + mpmath_mean(center_hyp_h[:, 1:],axis=1)) / 2
                if pitch_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis3_y, pitch_ell_h, center_wolter_h)
                if yaw_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis3_z, yaw_ell_h, center_wolter_h)
                if roll_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis3_x, roll_ell_h, center_wolter_h)
                if pitch_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis4_y, pitch_hyp_h, center_wolter_h)
                if yaw_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis4_z, yaw_hyp_h, center_wolter_h)
                if roll_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis4_x, roll_hyp_h, center_wolter_h)
                
                if False: #option_rotateLocal:
                    center_wolter_v = (mpmath_mean(center_ell_v[:, 1:],axis=1) + mpmath_mean(center_hyp_v[:, 1:],axis=1)) / 2
                    if yaw_ell_v != 0:
                        coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_z, yaw_ell_v, center_wolter_v)
                    if pitch_ell_v != 0:
                        coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_y, pitch_ell_v, center_wolter_v)
                    if roll_ell_v != 0:
                        coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_x, roll_ell_v, center_wolter_v)
                    if yaw_hyp_v != 0:
                        coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_z, yaw_hyp_v, center_wolter_v)
                    if pitch_hyp_v != 0:
                        coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_y, pitch_hyp_v, center_wolter_v)
                    if roll_hyp_v != 0:
                        coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_x, roll_hyp_v, center_wolter_v)
                else:
                    if yaw_ell_v != 0:
                        coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_z, yaw_ell_v, mpmath_mean(center_ell_v[:, 1:],axis=1))
                    if pitch_ell_v != 0:
                        coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_y, pitch_ell_v, mpmath_mean(center_ell_v[:, 1:],axis=1))
                    if roll_ell_v != 0:
                        coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_x, roll_ell_v, mpmath_mean(center_ell_v[:, 1:],axis=1))
                    
                    if yaw_hyp_v != 0:
                        coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_z, yaw_hyp_v, mpmath_mean(center_hyp_v[:, 1:],axis=1))
                    if pitch_hyp_v != 0:
                        coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_y, pitch_hyp_v, mpmath_mean(center_hyp_v[:, 1:],axis=1))
                    if roll_hyp_v != 0:
                        coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_x, roll_hyp_v, mpmath_mean(center_hyp_v[:, 1:],axis=1))

            
            else:
                if yaw_ell_v != 0:
                    coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_z, yaw_ell_v, mpmath_mean(center_ell_v[:, 1:],axis=1))
                if pitch_ell_v != 0:
                    coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_y, pitch_ell_v, mpmath_mean(center_ell_v[:, 1:],axis=1))
                if roll_ell_v != 0:
                    coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_x, roll_ell_v, mpmath_mean(center_ell_v[:, 1:],axis=1))
                
                if yaw_hyp_v != 0:
                    coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_z, yaw_hyp_v, mpmath_mean(center_hyp_v[:, 1:],axis=1))
                if pitch_hyp_v != 0:
                    coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_y, pitch_hyp_v, mpmath_mean(center_hyp_v[:, 1:],axis=1))
                if roll_hyp_v != 0:
                    coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_x, roll_hyp_v, mpmath_mean(center_hyp_v[:, 1:],axis=1))

                if pitch_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis4_y, pitch_ell_h, mpmath_mean(center_ell_h[:, 1:],axis=1))
                if yaw_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis4_z, yaw_ell_h, mpmath_mean(center_ell_h[:, 1:],axis=1))
                if roll_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis4_x, roll_ell_h, mpmath_mean(center_ell_h[:, 1:],axis=1))
                if pitch_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis3_y, pitch_hyp_h, mpmath_mean(center_hyp_h[:, 1:],axis=1))
                if yaw_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis3_z, yaw_hyp_h, mpmath_mean(center_hyp_h[:, 1:],axis=1))
                if roll_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis3_x, roll_hyp_h, mpmath_mean(center_hyp_h[:, 1:],axis=1))
        
            if option == 'ray':
                print('axis_x',axis_x)
                print('axis_y',axis_y)
                print('axis_z',axis_z)
                print('axis2_x',axis2_x)
                print('axis2_y',axis2_y)
                print('axis2_z',axis2_z)
                print('axis3_x',axis3_x)
                print('axis3_y',axis3_y)
                print('axis3_z',axis3_z)
                print('axis4_x',axis4_x)
                print('axis4_y',axis4_y)
                print('axis4_z',axis4_z)
            
            

            if decenterX_hyp_v != 0:
                coeffs_hyp_v = shift_x(coeffs_hyp_v, decenterX_hyp_v*axis_x[0])
                coeffs_hyp_v = shift_y(coeffs_hyp_v, decenterX_hyp_v*axis_x[1])
                coeffs_hyp_v = shift_z(coeffs_hyp_v, decenterX_hyp_v*axis_x[2])

            if decenterY_hyp_v != 0:
                coeffs_hyp_v = shift_x(coeffs_hyp_v, decenterY_hyp_v*axis_y[0])
                coeffs_hyp_v = shift_y(coeffs_hyp_v, decenterY_hyp_v*axis_y[1])
                coeffs_hyp_v = shift_z(coeffs_hyp_v, decenterY_hyp_v*axis_y[2])
            if decenterZ_hyp_v != 0:
                coeffs_hyp_v = shift_x(coeffs_hyp_v, decenterZ_hyp_v*axis_z[0])
                coeffs_hyp_v = shift_y(coeffs_hyp_v, decenterZ_hyp_v*axis_z[1])
                coeffs_hyp_v = shift_z(coeffs_hyp_v, decenterZ_hyp_v*axis_z[2])

            if decenterX_hyp_h != 0:
                coeffs_hyp_h = shift_x(coeffs_hyp_h, decenterX_hyp_h*axis4_x[0])
                coeffs_hyp_h = shift_y(coeffs_hyp_h, decenterX_hyp_h*axis4_x[1])
                coeffs_hyp_h = shift_z(coeffs_hyp_h, decenterX_hyp_h*axis4_x[2])
            if decenterY_hyp_h != 0:
                coeffs_hyp_h = shift_x(coeffs_hyp_h, decenterY_hyp_h*axis4_y[0])
                coeffs_hyp_h = shift_y(coeffs_hyp_h, decenterY_hyp_h*axis4_y[1])
                coeffs_hyp_h = shift_z(coeffs_hyp_h, decenterY_hyp_h*axis4_y[2])
            if decenterZ_hyp_h != 0:
                coeffs_hyp_h = shift_x(coeffs_hyp_h, decenterZ_hyp_h*axis4_z[0])
                coeffs_hyp_h = shift_y(coeffs_hyp_h, decenterZ_hyp_h*axis4_z[1])
                coeffs_hyp_h = shift_z(coeffs_hyp_h, decenterZ_hyp_h*axis4_z[2])

            if decenterX_ell_v != 0:
                coeffs_ell_v = shift_x(coeffs_ell_v, decenterX_ell_v*axis2_x[0])
                coeffs_ell_v = shift_y(coeffs_ell_v, decenterX_ell_v*axis2_x[1])
                coeffs_ell_v = shift_z(coeffs_ell_v, decenterX_ell_v*axis2_x[2])
            if decenterY_ell_v != 0:
                coeffs_ell_v = shift_x(coeffs_ell_v, decenterY_ell_v*axis2_y[0])
                coeffs_ell_v = shift_y(coeffs_ell_v, decenterY_ell_v*axis2_y[1])
                coeffs_ell_v = shift_z(coeffs_ell_v, decenterY_ell_v*axis2_y[2])
            if decenterZ_ell_v != 0:
                coeffs_ell_v = shift_x(coeffs_ell_v, decenterZ_ell_v*axis2_z[0])
                coeffs_ell_v = shift_y(coeffs_ell_v, decenterZ_ell_v*axis2_z[1])
                coeffs_ell_v = shift_z(coeffs_ell_v, decenterZ_ell_v*axis2_z[2])

            if decenterX_ell_h != 0:
                coeffs_ell_h = shift_x(coeffs_ell_h, decenterX_ell_h*axis3_x[0])
                coeffs_ell_h = shift_y(coeffs_ell_h, decenterX_ell_h*axis3_x[1])
                coeffs_ell_h = shift_z(coeffs_ell_h, decenterX_ell_h*axis3_x[2])
            if decenterY_ell_h != 0:
                coeffs_ell_h = shift_x(coeffs_ell_h, decenterY_ell_h*axis3_y[0])
                coeffs_ell_h = shift_y(coeffs_ell_h, decenterY_ell_h*axis3_y[1])
                coeffs_ell_h = shift_z(coeffs_ell_h, decenterY_ell_h*axis3_y[2])
            if decenterZ_ell_h != 0:
                coeffs_ell_h = shift_z(coeffs_ell_h, decenterZ_ell_h*axis3_z[0])
                coeffs_ell_h = shift_y(coeffs_ell_h, decenterZ_ell_h*axis3_z[1])
                coeffs_ell_h = shift_x(coeffs_ell_h, decenterZ_ell_h*axis3_z[2])
        else:
            if yaw_ell_v != 0:
                coeffs_ell_v = rotate_z(coeffs_ell_v, yaw_ell_v, center_ell_v[:, 0])
            if pitch_ell_v != 0:
                coeffs_ell_v = rotate_y(coeffs_ell_v, pitch_ell_v, center_ell_v[:, 0])
            if roll_ell_v != 0:
                coeffs_ell_v = rotate_x(coeffs_ell_v, roll_ell_v, center_ell_v[:, 0])


            if pitch_ell_h != 0:
                coeffs_ell_h = rotate_y(coeffs_ell_h, pitch_ell_h, center_ell_h[:, 0])
            if yaw_ell_h != 0:
                coeffs_ell_h = rotate_z(coeffs_ell_h, yaw_ell_h, center_ell_h[:, 0])
            if roll_ell_h != 0:
                coeffs_ell_h = rotate_x(coeffs_ell_h, roll_ell_h, center_ell_h[:, 0])

            if yaw_hyp_v != 0:
                coeffs_hyp_v = rotate_z(coeffs_hyp_v, yaw_hyp_v, center_hyp_v[:, 0])
            if pitch_hyp_v != 0:
                coeffs_hyp_v = rotate_y(coeffs_hyp_v, pitch_hyp_v, center_hyp_v[:, 0])
            if roll_hyp_v != 0:
                coeffs_hyp_v = rotate_x(coeffs_hyp_v, roll_hyp_v, center_hyp_v[:, 0])

            if pitch_hyp_h != 0:
                coeffs_hyp_h = rotate_y(coeffs_hyp_h, pitch_hyp_h, center_hyp_h[:, 0])
            if yaw_hyp_h != 0:
                coeffs_hyp_h = rotate_z(coeffs_hyp_h, yaw_hyp_h, center_hyp_h[:, 0])
            if roll_hyp_h != 0:
                coeffs_hyp_h = rotate_x(coeffs_hyp_h, roll_hyp_h, center_hyp_h[:, 0])

            if decenterX_hyp_v != 0:
                coeffs_hyp_v = shift_x(coeffs_hyp_v, decenterX_hyp_v)
            if decenterY_hyp_v != 0:
                coeffs_hyp_v = shift_y(coeffs_hyp_v, decenterY_hyp_v)
            if decenterZ_hyp_v != 0:
                coeffs_hyp_v = shift_z(coeffs_hyp_v, decenterZ_hyp_v)

            if decenterX_hyp_h != 0:
                coeffs_hyp_h = shift_x(coeffs_hyp_h, decenterX_hyp_h)
            if decenterY_hyp_h != 0:
                coeffs_hyp_h = shift_y(coeffs_hyp_h, decenterY_hyp_h)
            if decenterZ_hyp_h != 0:
                coeffs_hyp_h = shift_z(coeffs_hyp_h, decenterZ_hyp_h)

            if decenterX_ell_v != 0:
                coeffs_ell_v = shift_x(coeffs_ell_v, decenterX_ell_v)
            if decenterY_ell_v != 0:
                coeffs_ell_v = shift_y(coeffs_ell_v, decenterY_ell_v)
            if decenterZ_ell_v != 0:
                coeffs_ell_v = shift_z(coeffs_ell_v, decenterZ_ell_v)

            if decenterX_ell_h != 0:
                coeffs_ell_h = shift_x(coeffs_ell_h, decenterX_ell_h)
            if decenterY_ell_h != 0:
                coeffs_ell_h = shift_y(coeffs_ell_h, decenterY_ell_h)
            if decenterZ_ell_h != 0:
                coeffs_ell_h = shift_z(coeffs_ell_h, decenterZ_ell_h)

        if bool_point_source:
            source = mpmath_zeros((3, ray_num * ray_num))
            source[0, :] =+ source_shift[0] 
            source[1, :] =+ source_shift[1]
            source[2, :] =+ source_shift[2]
            # source[1, :] =+ source_shift[1] + np.random.uniform(-1e-4, 1e-4, ray_num_H * ray_num_V)
            # source[2, :] =+ source_shift[2] + np.random.uniform(-1e-4, 1e-4, ray_num_H * ray_num_V)
            if option_axial:
                rand_p0h = mpmath_linspace(atan((y1_h-source_shift[1]) / (x1_h-source_shift[0])), atan((y2_h-source_shift[1]) / (x2_h-source_shift[0])), ray_num)
                rand_p0v = mpmath_linspace(atan((y1_v-source_shift[2]) / (x1_v-source_shift[0])), atan((y2_v-source_shift[2]) / (x2_v-source_shift[0])), ray_num)
                rand_p0h = rand_p0h - mpmath_mean(rand_p0h)
                rand_p0v = rand_p0v - mpmath_mean(rand_p0v)
            if not option_axial:
                rand_p0h = mpmath_linspace(atan(y1_h / x1_h), atan(y2_h / x2_h), ray_num)
                rand_p0v = mpmath_linspace(atan(y1_v / x1_v), atan(y2_v / x2_v), ray_num)

            phai0 = mpmath_zeros((3, ray_num * ray_num))
            for i in range(ray_num):
                # rand_p0v_here = rand_p0v[i]
                # phai0[1, ray_num * i:ray_num * (i + 1)] = tan(rand_p0h)
                # phai0[2, ray_num * i:ray_num * (i + 1)] = tan(rand_p0v_here)
                # phai0[0, ray_num * i:ray_num * (i + 1)] = 1.
                rand_p0v_here = rand_p0v[i]

                # rand_p0h は 1×ray_num の行ベクトルなので、そのまま elementwise_tan
                tan_rand_p0h = elementwise_tan(rand_p0h)

                # rand_p0v_here はスカラーなので直接 tan
                tan_rand_p0v_here = tan(rand_p0v_here)

                phai0[1, ray_num * i : ray_num * (i + 1)] = tan_rand_p0h
                phai0[2, ray_num * i : ray_num * (i + 1)] = tan_rand_p0v_here
                phai0[0, ray_num * i : ray_num * (i + 1)] = mpf(1)

            phai0 = normalize_vector(phai0)

            vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
            reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))


            vmirr_ell = mirr_ray_intersection(coeffs_ell_v, reflect1, vmirr_hyp)
            reflect2 = reflect_ray(reflect1, norm_vector(coeffs_ell_v, vmirr_ell))
            # if option == 'ray':
            #     plot_ray_sideview(75,85,2,reflect2,vmirr_ell,ray_num)

            hmirr_ell = mirr_ray_intersection(coeffs_ell_h, reflect2, vmirr_ell)
            reflect3 = reflect_ray(reflect2, norm_vector(coeffs_ell_h, hmirr_ell))

            hmirr_hyp = mirr_ray_intersection(coeffs_hyp_h, reflect3, hmirr_ell,negative=True)
            reflect4 = reflect_ray(reflect3, norm_vector(coeffs_hyp_h, hmirr_hyp))

            # if option == 'ray':
            #     plot_ray_sideview(146,1,2,reflect4,hmirr_hyp,ray_num)

            mean_reflect4 = mpmath_mean(reflect4,1)
            # print(mean_reflect2)
            # print(np.sum(mean_reflect2*mean_reflect2))
            mean_reflect4 = normalize_vector(mean_reflect4)
            # print(mean_reflect2)
            # print(np.sum(mean_reflect2*mean_reflect2))

            coeffs_det = mpmath_zeros(10)
            coeffs_det[6] = 1
            coeffs_det[9] = -(s2f_middle + defocus)
            detcenter = plane_ray_intersection(coeffs_det, reflect4, hmirr_hyp)
            # angle = reflect2-mean_reflect2[:, np.newaxis]
            angle = reflect4

            if option == 'sep' or option == 'wave' or option == 'ray' or option == 'ray_wave':
                # 範囲内の値を間引く
                original_array = list(range(ray_num**2))
                thinned_array_v_y = original_array[round((ray_num-1)/2)::ray_num]
                # 範囲内の値を間引く
                start = round(ray_num*(ray_num-1)/2)
                end = round(ray_num*(ray_num+1)/2)
                thinned_array_h_y = crop(start, end, 1)
                # angle_h = atan(angle[1, :]/angle[0, :])
                # angle_v = atan(angle[2, :]/angle[0, :])
                angle_h = elementwise_atan(elementwise_div(angle[1, :], angle[0, :]))
                angle_v = elementwise_atan(elementwise_div(angle[2, :], angle[0, :]))
                print('angle_h.rows, angle_h.cols', angle_h.rows, angle_h.cols)
                print('thinned_array_v_y', thinned_array_v_y)
                angle_v_sep_y = select_columns_matrix(angle_v, thinned_array_v_y)
                angle_h_sep_y = select_columns_matrix(angle_h, thinned_array_h_y)

                plt.figure()
                plt.scatter(detcenter[1, :], detcenter[2, :], c='k')
                plt.scatter(select_columns_matrix(detcenter[1, :], thinned_array_h_y), select_columns_matrix(detcenter[2, :], thinned_array_h_y), c='r')
                
                plt.scatter(select_columns_matrix(detcenter[1, :], thinned_array_v_y), select_columns_matrix(detcenter[2, :], thinned_array_v_y), c='b')
                # plt.show()


                output_equal_v = mpmath_linspace(angle_v_sep_y[0],angle_v_sep_y[angle_v_sep_y.cols - 1],len(angle_v_sep_y))
                output_equal_h = mpmath_linspace(angle_h_sep_y[0],angle_h_sep_y[angle_h_sep_y.cols - 1],len(angle_h_sep_y))

                # interp_func_v = interp1d(angle_v_sep_y, rand_p0v, kind='linear')
                # interp_func_h = interp1d(angle_h_sep_y, rand_p0h, kind='linear')

                # rand_p0v_new = interp_func_v(output_equal_v)
                # rand_p0h_new = interp_func_h(output_equal_h)

                rand_p0v_new = mpmath_interp1d_wrapper_to_matrix(angle_v_sep_y, rand_p0v, output_equal_v, kind='linear')
                rand_p0h_new = mpmath_interp1d_wrapper_to_matrix(angle_h_sep_y, rand_p0h, output_equal_h, kind='linear')


                phai0 = mpmath_zeros((3, ray_num * ray_num))
                for i in range(ray_num):
                    # rand_p0v_here = rand_p0v_new[i]
                    # phai0[1, ray_num * i:ray_num * (i + 1)] = tan(rand_p0h_new)
                    # phai0[2, ray_num * i:ray_num * (i + 1)] = tan(rand_p0v_here)
                    # phai0[0, ray_num * i:ray_num * (i + 1)] = 1.

                    rand_p0v_here = rand_p0v_new[i]

                    # rand_p0h は 1×ray_num の行ベクトルなので、そのまま elementwise_tan
                    tan_rand_p0h_new = elementwise_tan(rand_p0h_new)

                    # rand_p0v_here はスカラーなので直接 tan
                    tan_rand_p0v_here = tan(rand_p0v_here)

                    phai0[1, ray_num * i : ray_num * (i + 1)] = tan_rand_p0h_new
                    phai0[2, ray_num * i : ray_num * (i + 1)] = tan_rand_p0v_here
                    phai0[0, ray_num * i : ray_num * (i + 1)] = mpf(1)

                phai0 = normalize_vector(phai0)

                vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
                reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))

                dist0to1 = mpmath_norm(vmirr_hyp - source,axis=0)


                vmirr_ell = mirr_ray_intersection(coeffs_ell_v, reflect1, vmirr_hyp)
                reflect2 = reflect_ray(reflect1, norm_vector(coeffs_ell_v, vmirr_ell))

                dist1to2 = mpmath_norm(vmirr_ell - vmirr_hyp,axis=0)
                hmirr_ell = mirr_ray_intersection(coeffs_ell_h, reflect2, vmirr_ell)
                reflect3 = reflect_ray(reflect2, norm_vector(coeffs_ell_h, hmirr_ell))
                dist2to3 = mpmath_norm(hmirr_ell - vmirr_ell,axis=0)

                hmirr_hyp = mirr_ray_intersection(coeffs_hyp_h, reflect3, hmirr_ell,negative=True)
                reflect4 = reflect_ray(reflect3, norm_vector(coeffs_hyp_h, hmirr_hyp))
                dist3to4 = mpmath_norm(hmirr_hyp - hmirr_ell,axis=0)

                coeffs_det = mpmath_zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)
                detcenter = plane_ray_intersection(coeffs_det, reflect4, hmirr_hyp)

                if option == 'ray':
                    from scipy.spatial import cKDTree
                    def mindist(A,B):
                        tree = cKDTree(B.T)
                        dist, idx = tree.query(A.T, k=1)  # Aの各点からBへの最近点距離
                        min_dist = mpmath_min(dist)
                        return min_dist
                    print('======================')
                    print('workX srs 1st',mpmath_min(vmirr_hyp[0,:]) - mpmath_max(source[0,:]))
                    print('workX 1st 2nd',mpmath_min(vmirr_ell[0,:]) - mpmath_max(vmirr_hyp[0,:]))
                    print('workX 2nd 3rd',mpmath_min(hmirr_ell[0,:]) - mpmath_max(vmirr_ell[0,:]))
                    print('workX 3rd 4th',mpmath_min(hmirr_hyp[0,:]) - mpmath_max(hmirr_ell[0,:]))
                    print('workX 4th fcs',mpmath_min(detcenter[0,:]) - mpmath_max(hmirr_hyp[0,:]))
                    print('======================')
                    # print('workAbs srs 1st',mindist(source,vmirr_hyp))
                    # print('workAbs 1st 2nd',mindist(vmirr_hyp,vmirr_ell))
                    # print('workAbs 2nd 3rd',mindist(vmirr_ell,hmirr_ell))
                    # print('workAbs 3rd 4th',mindist(hmirr_ell,hmirr_hyp))
                    # print('workAbs 4th fcs',mindist(hmirr_hyp,detcenter))

                    print('1st W upper',mpmath_norm(vmirr_hyp[:,0] - vmirr_hyp[:,ray_num-1]))
                    print('1st W lower',mpmath_norm(vmirr_hyp[:,-1] - vmirr_hyp[:,-ray_num]))
                    print('2nd W upper',mpmath_norm(vmirr_ell[:,0] - vmirr_ell[:,ray_num-1]))
                    print('2nd W lower',mpmath_norm(vmirr_ell[:,-1] - vmirr_ell[:,-ray_num]))
                    print('3rd W upper',mpmath_norm(hmirr_ell[:,0] - hmirr_ell[:,-ray_num]))
                    print('3rd W lower',mpmath_norm(hmirr_ell[:,ray_num-1] - hmirr_ell[:,-1]))
                    print('4th W lower',mpmath_norm(hmirr_hyp[:,0] - hmirr_hyp[:,-ray_num]))
                    print('4th W upper',mpmath_norm(hmirr_hyp[:,ray_num-1] - hmirr_hyp[:,-1]))

                    fig,axs = plt.subplots(2,1,sharex=True)
                    axs[0].plot(vmirr_hyp[0,:],vmirr_hyp[1,:])
                    axs[0].plot(vmirr_ell[0,:],vmirr_ell[1,:])
                    axs[0].plot(hmirr_ell[0,:],hmirr_ell[1,:])
                    axs[0].plot(hmirr_hyp[0,:],hmirr_hyp[1,:])
                    axs[0].plot(detcenter[0,:],detcenter[1,:])
                    axs[0].set_ylabel('Horizontal')

                    axs[1].plot(vmirr_hyp[0,:],vmirr_hyp[2,:])
                    axs[1].plot(vmirr_ell[0,:],vmirr_ell[2,:])
                    axs[1].plot(hmirr_ell[0,:],hmirr_ell[2,:])
                    axs[1].plot(hmirr_hyp[0,:],hmirr_hyp[2,:])
                    axs[1].plot(detcenter[0,:],detcenter[2,:])
                    axs[1].set_ylabel('Vertical')
                    axs[0].axis('equal')
                    axs[1].axis('equal')
                    plt.savefig('raytrace_mirror_configuration.png')
                    # plt.show()

                    vec0to1 = normalize_vector(vmirr_hyp - source)
                    vec1to2 = normalize_vector(vmirr_ell - vmirr_hyp)
                    vec2to3 = normalize_vector(hmirr_ell - vmirr_ell)
                    vec3to4 = normalize_vector(hmirr_hyp - hmirr_ell)
                    vec4to5 = normalize_vector(detcenter - hmirr_hyp)

                    # grazing_angle_1 = acos(np.sum(vec0to1 * vec1to2, axis=0)) / 2
                    # grazing_angle_2 = acos(np.sum(vec1to2 * vec2to3, axis=0)) / 2
                    # grazing_angle_3 = acos(np.sum(vec2to3 * vec3to4, axis=0)) / 2
                    # grazing_angle_4 = acos(np.sum(vec3to4 * vec4to5, axis=0)) / 2

                    # ### imshow colorbar
                    # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                    # im0 = axs[0, 0].imshow(grazing_angle_1.reshape(ray_num, ray_num), cmap='jet', aspect='auto')
                    # im1 = axs[0, 1].imshow(grazing_angle_2.reshape(ray_num, ray_num), cmap='jet', aspect='auto')
                    # im2 = axs[1, 0].imshow(grazing_angle_3.reshape(ray_num, ray_num), cmap='jet', aspect='auto')
                    # im3 = axs[1, 1].imshow(grazing_angle_4.reshape(ray_num, ray_num), cmap='jet', aspect='auto')
                    # fig.colorbar(im0, ax=axs[0, 0])
                    # fig.colorbar(im1, ax=axs[0, 1])
                    # fig.colorbar(im2, ax=axs[1, 0])
                    # fig.colorbar(im3, ax=axs[1, 1])
                    # ### title
                    # axs[0, 0].set_title(f'Grazing Angle 1\n{1e3*grazing_angle_1.min():.2f}-{1e3*grazing_angle_1.max():.2f} mrad')
                    # axs[0, 1].set_title(f'Grazing Angle 2\n{1e3*grazing_angle_2.min():.2f}-{1e3*grazing_angle_2.max():.2f} mrad')
                    # axs[1, 0].set_title(f'Grazing Angle 3\n{1e3*grazing_angle_3.min():.2f}-{1e3*grazing_angle_3.max():.2f} mrad')
                    # axs[1, 1].set_title(f'Grazing Angle 4\n{1e3*grazing_angle_4.min():.2f}-{1e3*grazing_angle_4.max():.2f} mrad')
                    # 
                    # plt.figure(figsize=(10, 5))
                    # plt.scatter(detcenter[1, :], detcenter[2, :])
                    # plt.show()

                angle = reflect4

            if option == 'wave':
                print('diverg angle H',atan(y1_h / x1_h) - atan(y2_h / x2_h))
                print('diverg angle V',atan(y1_v / x1_v) - atan(y2_v / x2_v))
                # 全データからランダムに10%だけを選択
                sample_indices = np.random.choice(detcenter.shape[1], size=int(detcenter.shape[1]*0.001), replace=False)

                theta_y = -mpmath_mean(atan(angle[2, :]/angle[0, :]))
                theta_z = mpmath_mean(atan(angle[1, :]/angle[0, :]))
                source = mpmath_zeros((3,1))
                if option_rotate==True:
                    reflect4_rotated = rotate_vectors(reflect4, -theta_y, -theta_z)
                    focus_apprx = mpmath_mean(detcenter,axis=1)
                    hmirr_ell_points_rotated = rotate_points(hmirr_ell, focus_apprx, -theta_y, -theta_z)
                    vmirr_ell_points_rotated = rotate_points(vmirr_ell, focus_apprx, -theta_y, -theta_z)
                    hmirr_hyp_points_rotated = rotate_points(hmirr_hyp, focus_apprx, -theta_y, -theta_z)
                    vmirr_hyp_points_rotated = rotate_points(vmirr_hyp, focus_apprx, -theta_y, -theta_z)
                    source_rotated = rotate_points(source, focus_apprx, -theta_y, -theta_z)

                    hmirr_ell_points_rotated_grid = hmirr_ell_points_rotated
                    vmirr_ell_points_rotated_grid = vmirr_ell_points_rotated
                    hmirr_hyp_points_rotated_grid = hmirr_hyp_points_rotated
                    vmirr_hyp_points_rotated_grid = vmirr_hyp_points_rotated

                else:
                    reflect2_rotated = reflect2
                    hmirr_hyp_points_rotated = hmirr_hyp
                    vmirr_hyp_points_rotated = vmirr_hyp
                    source_rotated = source
                coeffs_det = mpmath_zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)
                detcenter = plane_ray_intersection(coeffs_det, reflect4_rotated, hmirr_hyp_points_rotated)

                vec0to1 = normalize_vector(vmirr_hyp_points_rotated_grid - source_rotated)
                vec1to2 = normalize_vector(vmirr_ell_points_rotated_grid - vmirr_hyp_points_rotated_grid)
                vec2to3 = normalize_vector(hmirr_ell_points_rotated_grid - vmirr_ell_points_rotated_grid)
                vec3to4 = normalize_vector(hmirr_hyp_points_rotated_grid - hmirr_ell_points_rotated_grid)
                vec4to5 = normalize_vector(detcenter - hmirr_hyp_points_rotated_grid)

                vmirr_norm = normalize_vector( (-vec1to2 + vec0to1) / 2 )
                hmirr_norm = normalize_vector( (-vec2to3 + vec1to2) / 2 )
                vmirr2_norm = normalize_vector( (-vec3to4 + vec2to3) / 2 )
                hmirr2_norm = normalize_vector( (-vec4to5 + vec3to4) / 2 )

                if fabs(defocusForWave) > 1e-9:
                    coeffs_det2 = mpmath_zeros(10)
                    coeffs_det2[6] = 1
                    coeffs_det2[9] = -(s2f_middle + defocus+defocusForWave)
                    detcenter2 = plane_ray_intersection(coeffs_det2, reflect4_rotated, hmirr_hyp_points_rotated)
                    return source_rotated, vmirr_hyp_points_rotated_grid, hmirr_hyp_points_rotated_grid, vmirr_ell_points_rotated_grid, hmirr_ell_points_rotated_grid, detcenter, detcenter2, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vmirr2_norm, hmirr2_norm, vec0to1, vec1to2, vec2to3, vec3to4
                else:
                    return source_rotated, vmirr_hyp_points_rotated_grid, hmirr_hyp_points_rotated_grid, vmirr_ell_points_rotated_grid, hmirr_ell_points_rotated_grid, detcenter, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vmirr2_norm, hmirr2_norm, vec0to1, vec1to2, vec2to3, vec3to4

            
            hmirr_hyp0 = hmirr_hyp.copy()
            if option_tilt:
                angles_between_rad, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect4,norm_vector(coeffs_det, detcenter))
                theta_z = (np.max(angles_yx_rad) + np.min(angles_yx_rad))/2
                theta_y = -(np.max(angles_zx_rad) + np.min(angles_zx_rad))/2
                if option == 'ray':
                    print('NA_h')
                    print(sin((np.max(angles_yx_rad) - np.min(angles_yx_rad))/2))
                    print('angles_yx_rad',np.sort(angles_yx_rad)[:5][::-1])
                    print('angles_yx_rad',np.sort(angles_yx_rad)[-5:][::-1])
                    print('NA_v')
                    print(sin((np.max(angles_zx_rad) - np.min(angles_zx_rad))/2))
                    print('angles_zx_rad',np.sort(angles_zx_rad)[:5][::-1])
                    print('angles_zx_rad',np.sort(angles_zx_rad)[-5:][::-1])
                    print('type(detcenter[0,0])',type(detcenter[0,0]))
                    print('theta_y',theta_y)
                    print('theta_z',theta_z)

                reflect4_rotated = rotate_vectors(reflect4, -theta_y, -theta_z)
                focus_apprx = mpmath_mean(detcenter,axis=1)
                hmirr_hyp_points_rotated = rotate_points(hmirr_hyp, focus_apprx, -theta_y, -theta_z)
                coeffs_det = mpmath_zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)
                detcenter = plane_ray_intersection(coeffs_det, reflect4_rotated, hmirr_hyp_points_rotated)


                hmirr_hyp = hmirr_hyp_points_rotated.copy()
                reflect4 = reflect4_rotated.copy()
                angle = reflect4.copy()


            if option == 'ray_wave':
                if option_HighNA == True:
                    defocusWave = 1e-2
                    lambda_ = 13.5
                else:
                    defocusWave = 1e-3
                    lambda_ = 1.35
                coeffs_det2 = mpmath_zeros(10)
                coeffs_det2[6] = 1
                coeffs_det2[9] = -(s2f_middle + defocus + defocusWave)
                detcenter2 = plane_ray_intersection(coeffs_det2, reflect4, hmirr_hyp)

                dist4tofocus = mpmath_norm(detcenter - hmirr_hyp, axis=0)
                print('detcenter.dtype',type(detcenter))
                print('hmirr_hyp.dtype',type(hmirr_hyp))
                print('dist4tofocus.dtype',type(dist4tofocus))
                print('len(dist4tofocus)',len(dist4tofocus))
                print('(detcenter - hmirr_hyp).dtype',type(detcenter - hmirr_hyp))

                totalDist = matrix(dist0to1) + matrix(dist1to2) + matrix(dist2to3) + matrix(dist3to4) + matrix(dist4tofocus)
                DistError = (totalDist - mpmath_mean(totalDist))*1e9



                dist4tofocus2 = mpmath_norm(detcenter2 - hmirr_hyp, axis=0)
                totalDist2 = matrix(dist0to1) + matrix(dist1to2) + matrix(dist2to3) + matrix(dist3to4) + matrix(dist4tofocus2)
                print('totalDist2.dtype',type(totalDist2))
                print('totalDist2.shape',totalDist2.rows, totalDist2.cols)
                DistError2 = (totalDist2 - mpmath_mean(totalDist2))*1e9
                # print('detcenter',mpmath_mean(detcenter,axis=1))
                # print('detcenter2',mpmath_mean(detcenter2,axis=1))
                # print('dist0to1',mpmath_mean(dist0to1))
                # print('dist1to2',mpmath_mean(dist1to2))
                # print('dist2to3',mpmath_mean(dist2to3))
                # print('dist3to4',mpmath_mean(dist3to4))
                # print('totalDist',mpmath_mean(totalDist))
                # print('dist4tofocus std',mpmath_mean(dist4tofocus))
                # print('dist0to1 std',np.std(dist0to1))
                # print('dist1to2 std',np.std(dist1to2))
                # print('dist2to3 std',np.std(dist2to3))
                # print('dist3to4 std',np.std(dist3to4))
                # print('dist4tofocus std',np.std(dist4tofocus))
                # print('totalDist std',np.std(totalDist))
                # print('np.std(totalDist)',np.std(totalDist))
                # print('mpmath_mean(totalDist)',mpmath_mean(totalDist))
                # print('mpmath_mean(totalDist2)',mpmath_mean(totalDist2))
                # print('mpmath_mean(totalDist2-totalDist)',mpmath_mean(totalDist2-totalDist))
                # print('np.std(totalDist2-totalDist)',np.std(totalDist2-totalDist))
                ### 変数をnp.float64に変換 
                detcenter = mpmath_matrix_to_numpy(detcenter)
                detcenter2 = mpmath_matrix_to_numpy(detcenter2)
                DistError = mpmath_matrix_to_numpy(DistError)
                DistError2 = mpmath_matrix_to_numpy(DistError2)
                DistError = DistError.T
                DistError2 = DistError2.T
                ### 1次元配列に変換
                DistError = DistError.flatten()
                DistError2 = DistError2.flatten()
                print('detcenter.shape',detcenter.shape)
                print('detcenter2.shape',detcenter2.shape)
                print('DistError.shape',DistError.shape)
                print('DistError2.shape',DistError2.shape)
                print('type(detcenter)', type(detcenter))
                print('type(detcenter2)', type(detcenter2))
                print('type(DistError)', type(DistError))
                print('type(DistError2)', type(DistError2))

                # 補間するグリッドを作成
                grid_H, grid_V = np.meshgrid(
                    np.linspace(detcenter2[1, :].min(), detcenter2[1, :].max(), ray_num_H),
                    np.linspace(detcenter2[2, :].min(), detcenter2[2, :].max(), ray_num_V)
                )

                CosAngle = angle[0,:]
                # グリッド上にデータを補間 (method: 'linear', 'nearest', 'cubic' から選択)
                if False:
                    matrixDistError2 = griddata((detcenter2[1, :], detcenter2[2, :]), DistError2, (grid_H, grid_V), method='cubic')
                    meanFocus = mpmath_mean(detcenter,axis=1)
                    Sph = mpmath_norm(detcenter2 - meanFocus[:, np.newaxis], axis=0) * 1e9
                    matrixSph2 = griddata((detcenter2[1, :], detcenter2[2, :]), Sph, (grid_H, grid_V), method='cubic')

                    # matrixAngle2 = griddata((detcenter2[1, :], detcenter2[2, :]), CosAngle, (grid_H, grid_V), method='cubic')
                    # matrixWave2 = matrixDistError2 * matrixAngle2 - matrixSph2

                    matrixWave2 = matrixDistError2 - matrixSph2
                    matrixWave2 = matrixWave2 - np.nanmean(matrixWave2)
                else:
                    matrixDistError2 = griddata((detcenter2[1, :], detcenter2[2, :]), DistError2, (grid_H, grid_V), method='cubic')
                    meanFocus = np.mean(detcenter,axis=1)
                    Sph = np.linalg.norm(detcenter2 - meanFocus[:, np.newaxis], axis=0) * 1e9

                    Wave2 = DistError2 - Sph
                    print('meanFocus',meanFocus)
                    print('mpmath_mean(DistError2)',np.mean(DistError2))
                    print('np.std(DistError2)',np.std(DistError2))
                    print('mpmath_mean(Sph)',np.mean(Sph))
                    print('np.std(Sph)',np.std(Sph))
                    print('mpmath_mean(Wave2)',np.mean(Wave2))
                    print('np.std(Wave2)',np.std(Wave2))
                    print('grid_H.shape',grid_H.shape)
                    print('Wave2.shape',Wave2.shape)
                    print('detcenter2.shape',detcenter2.shape)

                    matrixWave2 = griddata((detcenter2[1, :], detcenter2[2, :]), Wave2, (grid_H, grid_V), method='cubic')
                    matrixWave2 = matrixWave2 - np.nanmean(matrixWave2)
                print('matrixWave2.shape', matrixWave2.shape)
                np.savetxt('matrixWave2(nm).txt',matrixWave2)
                tifffile.imwrite('matrixWave2(nm).tiff', matrixWave2)

                matrixWave2_Corrected = plane_correction_with_nan_and_outlier_filter(matrixWave2)
                matrixDistError2_Corrected = plane_correction_with_nan_and_outlier_filter(matrixDistError2)
                print('PV',np.nanmax(matrixWave2_Corrected)-np.nanmin(matrixWave2_Corrected))
                
                ### angle
                grid_H = np.arctan((grid_H - np.mean(grid_H)) / defocusWave)
                grid_V = np.arctan((grid_V - np.mean(grid_V)) / defocusWave)

                wave_error = matrixWave2_Corrected / lambda_
                ### PSF calculation
                if False:
                    phase = wave_error * 2 * pi
                    input_complex = np.exp(1j * phase)
                    offset = defocusWave * (np.sqrt(1. + tan(grid_H)**2 + tan(grid_V)**2) - 1)
                    input_complex *= np.exp(1j * offset)
                    output_complex = np.zeros_like(input_complex, dtype=np.complex128)
                    x_input = defocusWave * tan(grid_H)
                    y_input = defocusWave * tan(grid_V)
                    calcsize = 200e-9
                    xoutput , youtput = np.meshgrid(
                        mpmath_linspace(-calcsize, calcsize, output_complex.shape[1]),
                        mpmath_linspace(-calcsize, calcsize, output_complex.shape[0])
                    )
                    for i_o in range(output_complex.shape[0]):
                        for j_o in range(output_complex.shape[1]):
                            for i_i in range(input_complex.shape[0]):
                                for j_i in range(input_complex.shape[1]):
                                    if np.isnan(input_complex[i_i, j_i]) or np.isnan(output_complex[i_o, j_o]):
                                        continue
                                    else:
                                        dist = defocusWave * np.sqrt(defocusWave**2 + (xoutput[i_o, j_o] - x_input[i_i, j_i])**2 + (youtput[i_o, j_o] - y_input[i_i, j_i])**2)
                                        phase_shift =  - 2 * pi * dist / lambda_
                                        output_complex[i_o, j_o] += input_complex[i_i, j_i] * np.exp(1j * phase_shift) / dist
                    psf = fabs(output_complex)**2
                    psf = psf / np.max(psf)  # Normalize PSF
                    plt.figure()
                    plt.imshow(psf, extent=(xoutput.min()*1e9, xoutput.max()*1e9, youtput.min()*1e9, youtput.max()*1e9), cmap='jet')
                    plt.colorbar(label='Intensity')
                    plt.title('PSF')
                    plt.xlabel('X (nm)')
                    plt.ylabel('Y (nm)')


                plt.figure()
                plt.pcolormesh(grid_H, grid_V, matrixWave2_Corrected/lambda_, cmap='jet', shading='auto',vmin = -1/4,vmax = 1/4)
                plt.colorbar(label='\u03BB')
                # plt.colorbar(label='wavefront error (nm)')
                plt.title(f'PV 6σ={np.nanstd(matrixWave2_Corrected/lambda_)*6}')
                plt.savefig(os.path.join(directory_name, 'waveRaytrace_Corrected.png'), transparent=True, dpi=300)

                plt.figure()
                plt.pcolormesh(grid_H, grid_V, matrixWave2_Corrected/lambda_, cmap='jet', shading='auto',vmin = -1/128,vmax = 1/128)
                plt.colorbar(label='\u03BB')
                # plt.colorbar(label='wavefront error (nm)')
                plt.title(f'PV 6σ={np.nanstd(matrixWave2_Corrected/lambda_)*6}')
                plt.savefig(os.path.join(directory_name, 'waveRaytrace_Corrected_2.png'), transparent=True, dpi=300)

                np.savetxt(os.path.join(directory_name, 'matrixWave2_Corrected(lambda).txt'), matrixWave2_Corrected/lambda_)
                print('PV 6σ', np.nanstd(matrixWave2_Corrected/lambda_)*6)
                print('PV', np.nanmax(matrixWave2_Corrected/lambda_) - np.nanmin(matrixWave2_Corrected/lambda_))
                # plt.show()
                # rectified_img = extract_affine_square_region(matrixWave2_Corrected/lambda_, target_size=256)
                rectified_img = extract_affine_square_region(matrixWave2/lambda_, target_size=256)

                
                print('rectified_img.dtype', rectified_img.dtype)
                print('rectified_img.shape', rectified_img.shape)

                # plt.figure()
                # plt.imshow(rectified_img[1:-2, 1:-2], cmap='jet',vmin = -1/4,vmax = 1/4)
                # plt.colorbar(label='\u03BB')
                # plt.title("Affine-Corrected Square Cutout")
                # # plt.show()
                

                assesorder = 5
                fit_datas, inner_products, orders = lf.match_legendre_multi(rectified_img[1:-2, 1:-2], assesorder)
                length = len(inner_products)
                pvs = np.zeros(length+1)
                fig, axes = plt.subplots(assesorder, assesorder, figsize=(16, 16))
                for i in range(length):
                    ny = orders[i][0]
                    nx = orders[i][1]
                    print(f"ny: {ny}, nx: {nx}, Inner Product: {inner_products[i]:.3e}")
                    axes[ny, nx].imshow(fit_datas[i], cmap='jet', vmin=-1/4, vmax=1/4)
                    pvs[i] = (np.nanmax(fit_datas[i]) - np.nanmin(fit_datas[i])) * np.sign(inner_products[i])
                    axes[ny, nx].set_title(f"ny: {ny}, nx: {nx} \n Inner Product: {inner_products[i]:.3e} \n PV: {pvs[i]:.3e}")
                    axes[ny, nx].axis('off')

                    ### set colorbar for each subplot
                    # cbar = plt.colorbar(axes[ny, nx].images[0], ax=axes[ny, nx], fraction=0.046, pad=0.04)
                axes[-1, -1].imshow(rectified_img[1:-2, 1:-2], cmap='jet', vmin=-1/4, vmax=1/4)
                cbar = plt.colorbar(axes[-1, -1].images[0], ax=axes[-1, -1], fraction=0.046, pad=0.04)
                fit_sum = np.sum(fit_datas, axis=0)
                axes[-2, -1].imshow(fit_sum, cmap='jet', vmin=-1/4, vmax=1/4)
                # cbar = plt.colorbar(axes[-2, -1].images[0], ax=axes[-2, -1], fraction=0.046, pad=0.04)
                axes[-1, -2].imshow(rectified_img[1:-2, 1:-2]-fit_sum, cmap='jet', vmin=-1/4, vmax=1/4)
                # cbar = plt.colorbar(axes[-1, -2].images[0], ax=axes[-1, -2], fraction=0.046, pad=0.04)
                np.savetxt(os.path.join(directory_name, 'inner_products.csv'), inner_products, delimiter=',')
                np.savetxt(os.path.join(directory_name, 'orders.csv'), orders, delimiter=',')

                plt.tight_layout()
                if option_legendre:
                    plt.close()
                    print('inner_products',inner_products)
                    print('orders',orders)
                    pvs[-1] = np.nanstd(matrixWave2_Corrected/lambda_)*6 * np.sign(np.sum(inner_products))
                    return inner_products, orders, pvs
                else:
                    plt.savefig(os.path.join(directory_name, 'legendre_fit.png'), transparent=True, dpi=300)
                    plt.show()



                if False:
                    # psf, x_out, y_out = fresnel_psf(matrixWave2_Corrected, lambda_=lambda_, z=-defocusWave, grid_x=grid_H, grid_y=grid_V)
                    calcrange=1.e-6
                    psf, x_out, y_out = fresnel_integral(
                        phi=matrixWave2_Corrected*1e-9,
                        grid_x=grid_H-mpmath_mean(grid_H),
                        grid_y=grid_V-mpmath_mean(grid_V),
                        lambda_=lambda_*1e-9,
                        z=-defocusWave,
                        x_out_range=(-calcrange, calcrange),
                        y_out_range=(-calcrange, calcrange),
                        dx_out=calcrange/65,
                        dy_out=calcrange/65,
                    )

                    def compute_fwhm(x, intensity_1d):
                        """1次元の強度分布から FWHM を計算"""
                        dx = fabs(x[1] - x[0])
                        num_over_half_max = np.sum(intensity_1d >= 0.5 * np.max(intensity_1d))
                        fwhm = (num_over_half_max-1) * dx
                        return fwhm
                    psf_x = psf[psf.shape[0] // 2, :]
                    psf_y = psf[:, psf.shape[1] // 2]
                    half_max = 0.5 * np.max(psf)
                    mask = psf >= half_max
                    dx = fabs(x_out[1] - x_out[0])
                    dy = fabs(y_out[1] - y_out[0])
                    area = np.sum(mask) * dx * dy
                    effective_diameter = np.sqrt(area)
                    # fwhm_x = effective_diameter
                    # fwhm_y = effective_diameter
                    fwhm_x = compute_fwhm(x_out, psf_x)
                    fwhm_y = compute_fwhm(y_out, psf_y)

                    print(f"FWHM_x = {fwhm_x:.3e} m")
                    print(f"FWHM_y = {fwhm_y:.3e} m")
                    print(f"Effective spot diameter (sqrt of half-max area) = {effective_diameter:.3e} m")
                    
                    plt.imshow(psf, extent=[x_out[0], x_out[-1], y_out[0], y_out[-1]], origin='lower', cmap='hot')
                    plt.xlabel("x [m]")
                    plt.ylabel("y [m]")
                    plt.title("PSF (Fresnel Approx.)")
                    plt.colorbar()
                    # plt.show()
                    fig, ax = plt.subplots(2, 1, figsize=(6, 5))
                    ax[0].plot(x_out, psf_x)
                    ax[0].axhline(np.max(psf_x)/2, color='gray', linestyle='--')
                    ax[0].set_title(f"PSF (Fresnel Approx.) X Profile — FWHM = {fwhm_x*1e6:.2f} μm")
                    ax[0].set_xlabel("x [m]")
                    ax[0].set_ylabel("Intensity")

                    ax[1].plot(y_out, psf_y)
                    ax[1].axhline(np.max(psf_y)/2, color='gray', linestyle='--')
                    ax[1].set_title(f"Y Profile — FWHM = {fwhm_y*1e6:.2f} μm")
                    ax[1].set_xlabel("y [m]")
                    ax[1].set_ylabel("Intensity")

                    plt.tight_layout()
                    plt.show()

                    # plt.figure()
                    # sample_detcenter = detcenter2.copy()
                    # sample_DistError = DistError2.copy()
                    # sample = np.vstack([sample_detcenter, sample_DistError])
                    # sizeh_here = ray_num_H
                    # sizev_here = ray_num_V
                    # while sizeh_here > 33:
                    #     sample, sizev_here, sizeh_here = downsample_array_any_n(sample, sizev_here, sizeh_here, 2, 2)
                    # scatter = plt.scatter(sample[1, :], sample[2, :],c=sample[3,:], cmap='jet')
                    # # カラーバーを追加
                    # plt.colorbar(scatter, label='OPL error (nm)')
                    # plt.axis('equal')
                    # plt.show()

                return

            if option == 'ray':
                print(theta1_v)
                print(cos(theta1_v)*s2f_middle)
                print(theta1_h)
                print(mpmath_mean(detcenter[0,:]))
                print(mpmath_mean(detcenter[1,:]))
                print(mpmath_mean(detcenter[2,:]))
                # print(mpmath_mean(detcenter[0,:]))
                print(coeffs_det)
                print('s2f_H',s2f_H)
                print('s2f_V',s2f_V)
                mabiki = round(np.sqrt(ray_num_H*ray_num_V)/50)
                mabiki =  1
                defocussize = 4e-6
                coeffs_det = mpmath_zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus) + defocussize
                detcenter1 = plane_ray_intersection(coeffs_det, reflect4, hmirr_hyp)

                coeffs_det = mpmath_zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus) - defocussize
                detcenter2 = plane_ray_intersection(coeffs_det, reflect4, hmirr_hyp)
 
                vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
                reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))

                angle_1st, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect1,norm_vector(coeffs_hyp_v, vmirr_hyp))



                coeffs_det = mpmath_zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)

                detcenter0 = plane_ray_intersection(coeffs_det, reflect4, hmirr_hyp)

                # angles_between_rad, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect_ray(reflect4, norm_vector(coeffs_det, detcenter0)),norm_vector(coeffs_det, detcenter0))
                angles_between_rad, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect4,norm_vector(coeffs_det, detcenter0))
                print('NA_h')
                print(sin((np.max(angles_yx_rad) - np.min(angles_yx_rad))/2))
                print('angles_yx_rad',np.sort(angles_yx_rad)[:5][::-1])
                print('angles_yx_rad',np.sort(angles_yx_rad)[-5:][::-1])
                print('NA_v')
                print(sin((np.max(angles_zx_rad) - np.min(angles_zx_rad))/2))
                print('angles_zx_rad',np.sort(angles_zx_rad)[:5][::-1])
                print('angles_zx_rad',np.sort(angles_zx_rad)[-5:][::-1])
                print('type(detcenter[0,0])',type(detcenter[0,0]))
                if option_tilt:
                    reflect4_rotated = rotate_vectors(reflect4, -theta_y, -theta_z)
                    focus_apprx = mpmath_mean(detcenter,axis=1)
                    hmirr_hyp_points_rotated = rotate_points(hmirr_hyp, focus_apprx, -theta_y, -theta_z)
                    coeffs_det = mpmath_zeros(10)
                    coeffs_det[6] = 1
                    coeffs_det[9] = -(s2f_middle + defocus)
                    detcenter = plane_ray_intersection(coeffs_det, reflect4_rotated, hmirr_hyp_points_rotated)
                    hmirr_hyp = hmirr_hyp_points_rotated
                    reflect4 = reflect4_rotated
                    angle = reflect4

                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[::ray_num]
                thinned_array_v_r = first_thinned_array[::mabiki]
                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[round((ray_num-1)/2)::ray_num]
                thinned_array_v_y = first_thinned_array[::mabiki]
                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[ray_num-1::ray_num]
                thinned_array_v_g = first_thinned_array[::mabiki]
                # 範囲内の値を間引く
                start = 0
                end = ray_num
                thinned_array_h_r = crop(start, end, mabiki)
                # 範囲内の値を間引く
                start = round(ray_num*(ray_num-1)/2)
                end = round(ray_num*(ray_num+1)/2)
                thinned_array_h_y = crop(start, end, mabiki)
                # 範囲内の値を間引く
                start = ray_num**2 - ray_num
                end = ray_num**2
                thinned_array_h_g = crop(start, end, mabiki)

                # プロットの準備
                fig, axs = plt.subplots(2, 2, figsize=(10, 15))  # 2つのプロットを並べる
                # fig, axs = plt.subplots(2, 4, figsize=(10, 45))  # 2つのプロットを並べる
                input_val = -coeffs_det[9]

                # 初期のプロット
                # thinned_array_h_r, thinned_array_h_y, thinned_array_h_g の1/3と2/3のインデックス
                third_r = len(thinned_array_h_r) *2 // 3
                third_y = len(thinned_array_h_y) *2 // 3
                third_g = len(thinned_array_h_g) *2 // 3

                # 前1/3のプロット（赤）
                axs[0, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_h_r[:third_r]), select_columns_matrix(detcenter2[0, :], thinned_array_h_r[:third_r])],
                               [select_columns_matrix(detcenter1[1, :], thinned_array_h_r[:third_r]), select_columns_matrix(detcenter2[1, :], thinned_array_h_r[:third_r])], c='r')

                # 後ろ1/3のプロット（ピンク）
                axs[0, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_h_r[len(thinned_array_h_r)-third_r:]), select_columns_matrix(detcenter2[0, :], thinned_array_h_r[len(thinned_array_h_r)-third_r:])],
                               [select_columns_matrix(detcenter1[1, :], thinned_array_h_r[len(thinned_array_h_r)-third_r:]), select_columns_matrix(detcenter2[1, :], thinned_array_h_r[len(thinned_array_h_r)-third_r:])], c='purple')

                # thinned_array_h_y の前1/3のプロット（darkyellow）
                axs[0, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_h_y[:third_y]), select_columns_matrix(detcenter2[0, :], thinned_array_h_y[:third_y])],
                               [select_columns_matrix(detcenter1[1, :], thinned_array_h_y[:third_y]), select_columns_matrix(detcenter2[1, :], thinned_array_h_y[:third_y])], c='y')

                # thinned_array_h_y の後ろ1/3のプロット（purple）
                axs[0, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_h_y[len(thinned_array_h_y)-third_y:]), select_columns_matrix(detcenter2[0, :], thinned_array_h_y[len(thinned_array_h_y)-third_y:])],
                               [select_columns_matrix(detcenter1[1, :], thinned_array_h_y[len(thinned_array_h_y)-third_y:]), select_columns_matrix(detcenter2[1, :], thinned_array_h_y[len(thinned_array_h_y)-third_y:])], c='#B8860B')

                # thinned_array_h_g の前1/3のプロット（緑）
                axs[0, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_h_g[:third_g]), select_columns_matrix(detcenter2[0, :], thinned_array_h_g[:third_g])],
                               [select_columns_matrix(detcenter1[1, :], thinned_array_h_g[:third_g]), select_columns_matrix(detcenter2[1, :], thinned_array_h_g[:third_g])], c='g')

                # thinned_array_h_g の後ろ1/3のプロット（薄緑）
                axs[0, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_h_g[len(thinned_array_h_g)-third_g:]), select_columns_matrix(detcenter2[0, :], thinned_array_h_g[len(thinned_array_h_g)-third_g:])],
                               [select_columns_matrix(detcenter1[1, :], thinned_array_h_g[len(thinned_array_h_g)-third_g:]), select_columns_matrix(detcenter2[1, :], thinned_array_h_g[len(thinned_array_h_g)-third_g:])], c='lightgreen')
                axs[0,0].plot([input_val, input_val],
                            [np.min(detcenter2[1, :]), np.max(detcenter1[1, :])], color='k')
                axs[0,0].set_title('V aperture 0')
                axs[0,0].set_xlabel('Axial (m)')
                axs[0,0].set_ylabel('Horizontal Position (m)')

                axs[1,1].scatter(detcenter[1, :], detcenter[2, :])
                axs[1,1].scatter(detcenter[1, ::ray_num], detcenter[2, ::ray_num],color='r')
                axs[1,1].scatter(detcenter[1, round((ray_num-1)/2)::ray_num], detcenter[2, round((ray_num-1)/2)::ray_num],color='y')
                axs[1,1].scatter(detcenter[1, ray_num-1::ray_num], detcenter[2, ray_num-1::ray_num],color='g')
                axs[1,1].scatter(detcenter[1, ray_num-1::ray_num-1][:-1], detcenter[2, ray_num-1::ray_num-1][:-1],color='k')
                axs[1,1].scatter(detcenter[1, ::ray_num+1], detcenter[2, ::ray_num+1],color='gray')
                axs[1,1].scatter(mpmath_mean(detcenter[1, ::ray_num]), mpmath_mean(detcenter[2, ::ray_num]),color='r',marker='x',s=100)
                axs[1,1].scatter(mpmath_mean(detcenter[1, round((ray_num-1)/2)::ray_num]), mpmath_mean(detcenter[2, round((ray_num-1)/2)::ray_num]),color='y',marker='x',s=100)
                axs[1,1].scatter(mpmath_mean(detcenter[1, ray_num-1::ray_num]), mpmath_mean(detcenter[2, ray_num-1::ray_num]),color='g',marker='x',s=100)
                axs[1,1].set_title('focus @H aperture 0')
                axs[1,1].set_xlabel('Horizontal (m)')
                axs[1,1].set_ylabel('Vertical (m)')
                axs[1,1].axis('equal')

                # 初期のプロット
                third_r = len(thinned_array_v_r) *2 // 3
                third_y = len(thinned_array_v_y) *2 // 3
                third_g = len(thinned_array_v_g) *2 // 3
                # print('thinned_array_v_r[:third_r]',thinned_array_v_r[:third_r])
                # print('thinned_array_v_r[-third_r:]',thinned_array_v_r[-third_r:])

                # 前1/3のプロット（赤）
                axs[1, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_v_r[:third_r]), select_columns_matrix(detcenter2[0, :], thinned_array_v_r[:third_r])],
                               [select_columns_matrix(detcenter1[2, :], thinned_array_v_r[:third_r]), select_columns_matrix(detcenter2[2, :], thinned_array_v_r[:third_r])], c='r')

                # 後ろ1/3のプロット（ピンク）
                axs[1, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_v_r[len(thinned_array_v_r)-third_r:]), select_columns_matrix(detcenter2[0, :], thinned_array_v_r[len(thinned_array_v_r)-third_r:])],
                               [select_columns_matrix(detcenter1[2, :], thinned_array_v_r[len(thinned_array_v_r)-third_r:]), select_columns_matrix(detcenter2[2, :], thinned_array_v_r[len(thinned_array_v_r)-third_r:])], c='purple')

                # thinned_array_v_y の前1/3のプロット（darkyellow）
                axs[1, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_v_y[:third_y]), select_columns_matrix(detcenter2[0, :], thinned_array_v_y[:third_y])],
                               [select_columns_matrix(detcenter1[2, :], thinned_array_v_y[:third_y]), select_columns_matrix(detcenter2[2, :], thinned_array_v_y[:third_y])], c='y')

                # thinned_array_v_y の後ろ1/3のプロット（purple）
                axs[1, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_v_y[len(thinned_array_v_y)-third_y:]), select_columns_matrix(detcenter2[0, :], thinned_array_v_y[len(thinned_array_v_y)-third_y:])],
                               [select_columns_matrix(detcenter1[2, :], thinned_array_v_y[len(thinned_array_v_y)-third_y:]), select_columns_matrix(detcenter2[2, :], thinned_array_v_y[len(thinned_array_v_y)-third_y:])], c='#B8860B')

                # thinned_array_v_g の前1/3のプロット（緑）
                axs[1, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_v_g[:third_g]), select_columns_matrix(detcenter2[0, :], thinned_array_v_g[:third_g])],
                               [select_columns_matrix(detcenter1[2, :], thinned_array_v_g[:third_g]), select_columns_matrix(detcenter2[2, :], thinned_array_v_g[:third_g])], c='g')

                # thinned_array_v_g の後ろ1/3のプロット（薄緑）
                axs[1, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_v_g[len(thinned_array_v_g)-third_g:]), select_columns_matrix(detcenter2[0, :], thinned_array_v_g[len(thinned_array_v_g)-third_g:])],
                               [select_columns_matrix(detcenter1[2, :], thinned_array_v_g[len(thinned_array_v_g)-third_g:]), select_columns_matrix(detcenter2[2, :], thinned_array_v_g[len(thinned_array_v_g)-third_g:])], c='lightgreen')

                axs[1,0].plot([input_val, input_val],
                            [np.min(detcenter2[2, :]), np.max(detcenter1[2, :])], color='k')
                axs[1,0].set_title('H aperture 0')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Vertical Position (m)')

                cols_total = detcenter.cols

                axs[0,1].scatter(detcenter[1, :], detcenter[2, :])
                axs[0,1].scatter(detcenter[1, :ray_num], detcenter[2, :ray_num],color='r')
                axs[0,1].scatter(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)], detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)],color='y')
                axs[0,1].scatter(detcenter[1, cols_total-ray_num:], detcenter[2, cols_total-ray_num:],color='g')
                axs[0,1].scatter(detcenter[1, ray_num-1::ray_num-1][:-1], detcenter[2, ray_num-1::ray_num-1][:-1],color='k')
                axs[0,1].scatter(detcenter[1, ::ray_num+1], detcenter[2, ::ray_num+1],color='gray')
                axs[0,1].scatter(mpmath_mean(detcenter[1, :ray_num]), mpmath_mean(detcenter[2, :ray_num]),color='r',marker='x',s=100)
                axs[0,1].scatter(mpmath_mean(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]), mpmath_mean(detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]),color='y',marker='x',s=100)
                axs[0,1].scatter(mpmath_mean(detcenter[1, cols_total-ray_num:]), mpmath_mean(detcenter[2, cols_total-ray_num:]),color='g',marker='x',s=100)
                axs[0,1].set_title('focus @V aperture 0')
                axs[0,1].set_xlabel('Horizontal (m)')
                axs[0,1].set_ylabel('Vertical (m)')
                axs[0,1].axis('equal')

                # タイトル用の新しいサイズ計算
                size_v = np.max(detcenter[2,:]) - np.min(detcenter[2,:])
                size_h = np.max(detcenter[1,:]) - np.min(detcenter[1,:])

                # タイトルの更新
                title1 = f'Params 0-1: {nstr(params[0:2],9)}'
                title2 = f'Params 2-7: {nstr(params[2:8],9)}'
                title3 = f'Params 8-13: {nstr(params[8:14],9)}'
                title4 = f'Params 14-19: {nstr(params[14:20],9)}'
                title5 = f'Params 20-25: {nstr(params[20:26],9)}'
                title6 = f'Size V: {size_v}'
                title7 = f'Size H: {size_h}'

                fig.suptitle(f'{title1}\n{title2}\n{title3}\n{title4}\n{title5}\n{title6}\n{title7}', fontsize=12)
                fig.tight_layout(rect=[0, 0.05, 1, 0.95])
                fig.tight_layout(pad=2.0)
                # マウスイベントでクリックした位置のx座標を取得してプロットを更新
                def on_click(event):
                    if event.inaxes == axs[0,0] or event.inaxes == axs[1,0]:  # クリックが左のプロット内で行われたか確認
                        input_val = event.xdata  # x座標を取得
                        coeffs_det[9] = -input_val
                        detcenter = plane_ray_intersection(coeffs_det, reflect4, hmirr_hyp)
                        # 既存の範囲を保持するためにxlimとylimを記録
                        xlim_0_0 = axs[0,0].get_xlim()
                        ylim_0_0 = axs[0,0].get_ylim()
                        xlim_1_0 = axs[1,0].get_xlim()
                        ylim_1_0 = axs[1,0].get_ylim()

                        # input_valを使って再計算（例として新しいプロットを追加）
                        axs[1,1].cla()  # 右側プロットをクリア
                        axs[1,1].scatter(detcenter[1, :], detcenter[2, :])
                        axs[1,1].scatter(detcenter[1, ::ray_num], detcenter[2, ::ray_num],color='r')
                        axs[1,1].scatter(detcenter[1, round((ray_num-1)/2)::ray_num], detcenter[2, round((ray_num-1)/2)::ray_num],color='y')
                        axs[1,1].scatter(detcenter[1, ray_num-1::ray_num], detcenter[2, ray_num-1::ray_num],color='g')
                        axs[1,1].scatter(detcenter[1, ray_num-1::ray_num-1][:-1], detcenter[2, ray_num-1::ray_num-1][:-1],color='k')
                        axs[1,1].scatter(detcenter[1, ::ray_num+1], detcenter[2, ::ray_num+1],color='gray')
                        axs[1,1].scatter(mpmath_mean(detcenter[1, ::ray_num]), mpmath_mean(detcenter[2, ::ray_num]),color='r',marker='x',s=100)
                        axs[1,1].scatter(mpmath_mean(detcenter[1, round((ray_num-1)/2)::ray_num]), mpmath_mean(detcenter[2, round((ray_num-1)/2)::ray_num]),color='y',marker='x',s=100)
                        axs[1,1].scatter(mpmath_mean(detcenter[1, ray_num-1::ray_num]), mpmath_mean(detcenter[2, ray_num-1::ray_num]),color='g',marker='x',s=100)
                        axs[1,1].set_title('focus @H aperture 0')
                        axs[1,1].set_xlabel('Horizontal (m)')
                        axs[1,1].set_ylabel('Vertical (m)')
                        axs[1,1].axis('equal')

                        axs[0,0].cla()  # 左側プロットをクリア
                        # thinned_array_h_r, thinned_array_h_y, thinned_array_h_g の1/3と2/3のインデックス
                        third_r = len(thinned_array_h_r) *2 // 3
                        third_y = len(thinned_array_h_y) *2 // 3
                        third_g = len(thinned_array_h_g) *2 // 3

                        # 前1/3のプロット（赤）
                        axs[0, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_h_r[:third_r]), select_columns_matrix(detcenter2[0, :], thinned_array_h_r[:third_r])],
                                       [select_columns_matrix(detcenter1[1, :], thinned_array_h_r[:third_r]), select_columns_matrix(detcenter2[1, :], thinned_array_h_r[:third_r])], c='r')

                        # 後ろ1/3のプロット（ピンク）
                        axs[0, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_h_r[len(thinned_array_h_r)-third_r:]), select_columns_matrix(detcenter2[0, :], thinned_array_h_r[len(thinned_array_h_r)-third_r:])],
                                       [select_columns_matrix(detcenter1[1, :], thinned_array_h_r[len(thinned_array_h_r)-third_r:]), select_columns_matrix(detcenter2[1, :], thinned_array_h_r[len(thinned_array_h_r)-third_r:])], c='purple')

                        # thinned_array_h_y の前1/3のプロット（darkyellow）
                        axs[0, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_h_y[:third_y]), select_columns_matrix(detcenter2[0, :], thinned_array_h_y[:third_y])],
                                       [select_columns_matrix(detcenter1[1, :], thinned_array_h_y[:third_y]), select_columns_matrix(detcenter2[1, :], thinned_array_h_y[:third_y])], c='y')

                        # thinned_array_h_y の後ろ1/3のプロット（purple）
                        axs[0, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_h_y[len(thinned_array_h_y)-third_y:]), select_columns_matrix(detcenter2[0, :], thinned_array_h_y[len(thinned_array_h_y)-third_y:])],
                                       [select_columns_matrix(detcenter1[1, :], thinned_array_h_y[len(thinned_array_h_y)-third_y:]), select_columns_matrix(detcenter2[1, :], thinned_array_h_y[len(thinned_array_h_y)-third_y:])], c='#B8860B')

                        # thinned_array_h_g の前1/3のプロット（緑）
                        axs[0, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_h_g[:third_g]), select_columns_matrix(detcenter2[0, :], thinned_array_h_g[:third_g])],
                                       [select_columns_matrix(detcenter1[1, :], thinned_array_h_g[:third_g]), select_columns_matrix(detcenter2[1, :], thinned_array_h_g[:third_g])], c='g')

                        # thinned_array_h_g の後ろ1/3のプロット（薄緑）
                        axs[0, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_h_g[len(thinned_array_h_g)-third_g:]), select_columns_matrix(detcenter2[0, :], thinned_array_h_g[len(thinned_array_h_g)-third_g:])],
                                       [select_columns_matrix(detcenter1[1, :], thinned_array_h_g[len(thinned_array_h_g)-third_g:]), select_columns_matrix(detcenter2[1, :], thinned_array_h_g[len(thinned_array_h_g)-third_g:])], c='lightgreen')
                        axs[0,0].plot([input_val, input_val],
                                    [np.min(detcenter2[1, :]), np.max(detcenter1[1, :])], color='k')
                        axs[0,0].set_title('V aperture 0')
                        axs[0,0].set_xlabel('Axial (m)')
                        axs[0,0].set_ylabel('Horizontal Position (m)')

                        axs[1,0].cla()  # 左側プロットをクリア
                        # thinned_array_v_r, thinned_array_v_y, thinned_array_v_g の1/3と2/3のインデックス
                        third_r = len(thinned_array_v_r) *2 // 3
                        third_y = len(thinned_array_v_y) *2 // 3
                        third_g = len(thinned_array_v_g) *2 // 3
                        # print('thinned_array_v_r[:third_r]',thinned_array_v_r[:third_r])
                        # print('thinned_array_v_r[-third_r:]',thinned_array_v_r[-third_r:])

                        # 前1/3のプロット（赤）
                        axs[1, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_v_r[:third_r]), select_columns_matrix(detcenter2[0, :], thinned_array_v_r[:third_r])],
                                       [select_columns_matrix(detcenter1[2, :], thinned_array_v_r[:third_r]), select_columns_matrix(detcenter2[2, :], thinned_array_v_r[:third_r])], c='r')

                        # 後ろ1/3のプロット（ピンク）
                        axs[1, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_v_r[len(thinned_array_v_r)-third_r:]), select_columns_matrix(detcenter2[0, :], thinned_array_v_r[len(thinned_array_v_r)-third_r:])],
                                       [select_columns_matrix(detcenter1[2, :], thinned_array_v_r[len(thinned_array_v_r)-third_r:]), select_columns_matrix(detcenter2[2, :], thinned_array_v_r[len(thinned_array_v_r)-third_r:])], c='purple')

                        # thinned_array_v_y の前1/3のプロット（darkyellow）
                        axs[1, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_v_y[:third_y]), select_columns_matrix(detcenter2[0, :], thinned_array_v_y[:third_y])],
                                       [select_columns_matrix(detcenter1[2, :], thinned_array_v_y[:third_y]), select_columns_matrix(detcenter2[2, :], thinned_array_v_y[:third_y])], c='y')

                        # thinned_array_v_y の後ろ1/3のプロット（purple）
                        axs[1, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_v_y[len(thinned_array_v_y)-third_y:]), select_columns_matrix(detcenter2[0, :], thinned_array_v_y[len(thinned_array_v_y)-third_y:])],
                                       [select_columns_matrix(detcenter1[2, :], thinned_array_v_y[len(thinned_array_v_y)-third_y:]), select_columns_matrix(detcenter2[2, :], thinned_array_v_y[len(thinned_array_v_y)-third_y:])], c='#B8860B')

                        # thinned_array_v_g の前1/3のプロット（緑）
                        axs[1, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_v_g[:third_g]), select_columns_matrix(detcenter2[0, :], thinned_array_v_g[:third_g])],
                                       [select_columns_matrix(detcenter1[2, :], thinned_array_v_g[:third_g]), select_columns_matrix(detcenter2[2, :], thinned_array_v_g[:third_g])], c='g')

                        # thinned_array_v_g の後ろ1/3のプロット（薄緑）
                        axs[1, 0].plot([select_columns_matrix(detcenter1[0, :], thinned_array_v_g[len(thinned_array_v_g)-third_g:]), select_columns_matrix(detcenter2[0, :], thinned_array_v_g[len(thinned_array_v_g)-third_g:])],
                                       [select_columns_matrix(detcenter1[2, :], thinned_array_v_g[len(thinned_array_v_g)-third_g:]), select_columns_matrix(detcenter2[2, :], thinned_array_v_g[len(thinned_array_v_g)-third_g:])], c='lightgreen')

                        axs[1,0].plot([input_val, input_val],
                                    [np.min(detcenter2[2, :]), np.max(detcenter1[2, :])], color='k')
                        axs[1,0].set_title('H aperture 0')
                        axs[1,0].set_xlabel('Axial (m)')
                        axs[1,0].set_ylabel('Vertical Position (m)')

                        axs[0,1].cla()  # 右側プロットをクリア
                        axs[0,1].scatter(detcenter[1, :], detcenter[2, :])
                        axs[0,1].scatter(detcenter[1, :ray_num], detcenter[2, :ray_num],color='r')
                        axs[0,1].scatter(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)], detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)],color='y')
                        axs[0,1].scatter(detcenter[1, cols_total-ray_num:], detcenter[2, cols_total-ray_num:],color='g')
                        axs[0,1].scatter(detcenter[1, ray_num-1::ray_num-1][:-1], detcenter[2, ray_num-1::ray_num-1][:-1],color='k')
                        axs[0,1].scatter(detcenter[1, ::ray_num+1], detcenter[2, ::ray_num+1],color='gray')
                        axs[0,1].scatter(mpmath_mean(detcenter[1, :ray_num]), mpmath_mean(detcenter[2, :ray_num]),color='r',marker='x',s=100)
                        axs[0,1].scatter(mpmath_mean(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]), mpmath_mean(detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]),color='y',marker='x',s=100)
                        axs[0,1].scatter(mpmath_mean(detcenter[1, cols_total-ray_num:]), mpmath_mean(detcenter[2, cols_total-ray_num:]),color='g',marker='x',s=100)
                        axs[0,1].set_title('focus @V aperture 0')
                        axs[0,1].set_xlabel('Horizontal (m)')
                        axs[0,1].set_ylabel('Vertical (m)')
                        axs[0,1].axis('equal')

                        # axs[0,2].scatter(input_val,mpmath_mean(detcenter[1, :ray_num]),color='r')
                        # axs[0,2].scatter(input_val,mpmath_mean(detcenter[1, round((ray_num**2)/2) : round((ray_num**2 + ray_num*2)/2)]),color='y')
                        # axs[0,2].scatter(input_val,mpmath_mean(detcenter[1, -ray_num:-1]),color='g')

                        # axs[1,2].scatter(input_val,mpmath_mean(detcenter[2, ::ray_num]),color='r')
                        # axs[1,2].scatter(input_val,mpmath_mean(detcenter[2, round(ray_num/2)-1::ray_num]),color='y')
                        # axs[1,2].scatter(input_val,mpmath_mean(detcenter[2, ray_num-1::ray_num]),color='g')
                        # axs[1,3].scatter(input_val,mpmath_mean(detcenter[1, ::ray_num]),color='r')
                        # axs[1,3].scatter(input_val,mpmath_mean(detcenter[1, round(ray_num/2)-1::ray_num]),color='y')
                        # axs[1,3].scatter(input_val,mpmath_mean(detcenter[1, ray_num-1::ray_num]),color='g')

                        axs[0,0].set_xlim(xlim_0_0)  # クリア後に元の範囲を再設定
                        axs[0,0].set_ylim(ylim_0_0)

                        axs[1,0].set_xlim(xlim_1_0)
                        axs[1,0].set_ylim(ylim_1_0)

                        # タイトル用の新しいサイズ計算
                        size_v = np.max(detcenter[2,:]) - np.min(detcenter[2,:])
                        size_h = np.max(detcenter[1,:]) - np.min(detcenter[1,:])

                        std_obl1 = np.sqrt(np.std(detcenter[1, ray_num-1::ray_num-1][:-1])**2 + np.std(detcenter[2, ray_num-1::ray_num-1][:-1])**2)
                        std_obl2 = np.sqrt(np.std(detcenter[1, ::ray_num+1])**2 + np.std(detcenter[2, ::ray_num+1])**2)
                        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        print('std_obl1',std_obl1)
                        print('std_obl2',std_obl2)
                        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        # タイトルの更新
                        title1 = f'Params 0-1: {params[0:2]}'
                        title2 = f'Params 2-7: {params[2:8]}'
                        title3 = f'Params 8-13: {params[8:14]}'
                        title4 = f'Params 14-19: {params[14:20]}'
                        title5 = f'Params 20-25: {params[20:26]}'
                        title6 = f'Size V: {size_v}'
                        title7 = f'Size H: {size_h}'

                        fig.suptitle(f'{title1}\n{title2}\n{title3}\n{title4}\n{title5}\n{title6}\n{title7}', fontsize=12)

                        fig.canvas.draw_idle()  # 再描画
                # イベントリスナーを設定
                fig.canvas.mpl_connect('button_press_event', on_click)
                plt.savefig(os.path.join(directory_name, 'multipleAroundFocus.png'), dpi=300)
                plt.show()


if __name__ == "__main__":

    initial_params = np.array( [-5.74837795e-03, -2.87529633e-03, 0.00000000e+00, 0.00000000e+00,
                                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                0.00000000e+00, -3.44496117e-05, 1.30000000e-04, 0.00000000e+00,
                                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                0.00000000e+00, -3.44496117e-05, 1.30000000e-04, 0.00000000e+00,
                                0.00000000e+00, 0.00000000e+00])
    ### mpf
    initial_params = [mpf(str(param)) for param in initial_params]
    # plot_result_debug(initial_params,'ray')
    plot_result_debug(initial_params,'ray_wave')