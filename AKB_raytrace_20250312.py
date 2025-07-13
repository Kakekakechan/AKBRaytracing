import os
os.environ['NUMBA_NUM_THREADS'] = '10'
os.environ['OMP_NUM_THREADS'] = '10'
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import numba
from numba import njit, prange
import matplotlib.cm as cm
from numpy import abs, sin, cos,tan, arcsin,arccos,arctan, sqrt, pi
from mpmath import mp
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
import mpmath
from mpmath import mp, mpf, matrix
import json
mp.dps = 20
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
option_AKB = False
option_wolter_3_1 = False
option_wolter_3_3_tandem = False
option_HighNA = True
global LowNAratio
LowNAratio = 1.
defocusForWave = 1e-3

global option_mpmath
option_mpmath = False
global option_set
option_set = False

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
if option_AKB:
    if option_wolter_3_1:
        directory_name = f"output_{timestamp}_wolter_3_1"
    elif option_wolter_3_3_tandem:
        directory_name = f"output_{timestamp}_wolter_3_3_tandem"
    else:
        directory_name = f"output_{timestamp}_wolter_3_3_alternating"
else:
    directory_name = f"output_{timestamp}_KB"

# 新しいフォルダを作成
os.makedirs(directory_name, exist_ok=True)

def calculate_wavefront_error_v2(defocus_positions, path_length_distribution, angle_distribution, focal_plane_positions, wavelength):
    """
    入力データを基に波面誤差を計算する関数

    Parameters:
        defocus_positions (numpy.ndarray): デフォーカス位置 (3 x N)
        path_length_distribution (numpy.ndarray): 光路長分布 (1 x N)
        angle_distribution (numpy.ndarray): 光線の角度分布 (3 x N)
        focal_plane_positions (numpy.ndarray): 焦点面での位置 (3 x N)
        wavelength (float): 使用する光の波長

    Returns:
        wavefront_error (numpy.ndarray): 波面誤差分布 (N)
        rms_error (float): RMS波面誤差
    """
    N = defocus_positions.shape[1]

    # 光路長誤差 (実際の光路長 - 理想的な光路長の平均値)
    opd_error = path_length_distribution - np.mean(path_length_distribution)

    # 焦点面の位置誤差
    focal_plane_error = np.linalg.norm(focal_plane_positions - np.mean(focal_plane_positions, axis=1, keepdims=True), axis=0)

    # 角度誤差を光路長変化に換算 (角度偏差を法線方向と比較してスケーリング)
    ideal_normals = -defocus_positions / np.linalg.norm(defocus_positions, axis=0, keepdims=True)  # 理想波面の法線
    angle_dot_products = np.einsum('ij,ij->j', angle_distribution, ideal_normals)
    angle_error = 1 - angle_dot_products  # 法線とのずれ

    # 光路長誤差としての角度補正
    angle_opd_correction = wavelength * angle_error / (2 * np.pi)

    # 総波面誤差の計算
    wavefront_error = opd_error + focal_plane_error + angle_opd_correction

    # RMS波面誤差
    rms_error = np.sqrt(np.mean(wavefront_error**2))

    return wavefront_error, rms_error
def crop(start,end,mabiki):
    # 元の配列（例として 0 から end までの範囲）
    original_array = list(range(end + 1))
    # 間引き率を設定 (例: 2 つおきに取り出す場合)
    skip_rate = mabiki
    # 範囲内の値を間引く
    thinned_array = original_array[start:end:skip_rate]
    return thinned_array
def plot_ray_sideview(place,defocussize,mabiki,ray,point,ray_num):
    coeffs_det = np.zeros(10)
    coeffs_det[6] = 1.
    coeffs_det[7] = 0.
    coeffs_det[8] = 0.
    coeffs_det[9] = -(place) + defocussize
    detcenter1 = plane_ray_intersection(coeffs_det, ray, point)

    coeffs_det = np.zeros(10)
    coeffs_det[6] = 1.
    coeffs_det[7] = 0.
    coeffs_det[8] = 0.
    coeffs_det[9] = -(place) - defocussize
    detcenter2 = plane_ray_intersection(coeffs_det, ray, point)

    # 範囲内の値を間引く
    thinned_array1 = crop(0, ray_num,mabiki)
    # 範囲内の値を間引く
    thinned_array2 = crop(round((ray_num**2)/2), round((ray_num**2 + ray_num*2)/2), mabiki)
    # 範囲内の値を間引く
    thinned_array3 = crop(ray_num**2 - ray_num, ray_num**2, mabiki)

    # 範囲内の値を間引く
    original_array = list(range(len(detcenter1[0,:])))
    first_thinned_array = original_array[::ray_num]
    thinned_array1_v = first_thinned_array[::mabiki]
    # 範囲内の値を間引く
    original_array = list(range(len(detcenter1[0,:])))
    first_thinned_array = original_array[round((ray_num-1)/2)::ray_num]
    thinned_array2_v = first_thinned_array[::mabiki]
    # 範囲内の値を間引く
    original_array = list(range(len(detcenter1[0,:])))
    first_thinned_array = original_array[ray_num-1::ray_num]
    thinned_array3_v = first_thinned_array[::mabiki]

    # plt.figure()
    fig,axs = plt.subplots(2, 1,sharex=True)

    # for i in range(detcenter1.shape[1]):
    #     if i % mabiki == 0:
    #         plt.plot([detcenter1[0,i],detcenter2[0,i]],[detcenter1[1,i],detcenter2[1,i]])

    # axs[0].plot([detcenter1[0,thinned_array1],detcenter2[0,thinned_array1]],[detcenter1[1,thinned_array1],detcenter2[1,thinned_array1]],'r')
    # axs[0].plot([detcenter1[0,thinned_array3],detcenter2[0,thinned_array3]],[detcenter1[1,thinned_array3],detcenter2[1,thinned_array3]],'g')
    # axs[0].plot([detcenter1[0,thinned_array2],detcenter2[0,thinned_array2]],[detcenter1[1,thinned_array2],detcenter2[1,thinned_array2]],'y')

    axs[0].plot([detcenter1[0,:],detcenter2[0,:]],[detcenter1[1,:],detcenter2[1,:]],'k')


    axs[1].plot([detcenter1[0,thinned_array1_v],detcenter2[0,thinned_array1_v]],[detcenter1[2,thinned_array1_v],detcenter2[2,thinned_array1_v]],'r')
    axs[1].plot([detcenter1[0,thinned_array3_v],detcenter2[0,thinned_array3_v]],[detcenter1[2,thinned_array3_v],detcenter2[2,thinned_array3_v]],'g')
    axs[1].plot([detcenter1[0,thinned_array2_v],detcenter2[0,thinned_array2_v]],[detcenter1[2,thinned_array2_v],detcenter2[2,thinned_array2_v]],'y')
    # plt.plot([detcenter1[0,thinned_array2],detcenter2[0,thinned_array2]],[detcenter1[1,thinned_array2],detcenter2[1,thinned_array2]],'y')
    # plt.plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
    axs[0].set_title('Virtual focus of 2nd mirror')
    axs[0].set_xlabel('Axial (m)')
    axs[0].set_ylabel('Horizontal Position (m)')
    axs[1].set_title('Virtual focus of 2nd mirror')
    axs[1].set_xlabel('Axial (m)')
    axs[1].set_ylabel('Vertical Position (m)')
    plt.savefig('virtualSource.png', dpi=300)
    plt.show()

    # plt.figure()
    # for i in range(detcenter1.shape[1]):
    #     if i % mabiki == 0:
    #         plt.plot([detcenter1[0,i],detcenter2[0,i]],[detcenter1[2,i],detcenter2[2,i]])
    # # plt.plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
    # plt.title('Virtual focus of 2nd mirror')
    # plt.xlabel('Axial (m)')
    # plt.ylabel('Vertical Position (m)')
    # plt.show()
    return
def rotate_to_ray_basis(points, ray_direction):
    """
    点群を光線ベクトルに基づいて回転させる。

    Parameters:
    points (numpy array): 変換したい点群 (3 x n)
    ray_direction (numpy array): 光線の方向ベクトル [dx, dy, dz]

    Returns:
    new_points (numpy array): 新しい座標系での点群 (3 x n)
    """
    # 点群の平均中心を計算
    centroid = np.mean(points, axis=1)

    # 光線方向を新しい x 軸として正規化
    x_axis = normalize_vector(ray_direction)

    # 任意のベクトルと光線方向との外積で新しい y 軸を定義
    arbitrary_vector = np.array([0, 0, 1]) if not np.allclose(x_axis, [0, 0, 1]) else np.array([0, 1, 0])
    y_axis = normalize_vector(np.cross(x_axis, arbitrary_vector))

    # 新しい z 軸を定義
    z_axis = np.cross(x_axis, y_axis)

    # 回転行列の作成
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # 点群を平均中心を原点とする座標系に移動
    translated_points = points - centroid[:, np.newaxis]

    # 回転行列を使って新しい座標系に変換
    new_points = np.dot(rotation_matrix, translated_points) + centroid[:, np.newaxis]

    return new_points
def Ell_define(l1, inc, l2):
    sita1 = arctan(l2*sin(2.*inc) / (l1+l2*cos(2.*inc)))

    a_ell = (l1+l2)/2.
    b_ell = sqrt(l1 * l2 * sin(inc) ** 2)
    # b_ell = sqrt((l1+l2-l1*cos(sita1)-l2*cos(sita1-2.*inc)) * (l1+l2+l1*cos(sita1)+l2*cos(sita1-2*inc))) /2.

    sita3 = arcsin(l1*sin(sita1) / l2)

    return a_ell, b_ell, sita1, sita3

def calcEll_Yvalue(a, b, x):
    return sqrt(b**2. - (b*(x - sqrt(a**2. - b**2.)) /a)**2.)

def calcNA(a_h,b_h,sita1h,l1h,mlen_h):
    s2f_h = sqrt(a_h**2. - b_h**2.) * 2.    ###horizontal source-focus

    xh_s = l1h * cos(sita1h) - mlen_h/2.    ###Hmirror start
    xh_e = l1h * cos(sita1h) + mlen_h/2.    ###Hmirror end
    yh_s = calcEll_Yvalue(a_h,b_h,xh_s)
    yh_e = calcEll_Yvalue(a_h,b_h,xh_e)
    accept_h = abs(yh_e - yh_s)
    NA_h = sin(abs(arctan(yh_e/(s2f_h-xh_e)) - arctan(yh_s/(s2f_h-xh_s))))/2.
    return NA_h

def KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v , gapf=0.):

    a_h, b_h, sita1h, sita3h = Ell_define(l1h, inc_h, l2h)
    s2f_h = sqrt(a_h**2. - b_h**2.) * 2.    ###horizontal source-focus

    xh_s = l1h * cos(sita1h) - mlen_h/2.    ###Hmirror start
    xh_e = l1h * cos(sita1h) + mlen_h/2.    ###Hmirror end
    yh_s = calcEll_Yvalue(a_h,b_h,xh_s)
    yh_e = calcEll_Yvalue(a_h,b_h,xh_e)
    accept_h = abs(yh_e - yh_s)
    NA_h = sin(abs(arctan(yh_e/(s2f_h-xh_e)) - arctan(yh_s/(s2f_h-xh_s))))/2.

    l1v = l1h + (l2h - wd_v - mlen_v/2.)         ###1st guess of l1v
    l2v = wd_v + mlen_v/2.                  ###1st guess of l2v

    while True:
        a_v, b_v, sita1v, sita3v = Ell_define(l1v, inc_v, l2v)
        s2f_v = sqrt(a_v**2. - b_v**2.) * 2.    ###horizontal source-focus

        diff = s2f_h - s2f_v
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

#                 self.u[i] += u_back.u[j] * np.exp(-ii * k * dist) / np.sqrt(dist)
# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 端の値の密度を高く、中央の値の密度を低くする
def angle_between_2vector(ray1,ray2):
    # 列ごとの結果を保存するリスト
    angles_between_rad = []  # ray1とray2のなす角度
    angles_yx_rad = []       # y/x の角度
    angles_zx_rad = []       # z/x の角度

    # 各列に対して処理を行う
    for i in range(ray1.shape[1]):
        # 各列ごとにベクトルを取得
        v1 = np.array([ray1[0, i], ray1[1, i], ray1[2, i]])
        v2 = np.array([ray2[0, i], ray2[1, i], ray2[2, i]])

        # ray1 と ray2 のなす角度を計算
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cross_product = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross_product)
        angle_between_rad = np.arctan2(norm_cross, dot_product)
        angles_between_rad.append(angle_between_rad)

        # y/x と z/x の角度を計算 (ray1 に対して)
        if ray1[0, i] != 0:  # x が 0 でない場合のみ計算
            angle_yx_rad = np.arctan2(ray1[1, i], ray1[0, i])  # y/x の角度
            angle_zx_rad = np.arctan2(ray1[2, i], ray1[0, i])  # z/x の角度
        else:
            angle_yx_rad = np.nan  # 計算不可の場合は NaN
            angle_zx_rad = np.nan  # 計算不可の場合は NaN

        angles_yx_rad.append(angle_yx_rad)
        angles_zx_rad.append(angle_zx_rad)
    return angles_between_rad, angles_yx_rad, angles_zx_rad
def create_non_uniform_distribution(start, end, num_points):
    linear_space = np.linspace(-6, 6, num_points)  # -6から6の範囲でサンプルを生成
    sigmoid_space = sigmoid(linear_space)
    # 0から1にスケーリング
    scaled_distribution = (sigmoid_space - sigmoid_space.min()) / (sigmoid_space.max() - sigmoid_space.min())
    # 指定範囲にスケーリング
    final_distribution = start + (end - start) * scaled_distribution
    return final_distribution

def CalcDataPitch(points_array,h_num,v_num):
    matrix_y = points_array[1,:].reshape(h_num,v_num)
    matrix_z = points_array[2,:].reshape(h_num,v_num)
    dif_y0 = np.diff(matrix_y,axis=0)
    dif_y1 = np.diff(matrix_y,axis=1)
    dif_z0 = np.diff(matrix_z,axis=0)
    dif_z1 = np.diff(matrix_z,axis=1)
    print('dif_y0 data pitch',np.mean(dif_y0))
    print('dif_y1',np.mean(dif_y1))
    print('dif_z0',np.mean(dif_z0))
    print('dif_z1 data pitch',np.mean(dif_z1))
    return


def mirr_ray_intersection(coeffs, ray, source, negative=False):
    if option_mpmath:
        print("mpmathを使用して計算中...")
        coeffs = np_to_mpmath_matrix(coeffs)
        ray = np_to_mpmath_matrix(ray)
        source = np_to_mpmath_matrix(source)

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
                point[0, col] = mpmath.nan
                point[1, col] = mpmath.nan
                point[2, col] = mpmath.nan
                continue

            sqrtD = sqrt(D)
            if negative:
                t = (-B - sqrtD) / (2*A)
            else:
                t = (-B + sqrtD) / (2*A)

            point[0, col] = t*l + p
            point[1, col] = t*m + q
            point[2, col] = t*n + r
        # mpmath matrixをnumpy arrayに変換
        point = mpmath_matrix_to_numpy(point)
        return point
    else:
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
    if option_mpmath:
        ray = np_to_mpmath_matrix(ray)
        N = np_to_mpmath_matrix(N)
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

        phai = mpmath_normalize_vector(phai)
        phai = mpmath_matrix_to_numpy(phai)  # mpmath matrixをnumpy arrayに変換
        return phai
    else:
        l, m, n = ray
        nx, ny, nz = N

        A = l * nx + m * ny + n * nz
        phai = ray - 2 * A * N

        phai = normalize_vector(phai)
        return phai
def mpmath_normalize_vector(vector):
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

def normalize_vector(vector):
    norm = np.linalg.norm(vector, axis=0)
    return vector / norm if np.all(norm != 0) else vector

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


def generate_grid_on_mirror_with_normal(mirror_coeffs, points, num_divisions_H,num_divisions_V):
    """
    平均法線を使ったグリッド生成

    Parameters:
    - mirror_coeffs: ミラーの係数（10個）
    - points: ミラーの交点4点（3×4行列）
    - num_divisions: グリッド分割数（縦・横の分割数）

    Returns:
    - grid_points: グリッド点の座標（3次元）
    """
    # 平均法線を計算
    # ベクトル計算
    p1, p2, p3, p4 = points.T
    v1 = p2 - p1
    v2 = p4 - p1

    # 法線を外積で計算
    normal1 = np.cross(v1, v2)
    normal1 = normal1 / normal1[0]
    normal1 = normal1 / np.linalg.norm(normal1)  # 正規化

    v1 = p2 - p1
    v2 = p3 - p1

    normal2 = np.cross(v1, v2)
    normal2 = normal2 / normal2[0]
    normal2 = normal2 / np.linalg.norm(normal2)  # 正規化

    normal = normal1 + normal2
    normal = normal / np.linalg.norm(normal)  # 正規化

    # ミラー上のグリッドを平面で生成
    u = np.linspace(0, 1, num_divisions_H)
    v = np.linspace(0, 1, num_divisions_V)
    grid_points = []

    for j in v:
        for i in u:
            point = (1 - i) * (1 - j) * p1 + i * (1 - j) * p2 + i * j * p3 + (1 - i) * j * p4
            grid_points.append(point)

    grid_points = np.array(grid_points).T  # 3×N行列

    # グリッドをミラー上に投影
    projected_grid = []
    for grid_point in grid_points.T:
        source = grid_point.copy()  # 各点を光源とする
        # ray = normal  # 平均法線を方向ベクトルとして使用
        ray = np.array([-1,0,0])
        intersection = mirr_ray_intersection(mirror_coeffs, ray, source.reshape(3, 1))
        projected_grid.append(intersection[:, 0])

    return np.array(projected_grid).T

def mirr_ray_RoC(coeffs,ray,points):
    a, b, c, d, e, f, g, h, i, j = coeffs
    l, m, n = ray
    v = tang_vector(coeffs,ray,points)
    Hessian = np.array([
        [2 * a, d, e],
        [d, 2 * b, f],
        [e, f, 2 * c]
    ])  # ヘッセ行列
    # 各点に対して曲率を計算
    K = np.einsum('ij,jk,ik->i', v.T, Hessian, v.T)  # v^T H v の計算
    return K

def norm_vector(coeffs, point):
    a, b, c, d, e, f, g, h, i, j = coeffs
    x, y, z = point

    N = np.zeros_like(point)
    N[0, :] = 2 * a * x + d * y + e * z + g
    N[1, :] = 2 * b * y + d * x + f * z + h
    N[2, :] = 2 * c * z + e * x + f * y + i

    N = normalize_vector(N)
    return N

# 接線ベクトルを計算する関数
def tang_vector(coeffs, ray, points):
    N = norm_vector(coeffs, points)  # 法線ベクトルを取得
    dot_product = np.einsum('ij,ij->j', ray, N)  # rayと法線の内積
    T = ray - dot_product * N  # 法線方向の成分を引く
    return normalize_vector(T)

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

def rotate_x(coeffs, theta, center):
    coeffs = shift_x(coeffs, -center[0])
    coeffs = shift_y(coeffs, -center[1])
    coeffs = shift_z(coeffs, -center[2])

    a, b, c, d, e, f, g, h, i, j = coeffs
    Cos = np.cos(theta)
    Sin = np.sin(theta)

    a1 = a
    b1 = b * Cos**2 + c * Sin**2 - f * Sin * Cos
    c1 = b * Sin**2 + c * Cos**2 + f * Sin * Cos
    d1 = d * Cos - e * Sin
    e1 = d * Sin + e * Cos
    f1 = b * np.sin(2 * theta) - c * np.sin(2 * theta) + f * np.cos(2 * theta)
    g1 = g
    h1 = h * Cos - i * Sin
    i1 = h * Sin + i * Cos
    j1 = j

    coeffs = [a1, b1, c1, d1, e1, f1, g1, h1, i1, j1]
    coeffs = shift_x(coeffs, center[0])
    coeffs = shift_y(coeffs, center[1])
    coeffs = shift_z(coeffs, center[2])
    return coeffs

def rotate_y(coeffs, theta, center):
    coeffs = shift_x(coeffs, -center[0])
    coeffs = shift_y(coeffs, -center[1])
    coeffs = shift_z(coeffs, -center[2])

    a, b, c, d, e, f, g, h, i, j = coeffs
    Cos = np.cos(theta)
    Sin = np.sin(theta)

    a1 = a * Cos**2 + c * Sin**2 + e * Sin * Cos
    b1 = b
    c1 = a * Sin**2 + c * Cos**2 - e * Sin * Cos
    d1 = d * Cos + f * Sin
    e1 = -a * np.sin(2 * theta) + c * np.sin(2 * theta) + e * np.cos(2 * theta)
    f1 = -d * Sin + f * Cos
    g1 = g * Cos + i * Sin
    h1 = h
    i1 = i * Cos - g * Sin
    j1 = j

    coeffs = [a1, b1, c1, d1, e1, f1, g1, h1, i1, j1]
    coeffs = shift_x(coeffs, center[0])
    coeffs = shift_y(coeffs, center[1])
    coeffs = shift_z(coeffs, center[2])
    return coeffs

def rotate_z(coeffs, theta, center):
    coeffs = shift_x(coeffs, -center[0])
    coeffs = shift_y(coeffs, -center[1])
    coeffs = shift_z(coeffs, -center[2])

    a, b, c, d, e, f, g, h, i, j = coeffs
    Cos = np.cos(theta)
    Sin = np.sin(theta)

    a1 = a * Cos**2 + b * Sin**2 - d * Sin * Cos
    b1 = b * Cos**2 + a * Sin**2 + d * Sin * Cos
    c1 = c
    d1 = a * np.sin(2 * theta) - b * np.sin(2 * theta) + d * np.cos(2 * theta)
    e1 = e * Cos - f * Sin
    f1 = e * Sin + f * Cos
    g1 = g * Cos - h * Sin
    h1 = g * Sin + h * Cos
    i1 = i
    j1 = j

    coeffs = [a1, b1, c1, d1, e1, f1, g1, h1, i1, j1]
    coeffs = shift_x(coeffs, center[0])
    coeffs = shift_y(coeffs, center[1])
    coeffs = shift_z(coeffs, center[2])
    return coeffs

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

    return [a1, b1, c1, d1, e1, f1, g1, h1, i1, j1]

# 2. 目的関数の定義（最適化用）
def objective(R_flat, coeffs, target_coeffs):
    # 1次元配列を3x3の行列に変換
    R = R_flat.reshape(3, 3)

    # 計算した係数
    computed_coeffs = compute_coeffs(R, coeffs)

    # 目標係数との二乗誤差の合計を返す
    residual = np.sum((np.array(computed_coeffs) - np.array(target_coeffs))**2)
    return residual

# 3. 最適化関数
def optimize_rotation(coeffs, target_coeffs):
    # 回転行列 R を最適化
    initial_R = np.eye(3).flatten()  # 初期回転行列（単位行列）

    # 最適化
    result = minimize(objective, initial_R, args=(coeffs, target_coeffs), method='BFGS')

    # 最適化後の回転行列
    R_opt = result.x.reshape(3, 3)
    return R_opt
def rotate_axis(axis, R):
    """
    与えられた回転行列 R を使って、軸(axis)を回転させた結果を返す。
    axis: 回転軸のベクトル
    R: 3x3の回転行列
    """
    return np.dot(R, axis)
def compute_rotation_angle(axis, rotated_axis):
    """
    回転前後の軸ベクトルから回転角度を計算する。
    axis: 回転前の軸ベクトル
    rotated_axis: 回転後の軸ベクトル
    """
    cos_theta = np.dot(axis, rotated_axis)  # 軸ベクトル同士の内積
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 数値誤差による範囲外の値をクリップ
    return np.arccos(cos_theta)
def disp_axis_rotation(R, axis_x,axis_y,axis_z):
    # 各軸を回転させた結果を求める
    rotated_axis_x = rotate_axis(axis_x, R)
    rotated_axis_y = rotate_axis(axis_y, R)
    rotated_axis_z = rotate_axis(axis_z, R)

    # 回転角を計算
    angle_x = compute_rotation_angle(axis_x, rotated_axis_x)
    angle_y = compute_rotation_angle(axis_y, rotated_axis_y)
    angle_z = compute_rotation_angle(axis_z, rotated_axis_z)

    # 結果を出力
    print("x軸の回転角 (ラジアン):", angle_x)
    print("y軸の回転角 (ラジアン):", angle_y)
    print("z軸の回転角 (ラジアン):", angle_z)
    print("x軸の回転角 (度):", np.degrees(angle_x))
    print("y軸の回転角 (度):", np.degrees(angle_y))
    print("z軸の回転角 (度):", np.degrees(angle_z))
    return

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

def point_rotate_x(point, theta, center):
    x, y, z = point
    x, y, z = x - center[0], y - center[1], z - center[2]

    Cos, Sin = np.cos(theta), np.sin(theta)
    x1, y1, z1 = x, y * Cos + z * Sin, z * Cos - y * Sin

    x1, y1, z1 = x1 + center[0], y1 + center[1], z1 + center[3]
    return np.array([x1, y1, z1])

def point_rotate_y(point, theta, center):
    x, y, z = point
    x, y, z = x - center[0], y - center[1], z - center[2]

    Cos, Sin = np.cos(theta), np.sin(theta)
    x1, y1, z1 = x * Cos - z * Sin, y, x * Sin + z * Cos

    x1, y1, z1 = x1 + center[0], y1 + center[1], z1 + center[2]
    return np.array([x1, y1, z1])

def point_rotate_z(point, theta, center):
    x, y, z = point
    x, y, z = x - center[0], y - center[1], z - center[2]

    Cos, Sin = np.cos(theta), np.sin(theta)
    x1, y1, z1 = x * Cos + y * Sin, y * Cos - x * Sin, z

    x1, y1, z1 = x1 + center[0], y1 + center[1], z1 + center[2]
    return np.array([x1, y1, z1])

def rotate_vectors(vector, theta_y, theta_z):
    # 回転行列 (y軸中心)
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])

    # 回転行列 (z軸中心)
    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])

    # z軸 → y軸の順に回転させる
    reflect_rotated = R_y @ (R_z @ vector)

    return reflect_rotated

def rotate_points(points, focus_apprx, theta_y, theta_z):
    # focus_apprx を原点に移動
    points_shifted = points - focus_apprx[:, np.newaxis]

    # y軸とz軸で回転
    points_rotated = rotate_vectors(points_shifted, theta_y, theta_z)

    # 元の位置に戻す
    points_rotated += focus_apprx[:, np.newaxis]

    return points_rotated


def calc_Y_hyp(a, b, x):
    y = np.sqrt(-b ** 2 + (b * (x - np.sqrt(a ** 2 + b ** 2)) / a) ** 2)
    return y

def rotate_image(points):

    # 法線ベクトルを求める（例として2つのベクトルを使って計算）
    vector1 = points[:, 1] - points[:, 0]
    vector2 = points[:, -1] - points[:, 0]
    normal_vector = np.cross(vector1, vector2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 正規化

    # x軸との外積による回転軸
    x_axis = np.array([1, 0, 0])
    rotation_axis = np.cross(normal_vector, x_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # 正規化

    # 回転角度
    cos_theta = np.dot(normal_vector, x_axis)
    theta = np.arccos(cos_theta)

    # 回転行列の生成（ロドリゲスの回転公式を使用）
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    # 座標データに回転行列を適用
    rotated_points = np.dot(R, points)

    return rotated_points
#####################################
def rotatematrix(rotation_matrix, axis_x, axis_y, axis_z):
    axis_x_new = np.dot(rotation_matrix, axis_x)
    axis_y_new = np.dot(rotation_matrix, axis_y)
    axis_z_new = np.dot(rotation_matrix, axis_z)
    return axis_x_new, axis_y_new, axis_z_new
from numpy.fft import fft2, fftshift, fftfreq
def fresnel_integral(phi, grid_x, grid_y, lambda_, z, x_out_range, y_out_range, dx_out, dy_out):
    """
    フレネル積分により、指定の焦点面での波面分布を計算する。

    Parameters
    ----------
    phi : 2D ndarray
        光路長分布 [m]。NaN は無効領域（開口外）を示す。
    grid_x, grid_y : 2D ndarray
        入力波面の物理座標グリッド [m]。
    lambda_ : float
        波長 [m]
    z : float
        伝搬距離 [m]
    x_out_range : tuple (xmin, xmax)
        出力面 x の範囲 [m]
    y_out_range : tuple (ymin, ymax)
        出力面 y の範囲 [m]
    dx_out, dy_out : float
        出力面ピッチ [m]

    Returns
    -------
    psf : 2D ndarray
        出力面の強度（|U|²）
    x_out, y_out : 1D ndarray
        出力面の座標軸
    """
    k = 2 * np.pi / lambda_

    # 有効な点のみ使用（NaNマスク）
    mask = ~np.isnan(phi)
    phi_valid = phi[mask]
    x_in = grid_x[mask]
    y_in = grid_y[mask]

    # 入力波面
    U_in = np.exp(1j * k * phi_valid) * np.exp(-1j * k / (2 * z) * (x_in**2 + y_in**2))

    # 出力面座標系
    x_out = np.arange(x_out_range[0], x_out_range[1], dx_out)
    y_out = np.arange(y_out_range[0], y_out_range[1], dy_out)
    X_out, Y_out = np.meshgrid(x_out, y_out)
    print('X_out.shape', X_out.shape)

    # Fresnel積分（直接積分）
    psf = np.zeros_like(X_out, dtype=np.complex128)

    for idx in range(len(x_in)):
        r = np.sqrt( (X_out - x_in[idx])**2 + (Y_out - y_in[idx])**2 + z**2 )
        phase = np.exp(-1j * k * r)
        psf += U_in[idx] * phase/ r

    # # 定数前係数
    # psf *= (1j / (lambda_ * z)) * dx_out * dy_out
    # PSFの強度
    psf = np.abs(psf)**2
    psf /= np.nanmax(psf)  # 正規化
    return psf, x_out, y_out


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
        width = np.linalg.norm(pts_src_ordered[0] - pts_src_ordered[1])
        height = np.linalg.norm(pts_src_ordered[0] - pts_src_ordered[2])
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


def np_to_mpmath_matrix(arr):
    """
    numpy.ndarray または list を mpmath.matrix に変換
    """
    # numpy配列なら一度リスト化
    if isinstance(arr, np.ndarray):
        arr_list = arr.tolist()
    else:
        arr_list = arr

    # 次元数確認
    if isinstance(arr_list[0], list):  # 2次元配列
        rows = len(arr_list)
        cols = len(arr_list[0])
        mat = mpmath.matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                mat[i, j] = mpmath.mpf(arr_list[i][j])
    else:  # 1次元ベクトル
        mat = mpmath.matrix(len(arr_list), 1)
        for i in range(len(arr_list)):
            mat[i] = mpmath.mpf(arr_list[i])

    return mat

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

        # # Mirror parameters EUV
        # a_hyp_v = np.float64(72.96002945938)
        # b_hyp_v = np.float64(0.134829747201017)
        # a_ell_v = np.float64(0.442)
        # b_ell_v = np.float64(0.0607128830733533)
        # length_hyp_v = np.float64(0.115)
        # length_ell_v = np.float64(0.229790269646258)
        # theta1_v = np.float64(4.73536529533549E-05)
        #
        # a_hyp_h = np.float64(73.018730871665)
        # b_hyp_h = np.float64(0.0970536727319812)
        # a_ell_h = np.float64(0.38125)
        # b_ell_h = np.float64(0.0397791317992322)
        # length_hyp_h = np.float64(0.25)
        # length_ell_h = np.float64(0.0653872838592807)
        # theta1_h = np.float64(5.6880350884129E-05)

        if False:  # 初期設定 2025/05
            ### 初期設定
            # ### 3型
            # a_hyp_v  =   np.float64(72.985)
            # b_hyp_v  =   np.float64(0.250539088918923)
            # a_ell_v  =   np.float64(0.0833)
            # b_ell_v  =   np.float64(0.0211103685691917)
            # length_hyp_v  =   np.float64(0.0345)
            # length_ell_v  =   np.float64(0.0530861729741616)
            # theta1_v  =   np.float64(4.88527559126772E-05)
            # theta2_v  =   np.float64(0.12)
            # theta3_v  =   np.float64(0.240048852755913)
            # theta4_v  =   np.float64(0.256751946233106)
            # theta5_v  =   np.float64(-0.2734550397103)
            # phai_hyp_v  =   -0.120048852755913
            # phai_ell_v  =   0.00826201408949603
            # F_eff_v  =   0.0254480654760698
            # Mag_v  =   5741.36267534975
            # NA_v  =   0.0820569070569736
            # Aperture_v  =   0.01348071947248
            # F0_F2_v  =   146.13202134964

            ### 3型 Setting1
            a_hyp_v  =     72.985
            b_hyp_v  =     0.25261675784047
            a_ell_v  =     0.0933
            b_ell_v  =     0.0236745564714402
            length_hyp_v  =     0.0345
            length_ell_v  =     0.0594385752478948
            theta1_v  =     4.92519127861222E-05
            theta2_v  =     0.121
            theta3_v  =     0.242049251912786
            theta4_v  =     0.256975971148682
            theta5_v  =     -0.271902690384578
            phai_hyp_v  =     -0.121049251912786
            phai_ell_v  =     0.00989215579059824
            F_eff_v  =     0.0258133633376474
            Mag_v  =     5660.84906496756
            NA_v  =     0.0816159686218418
            Aperture_v  =     0.0151067293572333
            F0_F2_v  =     146.151367076927

            # ### 3型 Setting2
            # a_hyp_v  =     72.985
            # b_hyp_v  =     0.25261675784047
            # a_ell_v  =     0.104
            # b_ell_v  =     0.0264132202381823
            # length_hyp_v  =     0.0345
            # length_ell_v  =     0.0662598019783809
            # theta1_v  =     4.92519127861222E-05
            # theta2_v  =     0.121
            # theta3_v  =     0.242049251912786
            # theta4_v  =     0.25722413024862
            # theta5_v  =     -0.272399008584454
            # phai_hyp_v  =     -0.121049251912786
            # phai_ell_v  =     0.00969021318203535
            # F_eff_v  =     0.0257675833177302
            # Mag_v  =     5671.71103747572
            # NA_v  =     0.0817632660571617
            # Aperture_v  =     0.0168562943249794
            # F0_F2_v  =     146.172054295563

            # ### 3型 Setting3
            # a_hyp_v  =     72.98
            # b_hyp_v  =     0.238853696163023
            # a_ell_v  =     0.124
            # b_ell_v  =     0.0277912976743237
            # length_hyp_v  =   0.0485
            # length_ell_v  =   0.0842274770752208
            # theta1_v  =     5.39073028703234E-05
            # theta2_v  =     0.099
            # theta3_v  =     0.19805390730287
            # theta4_v  =     0.227982557779111
            # theta5_v  =     -0.257911208255352
            # phai_hyp_v  =     -0.0990539073028703
            # phai_ell_v  =     -0.00341217625956458
            # F_eff_v  =     0.0298718112105973
            # Mag_v  =     4893.3290306965
            # NA_v  =     0.082028477826814
            # Aperture_v  =     0.0190364833176426
            # F0_F2_v  =     146.202472807512

            # ### 3型 Setting4
            # a_hyp_v  =     72.975
            # b_hyp_v  =     0.258979496230151
            # a_ell_v  =     0.146
            # b_ell_v  =     0.0324589749420265
            # length_hyp_v  =     0.06
            # length_ell_v  =     0.0977147009051387
            # theta1_v  =     6.53721558428766E-05
            # theta2_v  =     0.096
            # theta3_v  =     0.192065372155843
            # theta4_v  =     0.226829703253149
            # theta5_v  =     -0.261594034350456
            # phai_hyp_v  =     -0.0960653721558429
            # phai_ell_v  =     -0.00847283029554443
            # F_eff_v  =     0.0356980368015059
            # Mag_v  =     4095.46088149609
            # NA_v  =     0.0821759344796697
            # Aperture_v  =     0.0219750172281408
            # F0_F2_v  =     146.235611303576

            ### 3型 Setting5
            a_hyp_v  =     72.965
            b_hyp_v  =     0.279401469315277
            a_ell_v  =     0.192
            b_ell_v  =     0.0398166258808826
            length_hyp_v =     0.088
            length_ell_v =     0.136096444142887
            theta1_v  =     8.35253813269807E-05
            theta2_v  =     0.08751
            theta3_v  =     0.175103525381327
            theta4_v  =     0.212097563755144
            theta5_v  =     -0.249091602128961
            phai_hyp_v  =     -0.087593525381327
            phai_ell_v  =     -0.00917354104873317
            F_eff_v  =     0.0480249101249228
            Mag_v  =     3045.47570759381
            NA_v  =     0.0825004980438573
            Aperture_v  =     0.0286497879308899
            F0_F2_v  =     146.306722054954

            # ### 3型 setting6
            # a_hyp_v  =     72.968
            # b_hyp_v  =     0.233129192075476
            # a_ell_v  =     0.319
            # b_ell_v  =     0.0541992812373669
            # length_hyp_v =     0.1
            # length_ell_v =     0.304544748239094
            # theta1_v  =     6.66973897662189E-05
            # theta2_v  =     0.07634
            # theta3_v  =     0.152746697389766
            # theta4_v  =     0.171779248790688
            # theta5_v  =     -0.19081180019161
            # phai_hyp_v  =     -0.0764066973897662
            # phai_ell_v  =     0.0164772127640783
            # F_eff_v  =     0.0505905421593695
            # Mag_v  =     2896.09227230842
            # NA_v  =     0.0824908283406065
            # Aperture_v  =     0.0520575638907512
            # F0_F2_v  =     146.565468741803






            ### 1型 setting1
            a_ell_h  =     np.float64(73.07505)
            b_ell_h  =     np.float64(0.420125678460643)
            a_hyp_h  =     np.float64(0.0072)
            b_hyp_h  =     np.float64(0.00369271404399535)
            length_hyp_h  =   np.float64(0.01008239076)
            length_ell_h  =   np.float64(0.026)
            theta1_h  =     np.float64(0.000109393749605896)
            theta2_h  =    np.float64(0.15)
            theta3_h  =     np.float64(0.299890606250394)
            theta4_h  =     np.float64(0.193862238668175)
            theta5_h  =     np.float64(0.736736912319257)
            phai_ell_h  =     -0.149890606250394
            phai_hyp_h  =     -0.509211953516557
            F_eff_h  =     0.0176204701590166
            Mag_h  =     8292.28047404524
            NA_h  =     0.0815169628107427
            Aperture_h  =     0.00388539144431358
            F0_F2_h  =     146.131501113269

            # ### 1型 setting3

            # a_ell_h  =     73.11055
            # b_ell_h  =     0.420227752780288
            # a_hyp_h  =     0.0072
            # b_hyp_h  =     0.00369271489837695
            # length_hyp_h  =   np.float64(0.01008238804)
            # length_ell_h  =   np.float64(0.026)
            # theta1_h  =     0.000109340630839105
            # theta2_h  =     0.15
            # theta3_h  =     0.299890659369161
            # theta4_h  =     0.193862292274329
            # theta5_h  =     0.736737068149589
            # phai_ell_h  =     -0.149890659369161
            # phai_hyp_h  =     -0.50921205866987
            # F_eff_h  =     0.0176204666209969
            # Mag_h  =     8296.31154439727
            # NA_h  =     0.0815169632942278
            # Aperture_h  =     0.00388539144431358
            # F0_F2_h  =     146.202501112065

            ### 1型 setting5
            a_ell_h  =     73.168374088788
            b_ell_h  =     0.459086041847713
            a_hyp_h  =     0.01245
            b_hyp_h  =     0.00540542675557545
            length_hyp_h  =   np.float64(0.01838060690)
            length_ell_h  =   np.float64(0.048)
            theta1_h  =     0.000153614692187686
            theta2_h  =     0.127523
            theta3_h  =     0.254892385307812
            theta4_h  =     0.170397163734124
            theta5_h  =     0.649584711266688
            phai_ell_h  =     -0.127369385307812
            phai_hyp_h  =     -0.441783192142318
            F_eff_h  =     0.0295837508341208
            Mag_h  =     4944.5095424719
            NA_h  =     0.0825004524526383
            Aperture_h  =     0.00610452713441869
            F0_F2_h  =     146.306722052255

            # ### 1型 setting6
            # a_ell_h  =     73.232511916154
            # b_ell_h  =     0.350911894865679
            # a_hyp_h  =     0.0193
            # b_hyp_h  =     0.00537029744119799
            # length_hyp_h  =   np.float64(0.02885466994)
            # length_ell_h  =   np.float64(0.100)
            # theta1_h  =     0.000147560113856669
            # theta2_h  =     0.07772
            # theta3_h  =     0.155292439886143
            # theta4_h  =     0.125914072797402
            # theta5_h  =     0.470329741849645
            # phai_ell_h  =     -0.0775724398861433
            # phai_hyp_h  =     -0.298520690144625
            # F_eff_h  =     0.0424877695542136
            # Mag_h  =     3445.24529428841
            # NA_h  =     0.0825007922962757
            # Aperture_h  =     0.00776417803336937
            # F0_F2_h  =     146.423275891019
        if False:  # 初期設定 2025/06/01
            ### 3型 Setting7
            a_hyp_v  =   np.float64(72.984)
            b_hyp_v  =   np.float64(0.194270774041555)
            a_ell_v  =   np.float64(0.1485)
            b_ell_v  =   np.float64(0.0291345266237788)
            hyp_length_v  =   np.float64(0.045)
            ell_length_v  =   np.float64(0.122095241889387)
            theta1_v  =   np.float64(3.92478218204609E-05)
            theta2_v  =   np.float64(0.09)
            theta3_v  =   np.float64(0.18003924782182)
            theta4_v  =   np.float64(0.198305036930121)
            theta5_v  =   np.float64(-0.216570826038422)
            phai_hyp_v  =   np.float64(-0.0900392478218205)
            phai_ell_v  =   np.float64(0.0129508135388115)
            F_eff_v  =   np.float64(0.0260854154971923)
            Mag_v  =   np.float64(5605.95477814208)
            NA_v  =   np.float64(0.0820087922773097)
            Aperture_v  =   np.float64(0.0240537234876166)
            F0_F2_v  =   np.float64(146.259745061804)
            ### 3型 Setting10
            a_hyp_v  =   72.9815
            b_hyp_v  =   0.160243976396275
            a_ell_v  =   0.223
            b_ell_v  =   0.0349789930209892
            hyp_length_v  =   0.06
            ell_length_v  =   0.225864538803214
            theta1_v  =   3.4870458354867E-05
            theta2_v  =   0.069
            theta3_v  =   0.138034870458355
            theta4_v  =   0.158867314327645
            theta5_v  =   -0.179699758196935
            phai_hyp_v  =   -0.0690348704583549
            phai_ell_v  =   0.0159600556958157
            F_eff_v  =   0.0280975824952918
            Mag_v  =   5209.54902203498
            NA_v  =   0.0816297367263195
            Aperture_v  =   0.0357317443925052
            F0_F2_v  =   146.40383099239

            ### 3型 setting9
            a_hyp_v  =   72.9755
            b_hyp_v  =   0.184407632092118
            a_ell_v  =   0.2195
            b_ell_v  =   0.0409556196334448
            hyp_length_v  =   0.06
            ell_length_v  =   0.145864752520461
            theta1_v  =   4.61835569510556E-05
            theta2_v  =   0.069
            theta3_v  =   0.138046183556951
            theta4_v  =   0.196465341435049
            theta5_v  =   -0.254884499313148
            phai_hyp_v  =   -0.0690461835569511
            phai_ell_v  =   -0.0315332449715707
            F_eff_v  =   0.0259422040898016
            Mag_v  =   5641.64917695336
            NA_v  =   0.0817613057404778
            Aperture_v  =   0.0284733680128275
            F0_F2_v  =   146.382756555675




            ### 1型 setting7
            a_ell_h  =   np.float64(73.13915)
            b_ell_h  =   np.float64(0.420309969273809)
            a_hyp_h  =   np.float64(0.0072)
            b_hyp_h  =   np.float64(0.0036927155860925)
            hyp_length_h  =   np.float64(0.01008238585)
            ell_length_h  =   np.float64(0.026)
            theta1_h  =   np.float64(0.000109297874071033)
            theta2_h  =   np.float64(0.15)
            theta3_h  =   np.float64(0.299890702125929)
            theta4_h  =   np.float64(0.193862335421518)
            theta5_h  =   np.float64(0.736737193581758)
            phai_ell_h  =   np.float64(-0.149890702125929)
            phai_hyp_h  =   np.float64(-0.509212143310027)
            F_eff_h  =   np.float64(0.0176204637731416)
            Mag_h  =   np.float64(8299.55911093751)
            NA_h  =   np.float64(0.0815169636832092)
            Aperture_h  =   np.float64(0.00388539144431358)
            F0_F2_h  =   np.float64(146.259701111095)

            ### 1型 setting5
            a_ell_h  =     73.168374088788
            b_ell_h  =     0.459086041847713
            a_hyp_h  =     0.01245
            b_hyp_h  =     0.00540542675557545
            hyp_length_h  =   np.float64(0.01838060690)
            ell_length_h  =   np.float64(0.048)
            theta1_h  =     0.000153614692187686
            theta2_h  =     0.127523
            theta3_h  =     0.254892385307812
            theta4_h  =     0.170397163734124
            theta5_h  =     0.649584711266688
            phai_ell_h  =     -0.127369385307812
            phai_hyp_h  =     -0.441783192142318
            F_eff_h  =     0.0295837508341208
            Mag_h  =     4944.5095424719
            NA_h  =     0.0825004524526383
            Aperture_h  =     0.00610452713441869
            F0_F2_h  =     146.306722052255

            length_hyp_v = hyp_length_v
            length_ell_v = ell_length_v
            length_hyp_h = hyp_length_h
            length_ell_h = ell_length_h

        if True:  # 初期設定 2025/06/12
            ### 3型 Setting11
            a_hyp_v  =   np.float64(72.981)
            b_hyp_v  =   np.float64(0.274955741326603)
            a_ell_v  =   np.float64(0.107)
            b_ell_v  =   np.float64(0.0270059974473355)
            hyp_length_v  =   np.float64(0.043)
            ell_length_v  =   np.float64(0.0668954744537463)
            theta1_v  =   np.float64(6.03651016038744E-05)
            theta2_v  =   np.float64(0.117)
            theta3_v  =   np.float64(0.234060365101604)
            theta4_v  =   np.float64(0.256079738847749)
            theta5_v  =   np.float64(-0.278099112593894)
            phai_hyp_v  =   np.float64(-0.117060365101604)
            phai_ell_v  =   np.float64(0.0024924654464218)
            F_eff_v  =   np.float64(0.0308994364264663)
            Mag_v  =   np.float64(4729.51047311538)
            NA_v  =   np.float64(0.0818997791446706)
            Aperture_v  =   np.float64(0.0169439604135074)
            F0_F2_v  =   np.float64(146.170107628762)

            ### 3型 Setting12
            a_hyp_v  =   np.float64(72.9825)
            b_hyp_v  =   np.float64(0.263879113520857)
            a_ell_v  =   np.float64(0.1175)
            b_ell_v  =   np.float64(0.0283168369674688)
            hyp_length_v  =   np.float64(0.043)
            ell_length_v  =   np.float64(0.0809220387326922)
            theta1_v  =   np.float64(5.55983241203018E-05)
            theta2_v  =   np.float64(0.117)
            theta3_v  =   np.float64(0.23405559832412)
            theta4_v  =   np.float64(0.243572583671924)
            theta5_v  =   np.float64(-0.253089569019727)
            phai_hyp_v  =   np.float64(-0.11705559832412)
            phai_ell_v  =   np.float64(0.0172109421954116)
            F_eff_v  =   np.float64(0.0314202608037988)
            Mag_v  =   np.float64(4651.85850857562)
            NA_v  =   np.float64(0.0819227398493535)
            Aperture_v  =   np.float64(0.0195160723325698)
            F0_F2_v  =   np.float64(146.19402782262)




            ### 1型 setting11
            a_ell_h  =   np.float64(73.1076714403445)
            b_ell_h  =   np.float64(0.517019631143022)
            a_hyp_h  =   np.float64(0.0077)
            b_hyp_h  =   np.float64(0.00432051448679384)
            hyp_length_h  =   np.float64(0.01380360633)
            ell_length_h  =   np.float64(0.030)
            theta1_h  =   np.float64(0.000145746388538841)
            theta2_h  =   np.float64(0.17)
            theta3_h  =   np.float64(0.339854253611461)
            theta4_h  =   np.float64(0.182330449161024)
            theta5_h  =   np.float64(0.757889356272919)
            phai_ell_h  =   np.float64(-0.169854253611461)
            phai_hyp_h  =   np.float64(-0.540136320213965)
            F_eff_h  =   np.float64(0.0225095323759206)
            Mag_h  =   np.float64(6493.76076981313)
            NA_h  =   np.float64(0.0819088461351015)
            Aperture_h  =   np.float64(0.00507547047200988)
            F0_F2_h  =   np.float64(146.194027821967)




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
        # print('omega_v',omega_v)
        # omega_v = omega_v/4
        # #
        omega_v = 0.

        org_hyp_v = np.sqrt(a_hyp_v**2 + b_hyp_v**2)
        org_hyp_h = np.sqrt(a_hyp_h**2 + b_hyp_h**2)

        org_ell_v = np.sqrt(a_ell_v**2 - b_ell_v**2)
        org_ell_h = np.sqrt(a_ell_h**2 - b_ell_h**2)

        # astig_v = (org_hyp_v - org_hyp_h)/2
        # astig_v_ = (org_hyp_v - org_hyp_h)/2*np.linspace(0,4,10)
        n = 20
        # param_ = np.linspace(-1,1,n)
        std_v = np.full(n, np.nan)
        std_h = np.full(n, np.nan)


        # astig_v = astig_v_[j]
        # astig_v = 0.5*-0.16626315789473686
        # astig_v = 0.5
        # print(astig_v_)

        # Input parameters
        ray_num = 53
        # defocus = 0.2*0.03157894736842107
        # defocus = 0
        # print(defocus)
        # astig_v = 0
        # pitch_hyp_h = 0
        # roll_hyp_h = 0
        # yaw_hyp_h = 0
        # pitch_ell_h = 0
        # roll_ell_h = 0
        # yaw_ell_h = 0
        # pitch_hyp_v = 0
        # roll_hyp_v = 0
        # yaw_hyp_v = 0.1*param_[j]
        # pitch_ell_v = 0
        # roll_ell_v = 0
        # yaw_ell_v = 0.1*param_[j]
        # option_2mirror = True
        # Input parameters
        ray_num_H = 53
        ray_num_V = 53
        ray_num = 53
        if option == 'ray':
            ray_num_H = 53
            ray_num_V = 53
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
        c_v = np.zeros(10)
        c_v[0] = 1 / a_hyp_v**2
        c_v[2] = -1 / b_hyp_v**2
        c_v[9] = -1.
        org_v = np.sqrt(a_hyp_v**2 + b_hyp_v**2)
        c_v = shift_x(c_v, org_v)

        center_v = mirr_ray_intersection(c_v, np.array([[np.cos(theta1_v)], [0.], [np.sin(theta1_v)]]), np.array([[0.], [0.], [0.]]))
        if not np.isreal(center_v).all():
            return np.inf
        x1_v = center_v[0, 0] - length_hyp_v / 2
        x2_v = center_v[0, 0] + length_hyp_v / 2
        # y1_v = np.sqrt((-1 + (x1_v / a_hyp_v)**2) * b_hyp_v**2)
        # y2_v = np.sqrt((-1 + (x2_v / a_hyp_v)**2) * b_hyp_v**2)
        y1_v = calc_Y_hyp(a_hyp_v, b_hyp_v, x1_v)
        y2_v = calc_Y_hyp(a_hyp_v, b_hyp_v, x2_v)
        # print(np.arctan(y1_v/x1_v))
        # print(np.arctan(y2_v/x2_v))
        accept_v = np.abs(y2_v - y1_v)
        l1v = np.linalg.norm(c_v)

        ### 1 ell
        c_h = np.zeros(10)
        c_h[0] = 1 / a_ell_h**2
        c_h[1] = 1 / b_ell_h**2
        c_h[9] = -1.
        org_h = np.sqrt(a_ell_h**2 - b_ell_h**2)
        c_h = shift_x(c_h, org_h)
        center_h = mirr_ray_intersection(c_h, np.array([[np.cos(theta1_h)], [np.sin(theta1_h)], [0.]]), np.array([[0.], [0.], [0.]]))
        if not np.isreal(center_h).all():
            return np.inf
        x1_h = center_h[0, 0] - length_ell_h / 2
        x2_h = center_h[0, 0] + length_ell_h / 2
        y1_h = calcEll_Yvalue(a_ell_h, b_ell_h, x1_h)
        y2_h = calcEll_Yvalue(a_ell_h, b_ell_h, x2_h)
        # print(np.arctan(y1_h/x1_h))
        # print(np.arctan(y2_h/x2_h))
        accept_h = np.abs(y2_h - y1_h)
        l1h = np.linalg.norm(c_h)

        if False:
            ratio_here = 0.25
            xw_v = x2_v - x1_v
            yw_v = y2_v - y1_v
            x1_v = x1_v + xw_v*ratio_here
            y1_v = y1_v + yw_v*ratio_here
            x2_v = x2_v - xw_v*ratio_here
            y2_v = y2_v - yw_v*ratio_here

            xw_h = x2_h - x1_h
            yw_h = y2_h - y1_h
            x1_h = x1_h + xw_h*ratio_here
            y1_h = y1_h + yw_h*ratio_here
            x2_h = x2_h - xw_h*ratio_here
            y2_h = y2_h - yw_h*ratio_here


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
        axis_x = np.array([1., 0., 0.])
        axis_y = np.array([0., 1., 0.])
        axis_z = np.array([0., 0., 1.])
        coeffs_hyp_v = np.zeros(10)
        coeffs_hyp_v[0] = 1 / a_hyp_v**2
        coeffs_hyp_v[2] = -1 / b_hyp_v**2
        coeffs_hyp_v[9] = -1.
        coeffs_hyp_v = shift_x(coeffs_hyp_v, org_hyp_v)
        if option_axial:
            # coeffs_hyp_v = rotate_y(coeffs_hyp_v, theta1_v, [0, 0, 0])
            coeffs_hyp_v, rotation_matrix = rotate_general_axis(coeffs_hyp_v, axis_y, theta1_v, [0, 0, 0])
            axis_x, axis_y, axis_z = rotatematrix(rotation_matrix, axis_x, axis_y, axis_z)

        if option_alignment and option_axial:
            bufray = np.zeros((3, 3))
            ### 4隅の光線
            # theta_cntr_h = (np.arctan(y2_h / x2_h) + np.arctan(y1_h / x1_h))/2.
            theta_cntr_v = (np.arctan(y2_v / x2_v) + np.arctan(y1_v / x1_v))/2.
            def print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h):
                l2_h = (4*a_hyp_h**2 + (org_hyp_h*2)**2 -4*a_hyp_h*(org_hyp_h*2)*np.cos(theta1_h))/(4*org_hyp_h - 4*a_hyp_h)
                # l2_v = (4*a_hyp_v**2 + (org_hyp_v*2)**2 -4*a_hyp_v*(org_hyp_v*2)*np.cos(theta1_v))/(4*org_hyp_v - 4*a_hyp_v)

                l1_h = 2*a_hyp_h +l2_h
                # l1_v = 2*a_hyp_v +l2_v

                theta2_h = np.arcsin(org_hyp_h*2*np.sin(theta1_h)/l2_h)/2
                # theta2_v = np.arcsin(org_hyp_v*2*np.sin(theta1_v)/l2_v)/2

                theta3_h = np.arcsin(l1_h*np.sin(theta1_h)/l2_h)
                # theta3_v = np.arcsin(l1_v*np.sin(theta1_v)/l2_v)

                l4_h = ((org_ell_h)**2 - 2*org_ell_h*a_ell_h*np.cos(theta3_h) + a_ell_h**2)/(a_ell_h - org_ell_h*np.cos(theta3_h))
                # l4_v = ((org_ell_v)**2 - 2*org_ell_v*a_ell_v*np.cos(theta3_v) + a_ell_v**2)/(a_ell_v - org_ell_v*np.cos(theta3_v))

                l3_h = 2*a_ell_h - l2_h - l4_h
                # l3_v = 2*a_ell_v - l2_v - l4_v

                theta5_h = np.arcsin((2*a_ell_h - l4_h)*np.sin(theta3_h)/l4_h)
                # theta5_v = np.arcsin((2*a_ell_v - l4_v)*np.sin(theta3_v)/l4_v)

                theta4_h = (theta5_h+theta3_h)/2.
                # theta4_v = (theta5_v+theta3_v)/2.
                # print('theta4_v incidence ell',theta4_v)
                theta4_h = np.arcsin(2*org_ell_h*np.sin(theta3_h)/l4_h)/2
                # theta4_v = np.arcsin(2*org_ell_v*np.sin(theta3_v)/l4_v)/2
                return theta2_h, theta3_h, theta4_h, theta5_h, l1_h, l2_h, l3_h, l4_h
            # theta3_h, theta5_h = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h)
            # theta3_h1, theta5_h1 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y1_h / x1_h))
            # theta3_h2, theta5_h2 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y2_h / x2_h))
            theta2_v, theta3_v, theta4_v, theta5_v, l1_v, l2_v, l3_v, l4_v = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,theta1_v)
            theta2_v1, theta3_v1, theta4_v1, theta5_v1, l1_v1, l2_v1, l3_v1, l4_v1 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y1_v / x1_v))
            theta2_v2, theta3_v2, theta4_v2, theta5_v2, l1_v2, l2_v2, l3_v2, l4_v2 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y2_v / x2_v))
            omega_v = (theta5_v1 + theta5_v2 + np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v))/2
            # omegah1 = (theta3_h1 + theta3_h2 - np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))/2
            # omegav2 = (np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v) + theta5_v1 + theta5_v2)/2
            # omega_v = omega_v*0.6
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
                print('theta1_v1',np.arctan(y1_v / x1_v))
                print('theta1_v2',np.arctan(y2_v / x2_v))
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
                # print('na_h',np.sin((theta5_h1 - theta5_h2))/2.)
                print('na_v',np.sin((theta5_v1 - theta5_v2))/2.)
                # print('div_h',(theta5_h1 - theta5_h2))
                print('div_v',(theta5_v1 - theta5_v2))
                print('div0_h',np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))
                print('div0_v',np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))
                print('===== ===== =====')



            # theta_source_1_h = np.arctan(y1_h / x1_h) - theta_cntr_h
            # theta_source_2_h = np.arctan(y2_h / x2_h) - theta_cntr_h
            theta_source_1_v = np.arctan(y1_v / x1_v) - theta_cntr_v
            theta_source_2_v = np.arctan(y2_v / x2_v) - theta_cntr_v

            bufray[0, 0] = 1.

            bufray[0, 1] = 1.
            # bufray[1, 1] = np.tan(theta_source_1_h)
            bufray[2, 1] = np.tan(theta_source_1_v)

            bufray[0, 2] = 1.
            # bufray[1, 2] = np.tan(theta_source_2_h)
            bufray[2, 2] = np.tan(theta_source_2_v)
            #
            # bufray[0, 3] = 1.
            # bufray[1, 3] = np.tan(theta_source_2_h)
            # bufray[2, 3] = np.tan(theta_source_1_v)
            #
            # bufray[0, 4] = 1.
            # bufray[1, 4] = np.tan(theta_source_2_h)
            # bufray[2, 4] = np.tan(theta_source_2_v)

            # source = np.zeros((3, 5))

            # bufray = np.zeros((3, 2))
            # bufray[0, 0] = 1.
            # bufray[1, 0] = np.tan(theta1_h)
            # bufray[2, 0] = np.tan(theta1_v)
            # bufray[0, 1] = 1.
            # bufray[1, 1] = np.tan(theta1_h)
            # bufray[2, 1] = np.tan(theta1_v)
            source = np.zeros((3, 3))

        else:
            bufray = np.zeros((3, 2))
            if option_axial:
                bufray[0, :] = 1.
                bufray[1, :] = np.tan(theta1_h)
                bufray[2, :] = np.tan(theta1_v)
            # def print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h):
            #     l2_h = (4*a_hyp_h**2 + (org_hyp_h*2)**2 -4*a_hyp_h*(org_hyp_h*2)*np.cos(theta1_h))/(4*org_hyp_h - 4*a_hyp_h)
            #     # l2_v = (4*a_hyp_v**2 + (org_hyp_v*2)**2 -4*a_hyp_v*(org_hyp_v*2)*np.cos(theta1_v))/(4*org_hyp_v - 4*a_hyp_v)
            #
            #     l1_h = 2*a_hyp_h +l2_h
            #     # l1_v = 2*a_hyp_v +l2_v
            #
            #     theta2_h = np.arcsin(org_hyp_h*2*np.sin(theta1_h)/l2_h)/2
            #     # theta2_v = np.arcsin(org_hyp_v*2*np.sin(theta1_v)/l2_v)/2
            #
            #     theta3_h = np.arcsin(l1_h*np.sin(theta1_h)/l2_h)
            #     # theta3_v = np.arcsin(l1_v*np.sin(theta1_v)/l2_v)
            #
            #     l4_h = ((org_ell_h)**2 - 2*org_ell_h*a_ell_h*np.cos(theta3_h) + a_ell_h**2)/(a_ell_h - org_ell_h*np.cos(theta3_h))
            #     # l4_v = ((org_ell_v)**2 - 2*org_ell_v*a_ell_v*np.cos(theta3_v) + a_ell_v**2)/(a_ell_v - org_ell_v*np.cos(theta3_v))
            #
            #     l3_h = 2*a_ell_h - l2_h - l4_h
            #     # l3_v = 2*a_ell_v - l2_v - l4_v
            #
            #     theta5_h = np.arcsin((2*a_ell_h - l4_h)*np.sin(theta3_h)/l4_h)
            #     # theta5_v = np.arcsin((2*a_ell_v - l4_v)*np.sin(theta3_v)/l4_v)
            #
            #     theta4_h = (theta5_h+theta3_h)/2.
            #     # theta4_v = (theta5_v+theta3_v)/2.
            #     print('theta1',theta1_h)
            #     print('l2',l2_h)
            #     # print('l2_v',l2_v)
            #     print('l1',l1_h)
            #     # print('l1_v',l1_v)
            #     print('theta2 incidence hyp',theta2_h)
            #     # print('theta2_v incidence hyp',theta2_v)
            #     print('theta3',theta3_h)
            #     # print('theta3_v',theta3_v)
            #     print('l3',l3_h)
            #     # print('l3_v',l3_v)
            #     print('l4',l4_h)
            #     # print('l4_v',l4_v)
            #     print('hyp to ell',np.cos(theta3_h)*(l3_h))
            #     # print('hyp to ell v',np.cos(theta3_v)*(l3_v))
            #     print('theta4 incidence ell',theta4_h)
            #     # print('theta4_v incidence ell',theta4_v)
            #     theta4_h = np.arcsin(2*org_ell_h*np.sin(theta3_h)/l4_h)/2
            #     # theta4_v = np.arcsin(2*org_ell_v*np.sin(theta3_v)/l4_v)/2
            #     print('theta4 incidence ell',theta4_h)
            #     # print('theta4_v incidence ell',theta4_v)
            #     print('theta5 focal',theta5_h)
            #     # print('theta5_v focal',theta5_v)
            #     print('width1',l1_h*np.cos(theta1_h))
            #     print('width2',l3_h*np.cos(theta3_h))
            #     print('width3',l4_h*np.cos(theta5_h))
            #     print('')
            #     return theta3_h, theta5_h
            #
            #
            # theta3_h, theta5_h = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h)
            # theta3_h1, theta5_h1 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y1_h / x1_h))
            # theta3_h2, theta5_h2 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y2_h / x2_h))
            # theta3_v, theta5_v = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,theta1_v)
            # theta3_v1, theta5_v1 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y1_v / x1_v))
            # theta3_v2, theta5_v2 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y2_v / x2_v))
            # omegav1 = (theta3_v1 + theta3_v2 - np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))/2
            # omegah1 = (theta3_h1 + theta3_h2 - np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))/2
            # omegav2 = (np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v) + theta5_v1 + theta5_v2)/2
            source = np.zeros((3, 2))


        bufray = normalize_vector(bufray)

        center_hyp_v = mirr_ray_intersection(coeffs_hyp_v, bufray, source)
        if not np.isreal(center_hyp_v).all():
            return np.inf
        bufreflect1 = reflect_ray(bufray, norm_vector(coeffs_hyp_v, center_hyp_v))

        bufreflangle1_y = np.arctan(np.mean(bufreflect1[2, 1:]) / np.mean(bufreflect1[0, 1:]))

        if option == 'ray':
            print('coeffs_hyp_v',coeffs_hyp_v)
            print('center_hyp_v',center_hyp_v)
            print('bufray',bufray)
            print('np.mean(bufreflect1[2, 1:])',np.mean(bufreflect1[2, 1:]))
            print('np.mean(bufreflect1[0, 1:])',np.mean(bufreflect1[0, 1:]))
            print('angle_y 1st to 2nd',bufreflangle1_y)

        # print(center_hyp_v)
        # print(bufreflect1)
        # print(np.arctan(bufreflect1[2, 0] / bufreflect1[0, 0]))
        # print('1st')

        # V ell mirror set (2nd)
        axis2_x = np.array([1., 0., 0.])
        axis2_y = np.array([0., 1., 0.])
        axis2_z = np.array([0., 0., 1.])
        coeffs_ell_v = np.zeros(10)
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
        mean_bufreflect1 = np.mean(bufreflect1[:, 1:],axis = 1)
        if option == 'ray':
            roty_local2 = np.arctan(np.dot(mean_bufreflect1,axis2_z) / np.dot(mean_bufreflect1,axis2_x))
            print('rot localy2',roty_local2)
            print('coeffs_ell_v',coeffs_ell_v)

        center_ell_v = mirr_ray_intersection(coeffs_ell_v, bufreflect1, center_hyp_v)
        if not np.isreal(center_ell_v).all():
            return np.inf

        # if option_alignment:
        #     if not optin_axialrotation:
        #         coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle1_y, np.mean(center_hyp_v[:, 1:],axis=1))
        #     if optin_axialrotation:
        #         if option_rotateLocal:
        #             coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis2_y, -omegav1, np.mean(center_hyp_h[:, 1:],axis=1))
        #             axis2_x, axis2_y, axis2_z = rotatematrix(rotation_matrix, axis2_x, axis2_y, axis2_z)
        #         else:
        #             coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle1_y, np.mean(center_hyp_h[:, 1:],axis=1))
        #         center_hyp_h = mirr_ray_intersection(coeffs_hyp_h, bufreflect1, center_hyp_v)
        bufreflect2 = reflect_ray(bufreflect1, norm_vector(coeffs_ell_v, center_ell_v))
        bufreflangle2_z = np.arctan(np.mean(bufreflect2[1, 1:]) / np.mean(bufreflect2[0, 1:]))
        bufreflangle2_y = np.arctan(np.mean(bufreflect2[2, 1:]) / np.mean(bufreflect2[0, 1:]))

        if option == 'ray':
            print('angle_y 2nd to 3rd',bufreflangle2_y)
            print('angle_z 2nd to 3rd',bufreflangle2_z)
        # print(bufreflect2)
        # print(bufreflangle2_z)

        # print('2nd')
        # print(center_hyp_h)



        # Set H ellipse mirror in the vert set (3rd)
        axis3_x = np.array([1., 0., 0.])
        axis3_y = np.array([0., 1., 0.])
        axis3_z = np.array([0., 0., 1.])
        coeffs_ell_h = np.zeros(10)
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
                coeffs_ell_h = rotate_y(coeffs_ell_h, -bufreflangle2_y, np.mean(center_ell_v[:, 1:],axis=1))
            if optin_axialrotation:
                if option_rotateLocal:
                    ### 正確に言うと　omega_h
                    coeffs_ell_h, rotation_matrix = rotate_general_axis(coeffs_ell_h, axis3_y, omega_v, np.mean(center_ell_v[:, 1:],axis=1))
                    axis3_x, axis3_y, axis3_z = rotatematrix(rotation_matrix, axis3_x, axis3_y, axis3_z)
                else:
                    coeffs_ell_h = rotate_y(coeffs_ell_h, -bufreflangle2_y, center_ell_h[:, 0])
                center_ell_h = mirr_ray_intersection(coeffs_ell_h, bufreflect2, center_ell_v)
        bufreflect3 = reflect_ray(bufreflect2, norm_vector(coeffs_ell_h, center_ell_h))
        bufreflangle3_y = np.arctan(np.mean(bufreflect3[2, 1:]) / np.mean(bufreflect3[0, 1:]))
        bufreflangle3_z = np.arctan(np.mean(bufreflect3[1, 1:]) / np.mean(bufreflect3[0, 1:]))

        if option == 'ray':
            print('angle_y 3rd to 4th',bufreflangle3_y)
            print('angle_z 3rd to 4th',bufreflangle3_z)
        # print(bufreflangle3_y)
        # print('3rd')



        # Set ellipse mirror in the horiz set (4th)
        axis4_x = np.array([1., 0., 0.])
        axis4_y = np.array([0., 1., 0.])
        axis4_z = np.array([0., 0., 1.])
        coeffs_hyp_h = np.zeros(10)
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
                coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle3_y, np.mean(center_hyp_v[:, 1:],axis=1))
            if optin_axialrotation:
                if option_rotateLocal:
                    coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis4_y, omega_v, np.mean(center_ell_v[:, 1:],axis=1))
                    axis4_x, axis4_y, axis4_z = rotatematrix(rotation_matrix, axis4_x, axis4_y, axis4_z)
                else:
                    coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle3_y, np.mean(center_hyp_h[:, 1:],axis=1))
                center_hyp_h = mirr_ray_intersection(coeffs_hyp_h, bufreflect3, center_ell_h,negative=True)
        bufreflect4 = reflect_ray(bufreflect3, norm_vector(coeffs_hyp_h, center_hyp_h))
        bufreflangle4_z = np.arctan(np.mean(bufreflect4[1, 1:]) / np.mean(bufreflect4[0, 1:]))
        bufreflangle4_y = np.arctan(np.mean(bufreflect4[2, 1:]) / np.mean(bufreflect4[0, 1:]))

        s2f_H = -2 * org_hyp_h + 2 * org_ell_h
        s2f_V = 2 * org_hyp_v + 2 * org_ell_v
        # print(s2f_H)
        # print(s2f_V)
        s2f_middle = (s2f_H + s2f_V) / 2
        coeffs_det = np.zeros(10)
        coeffs_det[6] = 1.
        coeffs_det[9] = -(s2f_middle + defocus)

        # if option_axial:
        #     coeffs_det = rotate_y(coeffs_det, theta1_v, [0, 0, 0])
        #     coeffs_det = rotate_y(coeffs_det, bufreflangle1_y, center_hyp_v[:, 0])
        #     coeffs_det = rotate_z(coeffs_det, -theta1_h, [0, 0, 0])

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

        # if pitch_ell_v != 0:
        #     coeffs_ell_v = rotate_y(coeffs_ell_v, pitch_ell_v, center_ell_v[:, 0])
        # if roll_ell_v != 0:
        #     coeffs_ell_v = rotate_x(coeffs_ell_v, roll_ell_v, center_ell_v[:, 0])
        # if yaw_ell_v != 0:
        #     coeffs_ell_v = rotate_z(coeffs_ell_v, yaw_ell_v, center_ell_v[:, 0])
        #
        # if pitch_ell_h != 0:
        #     coeffs_ell_h = rotate_y(coeffs_ell_h, pitch_ell_h, center_ell_h[:, 0])
        # if roll_ell_h != 0:
        #     coeffs_ell_h = rotate_x(coeffs_ell_h, roll_ell_h, center_ell_h[:, 0])
        # if yaw_ell_h != 0:
        #     coeffs_ell_h = rotate_z(coeffs_ell_h, yaw_ell_h, center_ell_h[:, 0])
        #
        # if pitch_hyp_v != 0:
        #     coeffs_hyp_v = rotate_y(coeffs_hyp_v, pitch_hyp_v, center_hyp_v[:, 0])
        # if roll_hyp_v != 0:
        #     coeffs_hyp_v = rotate_x(coeffs_hyp_v, roll_hyp_v, center_hyp_v[:, 0])
        # if yaw_hyp_v != 0:
        #     coeffs_hyp_v = rotate_z(coeffs_hyp_v, yaw_hyp_v, center_hyp_v[:, 0])
        #
        # if pitch_hyp_h != 0:
        #     coeffs_hyp_h = rotate_y(coeffs_hyp_h, pitch_hyp_h, center_hyp_h[:, 0])
        # if roll_hyp_h != 0:
        #     coeffs_hyp_h = rotate_x(coeffs_hyp_h, roll_hyp_h, center_hyp_h[:, 0])
        # if yaw_hyp_h != 0:
        #     coeffs_hyp_h = rotate_z(coeffs_hyp_h, yaw_hyp_h, center_hyp_h[:, 0])
        if option_rotateLocal:
            if option_set:
                center_wolter_h = (np.mean(center_ell_h[:, 1:],axis=1) + np.mean(center_hyp_h[:, 1:],axis=1)) / 2
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
                    center_wolter_v = (np.mean(center_ell_v[:, 1:],axis=1) + np.mean(center_hyp_v[:, 1:],axis=1)) / 2
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
                        coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_z, yaw_ell_v, np.mean(center_ell_v[:, 1:],axis=1))
                    if pitch_ell_v != 0:
                        coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_y, pitch_ell_v, np.mean(center_ell_v[:, 1:],axis=1))
                    if roll_ell_v != 0:
                        coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_x, roll_ell_v, np.mean(center_ell_v[:, 1:],axis=1))
                    
                    if yaw_hyp_v != 0:
                        coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_z, yaw_hyp_v, np.mean(center_hyp_v[:, 1:],axis=1))
                    if pitch_hyp_v != 0:
                        coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_y, pitch_hyp_v, np.mean(center_hyp_v[:, 1:],axis=1))
                    if roll_hyp_v != 0:
                        coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_x, roll_hyp_v, np.mean(center_hyp_v[:, 1:],axis=1))

            
            else:
                if yaw_ell_v != 0:
                    coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_z, yaw_ell_v, np.mean(center_ell_v[:, 1:],axis=1))
                if pitch_ell_v != 0:
                    coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_y, pitch_ell_v, np.mean(center_ell_v[:, 1:],axis=1))
                if roll_ell_v != 0:
                    coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis2_x, roll_ell_v, np.mean(center_ell_v[:, 1:],axis=1))
                
                if yaw_hyp_v != 0:
                    coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_z, yaw_hyp_v, np.mean(center_hyp_v[:, 1:],axis=1))
                if pitch_hyp_v != 0:
                    coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_y, pitch_hyp_v, np.mean(center_hyp_v[:, 1:],axis=1))
                if roll_hyp_v != 0:
                    coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_x, roll_hyp_v, np.mean(center_hyp_v[:, 1:],axis=1))

                if pitch_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis3_y, pitch_ell_h, np.mean(center_ell_h[:, 1:],axis=1))
                if yaw_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis3_z, yaw_ell_h, np.mean(center_ell_h[:, 1:],axis=1))
                if roll_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis3_x, roll_ell_h, np.mean(center_ell_h[:, 1:],axis=1))
                if pitch_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis4_y, pitch_hyp_h, np.mean(center_hyp_h[:, 1:],axis=1))
                if yaw_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis4_z, yaw_hyp_h, np.mean(center_hyp_h[:, 1:],axis=1))
                if roll_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis4_x, roll_hyp_h, np.mean(center_hyp_h[:, 1:],axis=1))
        
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
            source = np.zeros((3, ray_num * ray_num))
            source[0, :] =+ source_shift[0] 
            source[1, :] =+ source_shift[1]
            source[2, :] =+ source_shift[2]
            # source[1, :] =+ source_shift[1] + np.random.uniform(-1e-4, 1e-4, ray_num_H * ray_num_V)
            # source[2, :] =+ source_shift[2] + np.random.uniform(-1e-4, 1e-4, ray_num_H * ray_num_V)
            if option_axial:
                rand_p0h = np.linspace(np.arctan((y1_h-source_shift[1]) / (x1_h-source_shift[0])), np.arctan((y2_h-source_shift[1]) / (x2_h-source_shift[0])), ray_num)
                rand_p0v = np.linspace(np.arctan((y1_v-source_shift[2]) / (x1_v-source_shift[0])), np.arctan((y2_v-source_shift[2]) / (x2_v-source_shift[0])), ray_num)
                rand_p0h = rand_p0h - np.mean(rand_p0h)
                rand_p0v = rand_p0v - np.mean(rand_p0v)
            if not option_axial:
                rand_p0h = np.linspace(np.arctan(y1_h / x1_h), np.arctan(y2_h / x2_h), ray_num)
                rand_p0v = np.linspace(np.arctan(y1_v / x1_v), np.arctan(y2_v / x2_v), ray_num)
            # rand_p0h = create_non_uniform_distribution(np.arctan(y1_h / x1_h), np.arctan(y2_h / x2_h), ray_num)
            # rand_p0v = create_non_uniform_distribution(np.arctan(y1_v / x1_v), np.arctan(y2_v / x2_v), ray_num)
            # rand_p0h = rand_p0h*0.1
            # rand_p0h = rand_p0v*0.1

            phai0 = np.zeros((3, ray_num * ray_num))
            for i in range(ray_num):
                rand_p0v_here = rand_p0v[i]
                phai0[1, ray_num * i:ray_num * (i + 1)] = np.tan(rand_p0h)
                phai0[2, ray_num * i:ray_num * (i + 1)] = np.tan(rand_p0v_here)
                phai0[0, ray_num * i:ray_num * (i + 1)] = 1.

            phai0 = normalize_vector(phai0)

            vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
            reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))


            vmirr_ell = mirr_ray_intersection(coeffs_ell_v, reflect1, vmirr_hyp)
            reflect2 = reflect_ray(reflect1, norm_vector(coeffs_ell_v, vmirr_ell))

            hmirr_ell = mirr_ray_intersection(coeffs_ell_h, reflect2, vmirr_ell)
            reflect3 = reflect_ray(reflect2, norm_vector(coeffs_ell_h, hmirr_ell))

            hmirr_hyp = mirr_ray_intersection(coeffs_hyp_h, reflect3, hmirr_ell,negative=True)
            reflect4 = reflect_ray(reflect3, norm_vector(coeffs_hyp_h, hmirr_hyp))

            mean_reflect4 = np.mean(reflect4,1)
            mean_reflect4 = normalize_vector(mean_reflect4)

            if option == 'sep_direct':
                defocus = find_defocus(reflect4, hmirr_hyp, s2f_middle,defocus,ray_num)

            coeffs_det = np.zeros(10)
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
                angle_h = np.arctan(angle[1, :]/angle[0, :])
                angle_v = np.arctan(angle[2, :]/angle[0, :])

                angle_v_sep_y = angle_v[thinned_array_v_y]
                angle_h_sep_y = angle_h[thinned_array_h_y]

                output_equal_v = np.linspace(angle_v_sep_y[0],angle_v_sep_y[-1],len(angle_v_sep_y))
                output_equal_h = np.linspace(angle_h_sep_y[0],angle_h_sep_y[-1],len(angle_h_sep_y))

                interp_func_v = interp1d(angle_v_sep_y, rand_p0v, kind='linear')
                interp_func_h = interp1d(angle_h_sep_y, rand_p0h, kind='linear')

                rand_p0v_new = interp_func_v(output_equal_v)
                rand_p0h_new = interp_func_h(output_equal_h)

                phai0 = np.zeros((3, ray_num * ray_num))
                for i in range(ray_num):
                    rand_p0v_here = rand_p0v_new[i]
                    phai0[1, ray_num * i:ray_num * (i + 1)] = np.tan(rand_p0h_new)
                    phai0[2, ray_num * i:ray_num * (i + 1)] = np.tan(rand_p0v_here)
                    phai0[0, ray_num * i:ray_num * (i + 1)] = 1.

                phai0 = normalize_vector(phai0)

                vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
                reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))

                dist0to1 = np.linalg.norm(vmirr_hyp - source,axis=0)


                vmirr_ell = mirr_ray_intersection(coeffs_ell_v, reflect1, vmirr_hyp)
                reflect2 = reflect_ray(reflect1, norm_vector(coeffs_ell_v, vmirr_ell))

                dist1to2 = np.linalg.norm(vmirr_ell - vmirr_hyp,axis=0)
                hmirr_ell = mirr_ray_intersection(coeffs_ell_h, reflect2, vmirr_ell)
                reflect3 = reflect_ray(reflect2, norm_vector(coeffs_ell_h, hmirr_ell))
                dist2to3 = np.linalg.norm(hmirr_ell - vmirr_ell,axis=0)

                hmirr_hyp = mirr_ray_intersection(coeffs_hyp_h, reflect3, hmirr_ell,negative=True)
                reflect4 = reflect_ray(reflect3, norm_vector(coeffs_hyp_h, hmirr_hyp))
                dist3to4 = np.linalg.norm(hmirr_hyp - hmirr_ell,axis=0)

                if option == 'sep_direct':
                    defocus = find_defocus(reflect4, hmirr_hyp, s2f_middle,defocus,ray_num)

                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)
                detcenter = plane_ray_intersection(coeffs_det, reflect4, hmirr_hyp)

                if option == 'ray':
                    from scipy.spatial import cKDTree
                    def mindist(A,B):
                        tree = cKDTree(B.T)
                        dist, idx = tree.query(A.T, k=1)  # Aの各点からBへの最近点距離
                        min_dist = np.min(dist)
                        return min_dist
                    print('======================')
                    print('workX srs 1st',np.min(vmirr_hyp[0,:]) - np.max(source[0,:]))
                    print('workX 1st 2nd',np.min(vmirr_ell[0,:]) - np.max(vmirr_hyp[0,:]))
                    print('workX 2nd 3rd',np.min(hmirr_ell[0,:]) - np.max(vmirr_ell[0,:]))
                    print('workX 3rd 4th',np.min(hmirr_hyp[0,:]) - np.max(hmirr_ell[0,:]))
                    print('workX 4th fcs',np.min(detcenter[0,:]) - np.max(hmirr_hyp[0,:]))
                    print('======================')
                    print('workAbs srs 1st',mindist(source,vmirr_hyp))
                    print('workAbs 1st 2nd',mindist(vmirr_hyp,vmirr_ell))
                    print('workAbs 2nd 3rd',mindist(vmirr_ell,hmirr_ell))
                    print('workAbs 3rd 4th',mindist(hmirr_ell,hmirr_hyp))
                    print('workAbs 4th fcs',mindist(hmirr_hyp,detcenter))

                    print('1st W upper',np.linalg.norm(vmirr_hyp[:,0] - vmirr_hyp[:,ray_num-1]))
                    print('1st W lower',np.linalg.norm(vmirr_hyp[:,-1] - vmirr_hyp[:,-ray_num]))
                    print('2nd W upper',np.linalg.norm(vmirr_ell[:,0] - vmirr_ell[:,ray_num-1]))
                    print('2nd W lower',np.linalg.norm(vmirr_ell[:,-1] - vmirr_ell[:,-ray_num]))
                    print('3rd W upper',np.linalg.norm(hmirr_ell[:,0] - hmirr_ell[:,-ray_num]))
                    print('3rd W lower',np.linalg.norm(hmirr_ell[:,ray_num-1] - hmirr_ell[:,-1]))
                    print('4th W lower',np.linalg.norm(hmirr_hyp[:,0] - hmirr_hyp[:,-ray_num]))
                    print('4th W upper',np.linalg.norm(hmirr_hyp[:,ray_num-1] - hmirr_hyp[:,-1]))

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

                    grazing_angle_1 = np.arccos(np.sum(vec0to1 * vec1to2, axis=0)) / 2
                    grazing_angle_2 = np.arccos(np.sum(vec1to2 * vec2to3, axis=0)) / 2
                    grazing_angle_3 = np.arccos(np.sum(vec2to3 * vec3to4, axis=0)) / 2
                    grazing_angle_4 = np.arccos(np.sum(vec3to4 * vec4to5, axis=0)) / 2

                    ### imshow colorbar
                    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                    im0 = axs[0, 0].imshow(grazing_angle_1.reshape(ray_num, ray_num), cmap='jet', aspect='auto')
                    im1 = axs[0, 1].imshow(grazing_angle_2.reshape(ray_num, ray_num), cmap='jet', aspect='auto')
                    im2 = axs[1, 0].imshow(grazing_angle_3.reshape(ray_num, ray_num), cmap='jet', aspect='auto')
                    im3 = axs[1, 1].imshow(grazing_angle_4.reshape(ray_num, ray_num), cmap='jet', aspect='auto')
                    fig.colorbar(im0, ax=axs[0, 0])
                    fig.colorbar(im1, ax=axs[0, 1])
                    fig.colorbar(im2, ax=axs[1, 0])
                    fig.colorbar(im3, ax=axs[1, 1])
                    ### title
                    axs[0, 0].set_title(f'Grazing Angle 1\n{1e3*grazing_angle_1.min():.2f}-{1e3*grazing_angle_1.max():.2f} mrad')
                    axs[0, 1].set_title(f'Grazing Angle 2\n{1e3*grazing_angle_2.min():.2f}-{1e3*grazing_angle_2.max():.2f} mrad')
                    axs[1, 0].set_title(f'Grazing Angle 3\n{1e3*grazing_angle_3.min():.2f}-{1e3*grazing_angle_3.max():.2f} mrad')
                    axs[1, 1].set_title(f'Grazing Angle 4\n{1e3*grazing_angle_4.min():.2f}-{1e3*grazing_angle_4.max():.2f} mrad')
                    plt.tight_layout()
                    plt.show()

                angle = reflect4

            if option == 'wave':
                print('diverg angle H',np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))
                print('diverg angle V',np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))
                # 全データからランダムに10%だけを選択
                sample_indices = np.random.choice(detcenter.shape[1], size=int(detcenter.shape[1]*0.001), replace=False)

                theta_y = -np.mean(np.arctan(angle[2, :]/angle[0, :]))
                theta_z = np.mean(np.arctan(angle[1, :]/angle[0, :]))
                source = np.zeros((3,1))
                if option_rotate==True:
                    reflect4_rotated = rotate_vectors(reflect4, -theta_y, -theta_z)
                    focus_apprx = np.mean(detcenter,axis=1)
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
                coeffs_det = np.zeros(10)
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

                if np.abs(defocusForWave) > 1e-9:
                    coeffs_det2 = np.zeros(10)
                    coeffs_det2[6] = 1
                    coeffs_det2[9] = -(s2f_middle + defocus+defocusForWave)
                    detcenter2 = plane_ray_intersection(coeffs_det2, reflect4_rotated, hmirr_hyp_points_rotated)
                    return source_rotated, vmirr_hyp_points_rotated_grid, hmirr_hyp_points_rotated_grid, vmirr_ell_points_rotated_grid, hmirr_ell_points_rotated_grid, detcenter, detcenter2, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vmirr2_norm, hmirr2_norm, vec0to1, vec1to2, vec2to3, vec3to4
                else:
                    return source_rotated, vmirr_hyp_points_rotated_grid, hmirr_hyp_points_rotated_grid, vmirr_ell_points_rotated_grid, hmirr_ell_points_rotated_grid, detcenter, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vmirr2_norm, hmirr2_norm, vec0to1, vec1to2, vec2to3, vec3to4

            
            hmirr_hyp0 = hmirr_hyp.copy()
            if option_tilt:

                if option == 'ray':
                    angles_between_rad, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect4,norm_vector(coeffs_det, detcenter))
                    theta_z = (np.max(angles_yx_rad) + np.min(angles_yx_rad))/2
                    theta_y = -(np.max(angles_zx_rad) + np.min(angles_zx_rad))/2
                    print('NA_h')
                    print(np.sin((np.max(angles_yx_rad) - np.min(angles_yx_rad))/2))
                    print('angles_yx_rad',np.sort(angles_yx_rad)[:5][::-1])
                    print('angles_yx_rad',np.sort(angles_yx_rad)[-5:][::-1])
                    print('NA_v')
                    print(np.sin((np.max(angles_zx_rad) - np.min(angles_zx_rad))/2))
                    print('angles_zx_rad',np.sort(angles_zx_rad)[:5][::-1])
                    print('angles_zx_rad',np.sort(angles_zx_rad)[-5:][::-1])
                    print('type(detcenter[0,0])',type(detcenter[0,0]))
                    print('theta_y',theta_y)
                    print('theta_z',theta_z)
                else:
                    theta_y = -np.mean(np.arctan(angle[2, :]/angle[0, :]))
                    # if option == 'ray':
                    #     plt.figure()
                    #     plt.plot(np.arctan(angle[2, :]/angle[0, :]))
                    #     plt.show()
                    theta_z = np.mean(np.arctan(angle[1, :]/angle[0, :]))

                reflect4_rotated = rotate_vectors(reflect4, -theta_y, -theta_z)
                focus_apprx = np.mean(detcenter,axis=1)
                hmirr_hyp_points_rotated = rotate_points(hmirr_hyp, focus_apprx, -theta_y, -theta_z)
                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)
                detcenter = plane_ray_intersection(coeffs_det, reflect4_rotated, hmirr_hyp_points_rotated)


                hmirr_hyp = hmirr_hyp_points_rotated.copy()
                reflect4 = reflect4_rotated.copy()
                angle = reflect4.copy()


            if option == 'sep':
                focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2 = compare_sep(reflect4_rotated, hmirr_hyp_points_rotated, coeffs_det,ray_num_H,1e-4)
                return focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2
            if option == 'sep_direct':
                focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2 = compare_sep(reflect4_rotated, hmirr_hyp_points_rotated, coeffs_det,ray_num_H,1e-4)
                return focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2

            if option == 'ray_wave':
                if option_HighNA == True:
                    defocusWave = 1e-2
                    lambda_ = 13.5
                else:
                    defocusWave = 1e-3
                    lambda_ = 1.35
                coeffs_det2 = np.zeros(10)
                coeffs_det2[6] = 1
                coeffs_det2[9] = -(s2f_middle + defocus + defocusWave)
                detcenter2 = plane_ray_intersection(coeffs_det2, reflect4, hmirr_hyp)

                dist4tofocus = np.linalg.norm(detcenter - hmirr_hyp, axis=0)
                vector4tofocus = (detcenter - hmirr_hyp) / dist4tofocus
                totalDist = dist0to1 + dist1to2 + dist2to3 + dist3to4 + dist4tofocus
                DistError = (totalDist - np.mean(totalDist))*1e9



                dist4tofocus2 = np.linalg.norm(detcenter2 - hmirr_hyp, axis=0)
                vector4tofocus2 = (detcenter2 - hmirr_hyp) / dist4tofocus2
                totalDist2 = dist0to1 + dist1to2 + dist2to3 + dist3to4 + dist4tofocus2
                DistError2 = (totalDist2 - np.mean(totalDist2))*1e9
                print('detcenter',np.mean(detcenter,axis=1))
                print('detcenter2',np.mean(detcenter2,axis=1))
                print('dist0to1',np.mean(dist0to1))
                print('dist1to2',np.mean(dist1to2))
                print('dist2to3',np.mean(dist2to3))
                print('dist3to4',np.mean(dist3to4))
                print('totalDist',np.mean(totalDist))
                print('dist4tofocus std',np.mean(dist4tofocus))
                print('dist0to1 std',np.std(dist0to1))
                print('dist1to2 std',np.std(dist1to2))
                print('dist2to3 std',np.std(dist2to3))
                print('dist3to4 std',np.std(dist3to4))
                print('dist4tofocus std',np.std(dist4tofocus))
                print('totalDist std',np.std(totalDist))
                print('np.std(totalDist)',np.std(totalDist))
                print('np.mean(totalDist)',np.mean(totalDist))
                print('np.mean(totalDist2)',np.mean(totalDist2))
                print('np.mean(totalDist2-totalDist)',np.mean(totalDist2-totalDist))
                print('np.std(totalDist2-totalDist)',np.std(totalDist2-totalDist))
                # 補間するグリッドを作成
                grid_H, grid_V = np.meshgrid(
                    np.linspace(detcenter2[1, :].min(), detcenter2[1, :].max(), ray_num_H),
                    np.linspace(detcenter2[2, :].min(), detcenter2[2, :].max(), ray_num_V)
                )

                CosAngle = angle[0,:]
                # グリッド上にデータを補間 (method: 'linear', 'nearest', 'cubic' から選択)
                if False:
                    matrixDistError2 = griddata((detcenter2[1, :], detcenter2[2, :]), DistError2, (grid_H, grid_V), method='cubic')
                    meanFocus = np.mean(detcenter,axis=1)
                    Sph = np.linalg.norm(detcenter2 - meanFocus[:, np.newaxis], axis=0) * 1e9
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
                    print('np.mean(DistError2)',np.mean(DistError2))
                    print('np.std(DistError2)',np.std(DistError2))
                    print('np.mean(Sph)',np.mean(Sph))
                    print('np.std(Sph)',np.std(Sph))
                    print('np.mean(Wave2)',np.mean(Wave2))
                    print('np.std(Wave2)',np.std(Wave2))
                    print('grid_H.shape',grid_H.shape)
                    print('Wave2.shape',Wave2.shape)
                    print('detcenter2.shape',detcenter2.shape)

                    matrixWave2 = griddata((detcenter2[1, :], detcenter2[2, :]), Wave2, (grid_H, grid_V), method='cubic')
                    matrixWave2 = matrixWave2 - np.nanmean(matrixWave2)

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
                    phase = wave_error * 2 * np.pi
                    input_complex = np.exp(1j * phase)
                    offset = defocusWave * (np.sqrt(1. + np.tan(grid_H)**2 + np.tan(grid_V)**2) - 1)
                    input_complex *= np.exp(1j * offset)
                    output_complex = np.zeros_like(input_complex, dtype=np.complex128)
                    x_input = defocusWave * np.tan(grid_H)
                    y_input = defocusWave * np.tan(grid_V)
                    calcsize = 200e-9
                    xoutput , youtput = np.meshgrid(
                        np.linspace(-calcsize, calcsize, output_complex.shape[1]),
                        np.linspace(-calcsize, calcsize, output_complex.shape[0])
                    )
                    for i_o in range(output_complex.shape[0]):
                        for j_o in range(output_complex.shape[1]):
                            for i_i in range(input_complex.shape[0]):
                                for j_i in range(input_complex.shape[1]):
                                    if np.isnan(input_complex[i_i, j_i]) or np.isnan(output_complex[i_o, j_o]):
                                        continue
                                    else:
                                        dist = defocusWave * np.sqrt(defocusWave**2 + (xoutput[i_o, j_o] - x_input[i_i, j_i])**2 + (youtput[i_o, j_o] - y_input[i_i, j_i])**2)
                                        phase_shift =  - 2 * np.pi * dist / lambda_
                                        output_complex[i_o, j_o] += input_complex[i_i, j_i] * np.exp(1j * phase_shift) / dist
                    psf = np.abs(output_complex)**2
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
                # plt.show()
                # rectified_img = extract_affine_square_region(matrixWave2_Corrected/lambda_, target_size=256)
                rectified_img = extract_affine_square_region(matrixWave2/lambda_, target_size=256)

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

                    assesorder = 5
                    fit_datas, inner_products, orders = lf.match_legendre_multi(rectified_img[1:-2, 1:-2], assesorder)
                    length = len(inner_products)
                    pvs = np.zeros(length+1)
                    fig, axes = plt.subplots(assesorder, assesorder, figsize=(16, 16))
                    for i in range(length):
                        ny = orders[i][0]
                        nx = orders[i][1]
                        print(f"ny: {ny}, nx: {nx}, Inner Product: {inner_products[i]:.3e}")
                        axes[ny, nx].imshow(fit_datas[i], cmap='jet', vmin=-1/256, vmax=1/256)
                        pvs[i] = (np.nanmax(fit_datas[i]) - np.nanmin(fit_datas[i])) * np.sign(inner_products[i])
                        axes[ny, nx].set_title(f"ny: {ny}, nx: {nx} \n Inner Product: {inner_products[i]:.3e} \n PV: {pvs[i]:.3e}")
                        axes[ny, nx].axis('off')

                        ### set colorbar for each subplot
                        # cbar = plt.colorbar(axes[ny, nx].images[0], ax=axes[ny, nx], fraction=0.046, pad=0.04)
                    axes[-1, -1].imshow(rectified_img[1:-2, 1:-2], cmap='jet', vmin=-1/256, vmax=1/256)
                    cbar = plt.colorbar(axes[-1, -1].images[0], ax=axes[-1, -1], fraction=0.046, pad=0.04)
                    fit_sum = np.sum(fit_datas, axis=0)
                    axes[-2, -1].imshow(fit_sum, cmap='jet', vmin=-1/256, vmax=1/256)
                    # cbar = plt.colorbar(axes[-2, -1].images[0], ax=axes[-2, -1], fraction=0.046, pad=0.04)
                    axes[-1, -2].imshow(rectified_img[1:-2, 1:-2]-fit_sum, cmap='jet', vmin=-1/256, vmax=1/256)
                    plt.savefig(os.path.join(directory_name, 'legendre_fit2.png'), transparent=True, dpi=300)

                    conditions_file_path = os.path.join(directory_name, 'optical_params.txt')

                    # テキストファイルに変数の値や計算条件を書き込む
                    with open(conditions_file_path, 'w') as file:
                        file.write("input\n")
                        file.write("====================\n")
                        file.write(f"params[0]: {params[0]}\n")
                        file.write(f"params[1]: {params[1]}\n")
                        file.write(f"params[2]: {params[2]}\n")
                        file.write(f"params[3]: {params[3]}\n")
                        file.write(f"params[4]: {params[4]}\n")
                        file.write(f"params[5]: {params[5]}\n")
                        file.write(f"params[6]: {params[6]}\n")
                        file.write(f"params[7]: {params[7]}\n")
                        file.write(f"params[8]: {params[8]}\n")
                        file.write(f"params[9]: {params[9]}\n")
                        file.write(f"params[10]: {params[10]}\n")
                        file.write(f"params[11]: {params[11]}\n")
                        file.write(f"params[12]: {params[12]}\n")
                        file.write(f"params[13]: {params[13]}\n")
                        file.write(f"params[14]: {params[14]}\n")
                        file.write(f"params[15]: {params[15]}\n")
                        file.write(f"params[16]: {params[16]}\n")
                        file.write(f"params[17]: {params[17]}\n")
                        file.write(f"params[18]: {params[18]}\n")
                        file.write(f"params[19]: {params[19]}\n")
                        file.write(f"params[20]: {params[20]}\n")
                        file.write(f"params[21]: {params[21]}\n")
                        file.write(f"params[22]: {params[22]}\n")
                        file.write(f"params[23]: {params[23]}\n")
                        file.write(f"params[24]: {params[24]}\n")
                        file.write(f"params[25]: {params[25]}\n")
                    plt.show()



                if False:
                    # psf, x_out, y_out = fresnel_psf(matrixWave2_Corrected, lambda_=lambda_, z=-defocusWave, grid_x=grid_H, grid_y=grid_V)
                    calcrange=1.e-6
                    psf, x_out, y_out = fresnel_integral(
                        phi=matrixWave2_Corrected*1e-9,
                        grid_x=grid_H-np.mean(grid_H),
                        grid_y=grid_V-np.mean(grid_V),
                        lambda_=lambda_*1e-9,
                        z=-defocusWave,
                        x_out_range=(-calcrange, calcrange),
                        y_out_range=(-calcrange, calcrange),
                        dx_out=calcrange/65,
                        dy_out=calcrange/65,
                    )

                    def compute_fwhm(x, intensity_1d):
                        """1次元の強度分布から FWHM を計算"""
                        dx = np.abs(x[1] - x[0])
                        num_over_half_max = np.sum(intensity_1d >= 0.5 * np.max(intensity_1d))
                        fwhm = (num_over_half_max-1) * dx
                        return fwhm
                    psf_x = psf[psf.shape[0] // 2, :]
                    psf_y = psf[:, psf.shape[1] // 2]
                    half_max = 0.5 * np.max(psf)
                    mask = psf >= half_max
                    dx = np.abs(x_out[1] - x_out[0])
                    dy = np.abs(y_out[1] - y_out[0])
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
                print(np.cos(theta1_v)*s2f_middle)
                print(theta1_h)
                print(np.mean(detcenter[0,:]))
                print(np.mean(detcenter[1,:]))
                print(np.mean(detcenter[2,:]))
                # print(np.mean(detcenter[0,:]))
                print(coeffs_det)
                print('s2f_H',s2f_H)
                print('s2f_V',s2f_V)
                mabiki = round(np.sqrt(ray_num_H*ray_num_V)/50)
                mabiki =  1
                defocussize = 4e-6
                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus) + defocussize
                detcenter1 = plane_ray_intersection(coeffs_det, reflect4, hmirr_hyp)

                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus) - defocussize
                detcenter2 = plane_ray_intersection(coeffs_det, reflect4, hmirr_hyp)
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[::ray_num+1]
                thinned_array = first_thinned_array[::mabiki]

                print('Oblique1',thinned_array)

                obl_1 = (detcenter1[1,thinned_array] + detcenter1[2,thinned_array])/np.sqrt(2)
                obl_2 = (detcenter2[1,thinned_array] + detcenter2[2,thinned_array])/np.sqrt(2)
                axs[0,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[obl_1,obl_2],c='y')
                axs[0,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(obl_2),np.max(obl_1)],color='k')
                axs[0,0].set_title('Oblique1 aperture 0')
                axs[0,0].set_xlabel('Axial (m)')
                axs[0,0].set_ylabel('Oblique1 Position (m)')

                obl_1 = (-detcenter1[1,thinned_array] + detcenter1[2,thinned_array])/np.sqrt(2)
                obl_2 = (-detcenter2[1,thinned_array] + detcenter2[2,thinned_array])/np.sqrt(2)
                axs[0,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[obl_1,obl_2],c='y')
                axs[0,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(obl_2),np.max(obl_1)],color='k')
                axs[0,1].set_title('Oblique1 aperture 0')
                axs[0,1].set_xlabel('Axial (m)')
                axs[0,1].set_ylabel('Oblique2 Position (m)')

                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[ray_num-1::ray_num-1][:-1]
                # first_thinned_array = first_thinned_array[:-1]
                thinned_array = first_thinned_array[::mabiki]

                print('Oblique2',thinned_array)

                obl_1 = (detcenter1[1,thinned_array] + detcenter1[2,thinned_array])/np.sqrt(2)
                obl_2 = (detcenter2[1,thinned_array] + detcenter2[2,thinned_array])/np.sqrt(2)
                axs[1,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[obl_1,obl_2],c='y')
                axs[1,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(obl_2),np.max(obl_1)],color='k')
                axs[1,0].set_title('Oblique2 aperture 0')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Oblique1 Position (m)')

                obl_1 = (-detcenter1[1,thinned_array] + detcenter1[2,thinned_array])/np.sqrt(2)
                obl_2 = (-detcenter2[1,thinned_array] + detcenter2[2,thinned_array])/np.sqrt(2)
                axs[1,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[obl_1,obl_2],c='y')
                axs[1,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(obl_2),np.max(obl_1)],color='k')
                axs[1,1].set_title('Oblique2 aperture 0')
                axs[1,1].set_xlabel('Axial (m)')
                axs[1,1].set_ylabel('Oblique2 Position (m)')
                plt.savefig('multiple_plots_ray_oblique.png', dpi=300)
                # plt.show()
                plt.close()
                fig, axs = plt.subplots(3, 2, figsize=(10, 15))
                # 範囲内の値を間引く
                start = 0
                end = ray_num
                thinned_array = crop(start, end, mabiki)



                axs[0,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='r')
                axs[0,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[0,0].set_title('Ray from V')
                axs[0,0].set_xlabel('Axial (m)')
                axs[0,0].set_ylabel('Horizontal Position (m)')

                axs[0,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='r')
                axs[0,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[0,1].set_title('Ray from H')
                axs[0,1].set_xlabel('Axial (m)')
                axs[0,1].set_ylabel('Vertical Position (m)')
                # plt.show()

                # 範囲内の値を間引く
                start = round(ray_num*(ray_num-1)/2)
                end = round(ray_num*(ray_num+1)/2)
                thinned_array = crop(start, end, mabiki)

                axs[1,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='y')
                axs[1,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[1,0].set_title('Ray from V')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Horizontal Position (m)')

                axs[1,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='y')
                axs[1,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[1,1].set_title('Ray from H')
                axs[1,1].set_xlabel('Axial (m)')
                axs[1,1].set_ylabel('Vertical Position (m)')

                # 範囲内の値を間引く
                start = ray_num**2 - ray_num
                end = ray_num**2
                thinned_array = crop(start, end, mabiki)

                axs[2,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='g')
                axs[2,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[2,0].set_title('Ray from V')
                axs[2,0].set_xlabel('Axial (m)')
                axs[2,0].set_ylabel('Horizontal Position (m)')

                axs[2,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='g')
                axs[2,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[2,1].set_title('Ray from H')
                axs[2,1].set_xlabel('Axial (m)')
                axs[2,1].set_ylabel('Vertical Position (m)')
                fig.suptitle('V aperture 0')
                plt.savefig('multiple_plots_ray_v.png', dpi=300)
                # plt.show()
                plt.close()


                fig, axs = plt.subplots(3, 2, figsize=(10, 15))
                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[::ray_num]
                thinned_array = first_thinned_array[::mabiki]
                fig.suptitle('H aperture 0')

                axs[0,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='r')
                axs[0,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[0,0].set_title('Ray from V')
                axs[0,0].set_xlabel('Axial (m)')
                axs[0,0].set_ylabel('Horizontal Position (m)')

                axs[0,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='r')
                axs[0,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[0,1].set_title('Ray from H')
                axs[0,1].set_xlabel('Axial (m)')
                axs[0,1].set_ylabel('Vertical Position (m)')


                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[round((ray_num-1)/2)::ray_num]

                # さらに間引き (skip_rateでさらに間引く)
                thinned_array = first_thinned_array[::mabiki]

                axs[1,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='y')
                axs[1,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[1,0].set_title('Ray from V')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Horizontal Position (m)')

                axs[1,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='y')
                axs[1,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[1,1].set_title('Ray from H')
                axs[1,1].set_xlabel('Axial (m)')
                axs[1,1].set_ylabel('Vertical Position (m)')

                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                # 最初の間引き (ray_num-1から始めてray_numごとに要素を取得)
                first_thinned_array = original_array[ray_num-1::ray_num]

                # さらに間引き (skip_rateでさらに間引く)
                thinned_array = first_thinned_array[::mabiki]

                axs[2,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='g')
                axs[2,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[2,0].set_title('Ray from V')
                axs[2,0].set_xlabel('Axial (m)')
                axs[2,0].set_ylabel('Horizontal Position (m)')

                axs[2,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='g')
                axs[2,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[2,1].set_title('Ray from H')
                axs[2,1].set_xlabel('Axial (m)')
                axs[2,1].set_ylabel('Vertical Position (m)')
                plt.savefig('multiple_plots_ray_h.png', dpi=300)
                # plt.show()
                plt.close()

                # # # plot_ray_sideview(8,10,mabiki,reflect1,vmirr_hyp,ray_num)
                # plot_ray_sideview(-5,35,mabiki,reflect2,vmirr_ell,ray_num)
                # # # plot_ray_sideview(8,10,mabiki,reflect3,vmirr_ell,ray_num)
                # # # plot_ray_sideview(0.2,0.2,mabiki,reflect3,vmirr_ell,ray_num)

                phai0 = normalize_vector(phai0)

                plt.figure()
                plt.scatter(phai0[1, :], phai0[2, :])
                plt.scatter(phai0[1, ::ray_num], phai0[2, ::ray_num],color='r')
                plt.scatter(phai0[1, round((ray_num-1)/2)::ray_num], phai0[2, round((ray_num-1)/2)::ray_num],color='y')
                plt.scatter(phai0[1, ray_num-1::ray_num], phai0[2, ray_num-1::ray_num],color='g')
                plt.title('angle')
                plt.xlabel('Horizontal Angle (rad)')
                plt.ylabel('Vertical Angle (rad)')
                plt.axis('equal')
                # plt.show()
                plt.close()

                plt.figure()
                plt.scatter(phai0[1, :], phai0[2, :])
                plt.scatter((phai0[1, :ray_num]), (phai0[2, :ray_num]),color='r')
                plt.scatter(phai0[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)], phai0[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)],color='y')
                plt.scatter((phai0[1, -ray_num:]), (phai0[2, -ray_num:]),color='g')
                plt.title('angle')
                plt.xlabel('Horizontal Angle (rad)')
                plt.ylabel('Vertical Angle (rad)')
                plt.axis('equal')
                # plt.show()
                plt.close()

                vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
                reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))

                angle_1st, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect1,norm_vector(coeffs_hyp_v, vmirr_hyp))



                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)

                detcenter0 = plane_ray_intersection(coeffs_det, reflect4, hmirr_hyp)

                # angles_between_rad, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect_ray(reflect4, norm_vector(coeffs_det, detcenter0)),norm_vector(coeffs_det, detcenter0))
                angles_between_rad, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect4,norm_vector(coeffs_det, detcenter0))
                print('NA_h')
                print(np.sin((np.max(angles_yx_rad) - np.min(angles_yx_rad))/2))
                print('angles_yx_rad',np.sort(angles_yx_rad)[:5][::-1])
                print('angles_yx_rad',np.sort(angles_yx_rad)[-5:][::-1])
                print('NA_v')
                print(np.sin((np.max(angles_zx_rad) - np.min(angles_zx_rad))/2))
                print('angles_zx_rad',np.sort(angles_zx_rad)[:5][::-1])
                print('angles_zx_rad',np.sort(angles_zx_rad)[-5:][::-1])
                print('type(detcenter[0,0])',type(detcenter[0,0]))
                if option_tilt:
                    reflect4_rotated = rotate_vectors(reflect4, -theta_y, -theta_z)
                    focus_apprx = np.mean(detcenter,axis=1)
                    hmirr_hyp_points_rotated = rotate_points(hmirr_hyp, focus_apprx, -theta_y, -theta_z)
                    coeffs_det = np.zeros(10)
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
                axs[0, 0].plot([detcenter1[0, thinned_array_h_r[:third_r]], detcenter2[0, thinned_array_h_r[:third_r]]],
                               [detcenter1[1, thinned_array_h_r[:third_r]], detcenter2[1, thinned_array_h_r[:third_r]]], c='r')

                # 後ろ1/3のプロット（ピンク）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_r[-third_r:]], detcenter2[0, thinned_array_h_r[-third_r:]]],
                               [detcenter1[1, thinned_array_h_r[-third_r:]], detcenter2[1, thinned_array_h_r[-third_r:]]], c='purple')

                # thinned_array_h_y の前1/3のプロット（darkyellow）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_y[:third_y]], detcenter2[0, thinned_array_h_y[:third_y]]],
                               [detcenter1[1, thinned_array_h_y[:third_y]], detcenter2[1, thinned_array_h_y[:third_y]]], c='y')

                # thinned_array_h_y の後ろ1/3のプロット（purple）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_y[-third_y:]], detcenter2[0, thinned_array_h_y[-third_y:]]],
                               [detcenter1[1, thinned_array_h_y[-third_y:]], detcenter2[1, thinned_array_h_y[-third_y:]]], c='#B8860B')

                # thinned_array_h_g の前1/3のプロット（緑）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_g[:third_g]], detcenter2[0, thinned_array_h_g[:third_g]]],
                               [detcenter1[1, thinned_array_h_g[:third_g]], detcenter2[1, thinned_array_h_g[:third_g]]], c='g')

                # thinned_array_h_g の後ろ1/3のプロット（薄緑）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_g[-third_g:]], detcenter2[0, thinned_array_h_g[-third_g:]]],
                               [detcenter1[1, thinned_array_h_g[-third_g:]], detcenter2[1, thinned_array_h_g[-third_g:]]], c='lightgreen')
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
                axs[1,1].scatter(np.mean(detcenter[1, ::ray_num]), np.mean(detcenter[2, ::ray_num]),color='r',marker='x',s=100)
                axs[1,1].scatter(np.mean(detcenter[1, round((ray_num-1)/2)::ray_num]), np.mean(detcenter[2, round((ray_num-1)/2)::ray_num]),color='y',marker='x',s=100)
                axs[1,1].scatter(np.mean(detcenter[1, ray_num-1::ray_num]), np.mean(detcenter[2, ray_num-1::ray_num]),color='g',marker='x',s=100)
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
                axs[1, 0].plot([detcenter1[0, thinned_array_v_r[:third_r]], detcenter2[0, thinned_array_v_r[:third_r]]],
                               [detcenter1[2, thinned_array_v_r[:third_r]], detcenter2[2, thinned_array_v_r[:third_r]]], c='r')

                # 後ろ1/3のプロット（ピンク）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_r[-third_r:]], detcenter2[0, thinned_array_v_r[-third_r:]]],
                               [detcenter1[2, thinned_array_v_r[-third_r:]], detcenter2[2, thinned_array_v_r[-third_r:]]], c='purple')

                # thinned_array_v_y の前1/3のプロット（darkyellow）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_y[:third_y]], detcenter2[0, thinned_array_v_y[:third_y]]],
                               [detcenter1[2, thinned_array_v_y[:third_y]], detcenter2[2, thinned_array_v_y[:third_y]]], c='y')

                # thinned_array_v_y の後ろ1/3のプロット（purple）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_y[-third_y:]], detcenter2[0, thinned_array_v_y[-third_y:]]],
                               [detcenter1[2, thinned_array_v_y[-third_y:]], detcenter2[2, thinned_array_v_y[-third_y:]]], c='#B8860B')

                # thinned_array_v_g の前1/3のプロット（緑）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_g[:third_g]], detcenter2[0, thinned_array_v_g[:third_g]]],
                               [detcenter1[2, thinned_array_v_g[:third_g]], detcenter2[2, thinned_array_v_g[:third_g]]], c='g')

                # thinned_array_v_g の後ろ1/3のプロット（薄緑）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_g[-third_g:]], detcenter2[0, thinned_array_v_g[-third_g:]]],
                               [detcenter1[2, thinned_array_v_g[-third_g:]], detcenter2[2, thinned_array_v_g[-third_g:]]], c='lightgreen')

                axs[1,0].plot([input_val, input_val],
                            [np.min(detcenter2[2, :]), np.max(detcenter1[2, :])], color='k')
                axs[1,0].set_title('H aperture 0')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Vertical Position (m)')

                axs[0,1].scatter(detcenter[1, :], detcenter[2, :])
                axs[0,1].scatter(detcenter[1, :ray_num], detcenter[2, :ray_num],color='r')
                axs[0,1].scatter(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)], detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)],color='y')
                axs[0,1].scatter(detcenter[1, -ray_num:], detcenter[2, -ray_num:],color='g')
                axs[0,1].scatter(detcenter[1, ray_num-1::ray_num-1][:-1], detcenter[2, ray_num-1::ray_num-1][:-1],color='k')
                axs[0,1].scatter(detcenter[1, ::ray_num+1], detcenter[2, ::ray_num+1],color='gray')
                axs[0,1].scatter(np.mean(detcenter[1, :ray_num]), np.mean(detcenter[2, :ray_num]),color='r',marker='x',s=100)
                axs[0,1].scatter(np.mean(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]), np.mean(detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]),color='y',marker='x',s=100)
                axs[0,1].scatter(np.mean(detcenter[1, -ray_num:]), np.mean(detcenter[2, -ray_num:]),color='g',marker='x',s=100)
                axs[0,1].set_title('focus @V aperture 0')
                axs[0,1].set_xlabel('Horizontal (m)')
                axs[0,1].set_ylabel('Vertical (m)')
                axs[0,1].axis('equal')

                # タイトル用の新しいサイズ計算
                size_v = np.max(detcenter[2,:]) - np.min(detcenter[2,:])
                size_h = np.max(detcenter[1,:]) - np.min(detcenter[1,:])

                # タイトルの更新
                title1 = f'Params 0-1: {params[0:2]}'
                title2 = f'Params 2-7: {params[2:8]}'
                title3 = f'Params 8-13: {params[8:14]}'
                title4 = f'Params 14-19: {params[14:20]}'
                title5 = f'Params 20-25: {params[20:26]}'
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
                        axs[1,1].scatter(np.mean(detcenter[1, ::ray_num]), np.mean(detcenter[2, ::ray_num]),color='r',marker='x',s=100)
                        axs[1,1].scatter(np.mean(detcenter[1, round((ray_num-1)/2)::ray_num]), np.mean(detcenter[2, round((ray_num-1)/2)::ray_num]),color='y',marker='x',s=100)
                        axs[1,1].scatter(np.mean(detcenter[1, ray_num-1::ray_num]), np.mean(detcenter[2, ray_num-1::ray_num]),color='g',marker='x',s=100)
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
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_r[:third_r]], detcenter2[0, thinned_array_h_r[:third_r]]],
                                       [detcenter1[1, thinned_array_h_r[:third_r]], detcenter2[1, thinned_array_h_r[:third_r]]], c='r')

                        # 後ろ1/3のプロット（ピンク）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_r[-third_r:]], detcenter2[0, thinned_array_h_r[-third_r:]]],
                                       [detcenter1[1, thinned_array_h_r[-third_r:]], detcenter2[1, thinned_array_h_r[-third_r:]]], c='purple')

                        # thinned_array_h_y の前1/3のプロット（darkyellow）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_y[:third_y]], detcenter2[0, thinned_array_h_y[:third_y]]],
                                       [detcenter1[1, thinned_array_h_y[:third_y]], detcenter2[1, thinned_array_h_y[:third_y]]], c='y')

                        # thinned_array_h_y の後ろ1/3のプロット（purple）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_y[-third_y:]], detcenter2[0, thinned_array_h_y[-third_y:]]],
                                       [detcenter1[1, thinned_array_h_y[-third_y:]], detcenter2[1, thinned_array_h_y[-third_y:]]], c='#B8860B')

                        # thinned_array_h_g の前1/3のプロット（緑）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_g[:third_g]], detcenter2[0, thinned_array_h_g[:third_g]]],
                                       [detcenter1[1, thinned_array_h_g[:third_g]], detcenter2[1, thinned_array_h_g[:third_g]]], c='g')

                        # thinned_array_h_g の後ろ1/3のプロット（薄緑）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_g[-third_g:]], detcenter2[0, thinned_array_h_g[-third_g:]]],
                                       [detcenter1[1, thinned_array_h_g[-third_g:]], detcenter2[1, thinned_array_h_g[-third_g:]]], c='lightgreen')
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
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_r[:third_r]], detcenter2[0, thinned_array_v_r[:third_r]]],
                                       [detcenter1[2, thinned_array_v_r[:third_r]], detcenter2[2, thinned_array_v_r[:third_r]]], c='r')

                        # 後ろ1/3のプロット（ピンク）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_r[-third_r:]], detcenter2[0, thinned_array_v_r[-third_r:]]],
                                       [detcenter1[2, thinned_array_v_r[-third_r:]], detcenter2[2, thinned_array_v_r[-third_r:]]], c='purple')

                        # thinned_array_v_y の前1/3のプロット（darkyellow）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_y[:third_y]], detcenter2[0, thinned_array_v_y[:third_y]]],
                                       [detcenter1[2, thinned_array_v_y[:third_y]], detcenter2[2, thinned_array_v_y[:third_y]]], c='y')

                        # thinned_array_v_y の後ろ1/3のプロット（purple）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_y[-third_y:]], detcenter2[0, thinned_array_v_y[-third_y:]]],
                                       [detcenter1[2, thinned_array_v_y[-third_y:]], detcenter2[2, thinned_array_v_y[-third_y:]]], c='#B8860B')

                        # thinned_array_v_g の前1/3のプロット（緑）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_g[:third_g]], detcenter2[0, thinned_array_v_g[:third_g]]],
                                       [detcenter1[2, thinned_array_v_g[:third_g]], detcenter2[2, thinned_array_v_g[:third_g]]], c='g')

                        # thinned_array_v_g の後ろ1/3のプロット（薄緑）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_g[-third_g:]], detcenter2[0, thinned_array_v_g[-third_g:]]],
                                       [detcenter1[2, thinned_array_v_g[-third_g:]], detcenter2[2, thinned_array_v_g[-third_g:]]], c='lightgreen')

                        axs[1,0].plot([input_val, input_val],
                                    [np.min(detcenter2[2, :]), np.max(detcenter1[2, :])], color='k')
                        axs[1,0].set_title('H aperture 0')
                        axs[1,0].set_xlabel('Axial (m)')
                        axs[1,0].set_ylabel('Vertical Position (m)')

                        axs[0,1].cla()  # 右側プロットをクリア
                        axs[0,1].scatter(detcenter[1, :], detcenter[2, :])
                        axs[0,1].scatter(detcenter[1, :ray_num], detcenter[2, :ray_num],color='r')
                        axs[0,1].scatter(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)], detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)],color='y')
                        axs[0,1].scatter(detcenter[1, -ray_num:], detcenter[2, -ray_num:],color='g')
                        axs[0,1].scatter(detcenter[1, ray_num-1::ray_num-1][:-1], detcenter[2, ray_num-1::ray_num-1][:-1],color='k')
                        axs[0,1].scatter(detcenter[1, ::ray_num+1], detcenter[2, ::ray_num+1],color='gray')
                        axs[0,1].scatter(np.mean(detcenter[1, :ray_num]), np.mean(detcenter[2, :ray_num]),color='r',marker='x',s=100)
                        axs[0,1].scatter(np.mean(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]), np.mean(detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]),color='y',marker='x',s=100)
                        axs[0,1].scatter(np.mean(detcenter[1, -ray_num:]), np.mean(detcenter[2, -ray_num:]),color='g',marker='x',s=100)
                        axs[0,1].set_title('focus @V aperture 0')
                        axs[0,1].set_xlabel('Horizontal (m)')
                        axs[0,1].set_ylabel('Vertical (m)')
                        axs[0,1].axis('equal')

                        # axs[0,2].scatter(input_val,np.mean(detcenter[1, :ray_num]),color='r')
                        # axs[0,2].scatter(input_val,np.mean(detcenter[1, round((ray_num**2)/2) : round((ray_num**2 + ray_num*2)/2)]),color='y')
                        # axs[0,2].scatter(input_val,np.mean(detcenter[1, -ray_num:-1]),color='g')

                        # axs[1,2].scatter(input_val,np.mean(detcenter[2, ::ray_num]),color='r')
                        # axs[1,2].scatter(input_val,np.mean(detcenter[2, round(ray_num/2)-1::ray_num]),color='y')
                        # axs[1,2].scatter(input_val,np.mean(detcenter[2, ray_num-1::ray_num]),color='g')
                        # axs[1,3].scatter(input_val,np.mean(detcenter[1, ::ray_num]),color='r')
                        # axs[1,3].scatter(input_val,np.mean(detcenter[1, round(ray_num/2)-1::ray_num]),color='y')
                        # axs[1,3].scatter(input_val,np.mean(detcenter[1, ray_num-1::ray_num]),color='g')

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





    # 焦点面での標準偏差を計算
        return vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle
elif option_wolter_3_3_tandem:
    def plot_result_debug(params,option,source_shift=[0.,0.,0.],option_tilt = True,option_legendre=False):
        defocus, astigH, \
        pitch_hyp_v, roll_hyp_v, yaw_hyp_v, decenterX_hyp_v, decenterY_hyp_v, decenterZ_hyp_v,\
        pitch_hyp_h, roll_hyp_h, yaw_hyp_h, decenterX_hyp_h, decenterY_hyp_h, decenterZ_hyp_h,\
        pitch_ell_v, roll_ell_v, yaw_ell_v, decenterX_ell_v, decenterY_ell_v, decenterZ_ell_v,\
        pitch_ell_h, roll_ell_h, yaw_ell_h, decenterX_ell_h, decenterY_ell_h, decenterZ_ell_h  = params ### 3型 1st 1型 4th 3型 2nd 1型 3rd

        # # Mirror parameters EUV
        # a_hyp_v = np.float64(72.96002945938)
        # b_hyp_v = np.float64(0.134829747201017)
        # a_ell_v = np.float64(0.442)
        # b_ell_v = np.float64(0.0607128830733533)
        # length_hyp_v = np.float64(0.115)
        # length_ell_v = np.float64(0.229790269646258)
        # theta1_v = np.float64(4.73536529533549E-05)
        #
        # a_hyp_h = np.float64(73.018730871665)
        # b_hyp_h = np.float64(0.0970536727319812)
        # a_ell_h = np.float64(0.38125)
        # b_ell_h = np.float64(0.0397791317992322)
        # length_hyp_h = np.float64(0.25)
        # length_ell_h = np.float64(0.0653872838592807)
        # theta1_h = np.float64(5.6880350884129E-05)

        if True:  # 初期設定 2025/06/30
            ### 3型V Setting2
            a_hyp_v  =   np.float64(72.9845)
            b_hyp_v  =   np.float64(0.21238958153899)
            a_ell_v  =   np.float64(0.281)
            b_ell_v  =   np.float64(0.0531241733689647)
            hyp_length_v  =   np.float64(0.05)
            ell_length_v  =   np.float64(0.266860702984731)
            theta1_v  =   np.float64(4.21919940469323E-05)
            theta2_v  =   np.float64(0.1)
            theta3_v  =   np.float64(0.200042191994047)
            theta4_v  =   np.float64(0.190438080754236)
            theta5_v  =   np.float64(-0.180833969514426)
            phai_hyp_v  =   np.float64(-0.100042191994047)
            phai_ell_v  =   np.float64(0.0453183883535131)
            F_eff_v  =   np.float64(0.0338049837589551)
            Mag_v  =   np.float64(4333.31604026723)
            NA_v  =   np.float64(0.0818366883383903)
            Aperture_v  =   np.float64(0.0505138153539869)
            F0_F2_v  =   np.float64(146.521483347413)

            a_hyp_v  =   np.float64(72.9848)
            b_hyp_v  =   np.float64(0.210324155665437)
            a_ell_v  =   np.float64(0.2857)
            b_ell_v  =   np.float64(0.0530618591483233)
            hyp_length_v  =   np.float64(0.05)
            ell_length_v  =   np.float64(0.276166893612172)
            theta1_v  =   np.float64(4.13752081278497E-05)
            theta2_v  =   np.float64(0.1)
            theta3_v  =   np.float64(0.200041375208128)
            theta4_v  =   np.float64(0.187243234395316)
            theta5_v  =   np.float64(-0.174445093582505)
            phai_hyp_v  =   np.float64(-0.100041375208128)
            phai_ell_v  =   np.float64(0.0488521496575497)
            F_eff_v  =   np.float64(0.0343933172595428)
            Mag_v  =   np.float64(4259.46907766477)
            NA_v  =   np.float64(0.0811197336351448)
            Aperture_v  =   np.float64(0.0514087503904502)
            F0_F2_v  =   np.float64(146.531664662596)

            # a_hyp_v  =   np.float64(72.9848)
            # b_hyp_v  =   np.float64(0.210324155665437)
            # a_ell_v  =   np.float64(0.3257)
            # b_ell_v  =   np.float64(0.0609957911371367)
            # hyp_length_v  =   np.float64(0.05)
            # ell_length_v  =   np.float64(0.316162847545838)
            # theta1_v  =   np.float64(4.13752081278497E-05)
            # theta2_v  =   np.float64(0.1)
            # theta3_v  =   np.float64(0.200041375208128)
            # theta4_v  =   np.float64(0.188723956733222)
            # theta5_v  =   np.float64(-0.177406538258317)
            # phai_hyp_v  =   np.float64(-0.100041375208128)
            # phai_ell_v  =   np.float64(0.0480340729020616)
            # F_eff_v  =   np.float64(0.0338255011587307)
            # Mag_v  =   np.float64(4333.30625052992)
            # NA_v  =   np.float64(0.0824831338396396)
            # Aperture_v  =   np.float64(0.0593139402081157)
            # F0_F2_v  =   np.float64(146.610081099594)



            ### 3型H setting2
            a_hyp_h  =   np.float64(73.1626385934245)
            b_hyp_h  =   np.float64(0.28245130901327)
            a_ell_h  =   np.float64(0.101)
            b_ell_h  =   np.float64(0.0261431003274516)
            hyp_length_h  =   np.float64(0.043)
            ell_length_h  =   np.float64(0.0593351406813002)
            theta1_h  =   np.float64(6.33844354505952E-05)
            theta2_h  =   np.float64(0.117)
            theta3_h  =   np.float64(0.234063384435451)
            theta4_h  =   np.float64(0.263427681323838)
            theta5_h  =   np.float64(-0.292791978212226)
            phai_hyp_h  =   np.float64(-0.117063384435451)
            phai_ell_h  =   np.float64(-0.00628857148389368)
            F_eff_h  =   np.float64(0.0308012818374266)
            Mag_h  =   np.float64(4755.99304104836)
            NA_h  =   np.float64(0.0813737660485408)
            Aperture_h  =   np.float64(0.0154503670007265)
            F0_F2_h  =   np.float64(146.521483356008)

            # a_hyp_h  =   np.float64(73.206937469515)
            # b_hyp_h  =   np.float64(0.282536782718687)
            # a_ell_h  =   np.float64(0.101)
            # b_ell_h  =   np.float64(0.0261430961181383)
            # hyp_length_h  =   np.float64(0.043)
            # ell_length_h  =   np.float64(0.0593351486637329)
            # theta1_h  =   np.float64(6.33460806383912E-05)
            # theta2_h  =   np.float64(0.117)
            # theta3_h  =   np.float64(0.234063346080638)
            # theta4_h  =   np.float64(0.263427637900091)
            # theta5_h  =   np.float64(-0.292791929719544)
            # phai_hyp_h  =   np.float64(-0.117063346080638)
            # phai_ell_h  =   np.float64(-0.0062885701083498)
            # F_eff_h  =   np.float64(0.0308012663400348)
            # Mag_h  =   np.float64(4758.87186668706)
            # NA_h  =   np.float64(0.0813737648885928)
            # Aperture_h  =   np.float64(0.0154503665916128)
            # F0_F2_h  =   np.float64(146.610081110267)






            length_hyp_v = hyp_length_v
            length_ell_v = ell_length_v
            length_hyp_h = hyp_length_h
            length_ell_h = ell_length_h


        omega_v = theta1_v - theta5_v
        if option == 'ray':
            omega_v = theta1_v - theta5_v
            print('omega_v',omega_v)
            # omega_h = theta1_h + theta5_h
            omega_h = theta1_h - theta5_h
            print('omega_h',omega_h)
        # print('omega_v',omega_v)
        # omega_v = omega_v/4
        # #
        omega_v = 0.

        org_hyp_v = np.sqrt(a_hyp_v**2 + b_hyp_v**2)
        org_hyp_h = np.sqrt(a_hyp_h**2 + b_hyp_h**2)

        org_ell_v = np.sqrt(a_ell_v**2 - b_ell_v**2)
        org_ell_h = np.sqrt(a_ell_h**2 - b_ell_h**2)

        # astig_v = (org_hyp_v - org_hyp_h)/2
        # astig_v_ = (org_hyp_v - org_hyp_h)/2*np.linspace(0,4,10)
        n = 20
        # param_ = np.linspace(-1,1,n)
        std_v = np.full(n, np.nan)
        std_h = np.full(n, np.nan)


        # astig_v = astig_v_[j]
        # astig_v = 0.5*-0.16626315789473686
        # astig_v = 0.5
        # print(astig_v_)

        # Input parameters
        ray_num = 53
        # defocus = 0.2*0.03157894736842107
        # defocus = 0
        # print(defocus)
        # astig_v = 0
        # pitch_hyp_h = 0
        # roll_hyp_h = 0
        # yaw_hyp_h = 0
        # pitch_ell_h = 0
        # roll_ell_h = 0
        # yaw_ell_h = 0
        # pitch_hyp_v = 0
        # roll_hyp_v = 0
        # yaw_hyp_v = 0.1*param_[j]
        # pitch_ell_v = 0
        # roll_ell_v = 0
        # yaw_ell_v = 0.1*param_[j]
        # option_2mirror = True
        # Input parameters
        ray_num_H = 53
        ray_num_V = 53
        ray_num = 53
        if option == 'ray':
            ray_num_H = 53
            ray_num_V = 53
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
        c_v = np.zeros(10)
        c_v[0] = 1 / a_hyp_v**2
        c_v[2] = -1 / b_hyp_v**2
        c_v[9] = -1.
        org_v = np.sqrt(a_hyp_v**2 + b_hyp_v**2)
        c_v = shift_x(c_v, org_v)

        center_v = mirr_ray_intersection(c_v, np.array([[np.cos(theta1_v)], [0.], [np.sin(theta1_v)]]), np.array([[0.], [0.], [0.]]))
        if not np.isreal(center_v).all():
            return np.inf
        x1_v = center_v[0, 0] - length_hyp_v / 2
        x2_v = center_v[0, 0] + length_hyp_v / 2
        # y1_v = np.sqrt((-1 + (x1_v / a_hyp_v)**2) * b_hyp_v**2)
        # y2_v = np.sqrt((-1 + (x2_v / a_hyp_v)**2) * b_hyp_v**2)
        y1_v = calc_Y_hyp(a_hyp_v, b_hyp_v, x1_v)
        y2_v = calc_Y_hyp(a_hyp_v, b_hyp_v, x2_v)
        # print(np.arctan(y1_v/x1_v))
        # print(np.arctan(y2_v/x2_v))
        accept_v = np.abs(y2_v - y1_v)
        l1v = np.linalg.norm(c_v)

        ### 1 ell
        c_h = np.zeros(10)
        c_h[0] = 1 / a_hyp_h**2
        c_h[1] = -1 / b_hyp_h**2
        c_h[9] = -1.
        org_h = np.sqrt(a_hyp_h**2 + b_hyp_h**2)
        c_h = shift_x(c_h, org_h)
        center_h = mirr_ray_intersection(c_h, np.array([[np.cos(theta1_h)], [np.sin(theta1_h)], [0.]]), np.array([[0.], [0.], [0.]]))
        if not np.isreal(center_h).all():
            return np.inf
        x1_h = center_h[0, 0] - length_hyp_h / 2
        x2_h = center_h[0, 0] + length_hyp_h / 2
        y1_h = calc_Y_hyp(a_hyp_h, b_hyp_h, x1_h)
        y2_h = calc_Y_hyp(a_hyp_h, b_hyp_h, x2_h)
        # print(np.arctan(y1_h/x1_h))
        # print(np.arctan(y2_h/x2_h))
        accept_h = np.abs(y2_h - y1_h)
        l1h = np.linalg.norm(c_h)

        if False:
            ratio_here = 0.25
            xw_v = x2_v - x1_v
            yw_v = y2_v - y1_v
            x1_v = x1_v + xw_v*ratio_here
            y1_v = y1_v + yw_v*ratio_here
            x2_v = x2_v - xw_v*ratio_here
            y2_v = y2_v - yw_v*ratio_here

            xw_h = x2_h - x1_h
            yw_h = y2_h - y1_h
            x1_h = x1_h + xw_h*ratio_here
            y1_h = y1_h + yw_h*ratio_here
            x2_h = x2_h - xw_h*ratio_here
            y2_h = y2_h - yw_h*ratio_here


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
        axis_x = np.array([1., 0., 0.])
        axis_y = np.array([0., 1., 0.])
        axis_z = np.array([0., 0., 1.])
        coeffs_hyp_v = np.zeros(10)
        coeffs_hyp_v[0] = 1 / a_hyp_v**2
        coeffs_hyp_v[2] = -1 / b_hyp_v**2
        coeffs_hyp_v[9] = -1.
        coeffs_hyp_v = shift_x(coeffs_hyp_v, org_hyp_v)
        if option_axial:
            # coeffs_hyp_v = rotate_y(coeffs_hyp_v, theta1_v, [0, 0, 0])
            coeffs_hyp_v, rotation_matrix = rotate_general_axis(coeffs_hyp_v, axis_y, theta1_v, [0, 0, 0])
            axis_x, axis_y, axis_z = rotatematrix(rotation_matrix, axis_x, axis_y, axis_z)

        if option_alignment and option_axial:
            bufray = np.zeros((3, 3))
            ### 4隅の光線
            # theta_cntr_h = (np.arctan(y2_h / x2_h) + np.arctan(y1_h / x1_h))/2.
            theta_cntr_v = (np.arctan(y2_v / x2_v) + np.arctan(y1_v / x1_v))/2.
            def print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h):
                l2_h = (4*a_hyp_h**2 + (org_hyp_h*2)**2 -4*a_hyp_h*(org_hyp_h*2)*np.cos(theta1_h))/(4*org_hyp_h - 4*a_hyp_h)
                # l2_v = (4*a_hyp_v**2 + (org_hyp_v*2)**2 -4*a_hyp_v*(org_hyp_v*2)*np.cos(theta1_v))/(4*org_hyp_v - 4*a_hyp_v)

                l1_h = 2*a_hyp_h +l2_h
                # l1_v = 2*a_hyp_v +l2_v

                theta2_h = np.arcsin(org_hyp_h*2*np.sin(theta1_h)/l2_h)/2
                # theta2_v = np.arcsin(org_hyp_v*2*np.sin(theta1_v)/l2_v)/2

                theta3_h = np.arcsin(l1_h*np.sin(theta1_h)/l2_h)
                # theta3_v = np.arcsin(l1_v*np.sin(theta1_v)/l2_v)

                l4_h = ((org_ell_h)**2 - 2*org_ell_h*a_ell_h*np.cos(theta3_h) + a_ell_h**2)/(a_ell_h - org_ell_h*np.cos(theta3_h))
                # l4_v = ((org_ell_v)**2 - 2*org_ell_v*a_ell_v*np.cos(theta3_v) + a_ell_v**2)/(a_ell_v - org_ell_v*np.cos(theta3_v))

                l3_h = 2*a_ell_h - l2_h - l4_h
                # l3_v = 2*a_ell_v - l2_v - l4_v

                theta5_h = np.arcsin((2*a_ell_h - l4_h)*np.sin(theta3_h)/l4_h)
                # theta5_v = np.arcsin((2*a_ell_v - l4_v)*np.sin(theta3_v)/l4_v)

                theta4_h = (theta5_h+theta3_h)/2.
                # theta4_v = (theta5_v+theta3_v)/2.
                # print('theta4_v incidence ell',theta4_v)
                theta4_h = np.arcsin(2*org_ell_h*np.sin(theta3_h)/l4_h)/2
                # theta4_v = np.arcsin(2*org_ell_v*np.sin(theta3_v)/l4_v)/2
                return theta2_h, theta3_h, theta4_h, theta5_h, l1_h, l2_h, l3_h, l4_h
            # theta3_h, theta5_h = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h)
            # theta3_h1, theta5_h1 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y1_h / x1_h))
            # theta3_h2, theta5_h2 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y2_h / x2_h))
            theta2_v, theta3_v, theta4_v, theta5_v, l1_v, l2_v, l3_v, l4_v = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,theta1_v)
            theta2_v1, theta3_v1, theta4_v1, theta5_v1, l1_v1, l2_v1, l3_v1, l4_v1 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y1_v / x1_v))
            theta2_v2, theta3_v2, theta4_v2, theta5_v2, l1_v2, l2_v2, l3_v2, l4_v2 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y2_v / x2_v))
            omega_v = (theta5_v1 + theta5_v2 + np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v))/2
            # omegah1 = (theta3_h1 + theta3_h2 - np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))/2
            # omegav2 = (np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v) + theta5_v1 + theta5_v2)/2
            # omega_v = omega_v*0.6
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
                print('theta1_v1',np.arctan(y1_v / x1_v))
                print('theta1_v2',np.arctan(y2_v / x2_v))
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
                # print('na_h',np.sin((theta5_h1 - theta5_h2))/2.)
                print('na_v',np.sin((theta5_v1 - theta5_v2))/2.)
                # print('div_h',(theta5_h1 - theta5_h2))
                print('div_v',(theta5_v1 - theta5_v2))
                print('div0_h',np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))
                print('div0_v',np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))
                print('===== ===== =====')



            # theta_source_1_h = np.arctan(y1_h / x1_h) - theta_cntr_h
            # theta_source_2_h = np.arctan(y2_h / x2_h) - theta_cntr_h
            theta_source_1_v = np.arctan(y1_v / x1_v) - theta_cntr_v
            theta_source_2_v = np.arctan(y2_v / x2_v) - theta_cntr_v

            bufray[0, 0] = 1.

            bufray[0, 1] = 1.
            # bufray[1, 1] = np.tan(theta_source_1_h)
            bufray[2, 1] = np.tan(theta_source_1_v)

            bufray[0, 2] = 1.
            # bufray[1, 2] = np.tan(theta_source_2_h)
            bufray[2, 2] = np.tan(theta_source_2_v)
            #
            # bufray[0, 3] = 1.
            # bufray[1, 3] = np.tan(theta_source_2_h)
            # bufray[2, 3] = np.tan(theta_source_1_v)
            #
            # bufray[0, 4] = 1.
            # bufray[1, 4] = np.tan(theta_source_2_h)
            # bufray[2, 4] = np.tan(theta_source_2_v)

            # source = np.zeros((3, 5))

            # bufray = np.zeros((3, 2))
            # bufray[0, 0] = 1.
            # bufray[1, 0] = np.tan(theta1_h)
            # bufray[2, 0] = np.tan(theta1_v)
            # bufray[0, 1] = 1.
            # bufray[1, 1] = np.tan(theta1_h)
            # bufray[2, 1] = np.tan(theta1_v)
            source = np.zeros((3, 3))

        else:
            bufray = np.zeros((3, 2))
            if option_axial:
                bufray[0, :] = 1.
                bufray[1, :] = np.tan(theta1_h)
                bufray[2, :] = np.tan(theta1_v)
            # def print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h):
            #     l2_h = (4*a_hyp_h**2 + (org_hyp_h*2)**2 -4*a_hyp_h*(org_hyp_h*2)*np.cos(theta1_h))/(4*org_hyp_h - 4*a_hyp_h)
            #     # l2_v = (4*a_hyp_v**2 + (org_hyp_v*2)**2 -4*a_hyp_v*(org_hyp_v*2)*np.cos(theta1_v))/(4*org_hyp_v - 4*a_hyp_v)
            #
            #     l1_h = 2*a_hyp_h +l2_h
            #     # l1_v = 2*a_hyp_v +l2_v
            #
            #     theta2_h = np.arcsin(org_hyp_h*2*np.sin(theta1_h)/l2_h)/2
            #     # theta2_v = np.arcsin(org_hyp_v*2*np.sin(theta1_v)/l2_v)/2
            #
            #     theta3_h = np.arcsin(l1_h*np.sin(theta1_h)/l2_h)
            #     # theta3_v = np.arcsin(l1_v*np.sin(theta1_v)/l2_v)
            #
            #     l4_h = ((org_ell_h)**2 - 2*org_ell_h*a_ell_h*np.cos(theta3_h) + a_ell_h**2)/(a_ell_h - org_ell_h*np.cos(theta3_h))
            #     # l4_v = ((org_ell_v)**2 - 2*org_ell_v*a_ell_v*np.cos(theta3_v) + a_ell_v**2)/(a_ell_v - org_ell_v*np.cos(theta3_v))
            #
            #     l3_h = 2*a_ell_h - l2_h - l4_h
            #     # l3_v = 2*a_ell_v - l2_v - l4_v
            #
            #     theta5_h = np.arcsin((2*a_ell_h - l4_h)*np.sin(theta3_h)/l4_h)
            #     # theta5_v = np.arcsin((2*a_ell_v - l4_v)*np.sin(theta3_v)/l4_v)
            #
            #     theta4_h = (theta5_h+theta3_h)/2.
            #     # theta4_v = (theta5_v+theta3_v)/2.
            #     print('theta1',theta1_h)
            #     print('l2',l2_h)
            #     # print('l2_v',l2_v)
            #     print('l1',l1_h)
            #     # print('l1_v',l1_v)
            #     print('theta2 incidence hyp',theta2_h)
            #     # print('theta2_v incidence hyp',theta2_v)
            #     print('theta3',theta3_h)
            #     # print('theta3_v',theta3_v)
            #     print('l3',l3_h)
            #     # print('l3_v',l3_v)
            #     print('l4',l4_h)
            #     # print('l4_v',l4_v)
            #     print('hyp to ell',np.cos(theta3_h)*(l3_h))
            #     # print('hyp to ell v',np.cos(theta3_v)*(l3_v))
            #     print('theta4 incidence ell',theta4_h)
            #     # print('theta4_v incidence ell',theta4_v)
            #     theta4_h = np.arcsin(2*org_ell_h*np.sin(theta3_h)/l4_h)/2
            #     # theta4_v = np.arcsin(2*org_ell_v*np.sin(theta3_v)/l4_v)/2
            #     print('theta4 incidence ell',theta4_h)
            #     # print('theta4_v incidence ell',theta4_v)
            #     print('theta5 focal',theta5_h)
            #     # print('theta5_v focal',theta5_v)
            #     print('width1',l1_h*np.cos(theta1_h))
            #     print('width2',l3_h*np.cos(theta3_h))
            #     print('width3',l4_h*np.cos(theta5_h))
            #     print('')
            #     return theta3_h, theta5_h
            #
            #
            # theta3_h, theta5_h = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h)
            # theta3_h1, theta5_h1 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y1_h / x1_h))
            # theta3_h2, theta5_h2 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y2_h / x2_h))
            # theta3_v, theta5_v = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,theta1_v)
            # theta3_v1, theta5_v1 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y1_v / x1_v))
            # theta3_v2, theta5_v2 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y2_v / x2_v))
            # omegav1 = (theta3_v1 + theta3_v2 - np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))/2
            # omegah1 = (theta3_h1 + theta3_h2 - np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))/2
            # omegav2 = (np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v) + theta5_v1 + theta5_v2)/2
            source = np.zeros((3, 2))


        bufray = normalize_vector(bufray)

        center_hyp_v = mirr_ray_intersection(coeffs_hyp_v, bufray, source)
        if not np.isreal(center_hyp_v).all():
            return np.inf
        bufreflect1 = reflect_ray(bufray, norm_vector(coeffs_hyp_v, center_hyp_v))

        bufreflangle1_y = np.arctan(np.mean(bufreflect1[2, 1:]) / np.mean(bufreflect1[0, 1:]))

        if option == 'ray':
            print('coeffs_hyp_v',coeffs_hyp_v)
            print('center_hyp_v',center_hyp_v)
            print('bufray',bufray)
            print('np.mean(bufreflect1[2, 1:])',np.mean(bufreflect1[2, 1:]))
            print('np.mean(bufreflect1[0, 1:])',np.mean(bufreflect1[0, 1:]))
            print('angle_y 1st to 2nd',bufreflangle1_y)

        # print(center_hyp_v)
        # print(bufreflect1)
        # print(np.arctan(bufreflect1[2, 0] / bufreflect1[0, 0]))
        # print('1st')

        # V ell mirror set (2nd)
        axis2_x = np.array([1., 0., 0.])
        axis2_y = np.array([0., 1., 0.])
        axis2_z = np.array([0., 0., 1.])
        coeffs_ell_v = np.zeros(10)
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
        mean_bufreflect1 = np.mean(bufreflect1[:, 1:],axis = 1)
        if option == 'ray':
            roty_local2 = np.arctan(np.dot(mean_bufreflect1,axis2_z) / np.dot(mean_bufreflect1,axis2_x))
            print('rot localy2',roty_local2)
            print('coeffs_ell_v',coeffs_ell_v)

        center_ell_v = mirr_ray_intersection(coeffs_ell_v, bufreflect1, center_hyp_v)
        if not np.isreal(center_ell_v).all():
            return np.inf

        # if option_alignment:
        #     if not optin_axialrotation:
        #         coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle1_y, np.mean(center_hyp_v[:, 1:],axis=1))
        #     if optin_axialrotation:
        #         if option_rotateLocal:
        #             coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis2_y, -omegav1, np.mean(center_hyp_h[:, 1:],axis=1))
        #             axis2_x, axis2_y, axis2_z = rotatematrix(rotation_matrix, axis2_x, axis2_y, axis2_z)
        #         else:
        #             coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle1_y, np.mean(center_hyp_h[:, 1:],axis=1))
        #         center_hyp_h = mirr_ray_intersection(coeffs_hyp_h, bufreflect1, center_hyp_v)
        bufreflect2 = reflect_ray(bufreflect1, norm_vector(coeffs_ell_v, center_ell_v))
        bufreflangle2_z = np.arctan(np.mean(bufreflect2[1, 1:]) / np.mean(bufreflect2[0, 1:]))
        bufreflangle2_y = np.arctan(np.mean(bufreflect2[2, 1:]) / np.mean(bufreflect2[0, 1:]))

        if option == 'ray':
            print('angle_y 2nd to 3rd',bufreflangle2_y)
            print('angle_z 2nd to 3rd',bufreflangle2_z)
        # print(bufreflect2)
        # print(bufreflangle2_z)

        # print('2nd')
        # print(center_hyp_h)



        # Set H ellipse mirror in the vert set (3rd)
        axis3_x = np.array([1., 0., 0.])
        axis3_y = np.array([0., 1., 0.])
        axis3_z = np.array([0., 0., 1.])
        coeffs_hyp_h = np.zeros(10)
        coeffs_hyp_h[0] = 1 / a_hyp_h**2
        coeffs_hyp_h[1] = -1 / b_hyp_h**2
        coeffs_hyp_h[9] = -1.

        coeffs_hyp_h = shift_x(coeffs_hyp_h, org_hyp_h + astigH)

        if option_axial:
            coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis3_z, -theta1_h, [0, 0, 0])
            axis3_x, axis3_y, axis3_z = rotatematrix(rotation_matrix, axis3_x, axis3_y, axis3_z)

        center_hyp_h = mirr_ray_intersection(coeffs_hyp_h, bufreflect2, center_ell_v)
        if not np.isreal(center_hyp_h).all():
            return np.inf
        if option_alignment:
            if not optin_axialrotation:
                coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle2_y, np.mean(center_ell_v[:, 1:],axis=1))
            if optin_axialrotation:
                if option_rotateLocal:
                    ### 正確に言うと　omega_h
                    coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis3_y, omega_v, np.mean(center_ell_v[:, 1:],axis=1))
                    axis3_x, axis3_y, axis3_z = rotatematrix(rotation_matrix, axis3_x, axis3_y, axis3_z)
                else:
                    coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle2_y, center_hyp_h[:, 0])
                center_hyp_h = mirr_ray_intersection(coeffs_hyp_h, bufreflect2, center_ell_v)
        bufreflect3 = reflect_ray(bufreflect2, norm_vector(coeffs_hyp_h, center_hyp_h))
        bufreflangle3_y = np.arctan(np.mean(bufreflect3[2, 1:]) / np.mean(bufreflect3[0, 1:]))
        bufreflangle3_z = np.arctan(np.mean(bufreflect3[1, 1:]) / np.mean(bufreflect3[0, 1:]))

        if option == 'ray':
            print('angle_y 3rd to 4th',bufreflangle3_y)
            print('angle_z 3rd to 4th',bufreflangle3_z)
        # print(bufreflangle3_y)
        # print('3rd')



        # Set ellipse mirror in the horiz set (4th)
        axis4_x = np.array([1., 0., 0.])
        axis4_y = np.array([0., 1., 0.])
        axis4_z = np.array([0., 0., 1.])
        coeffs_ell_h = np.zeros(10)
        coeffs_ell_h[0] = 1 / a_ell_h**2
        coeffs_ell_h[1] = 1 / b_ell_h**2
        coeffs_ell_h[9] = -1.

        coeffs_ell_h = shift_x(coeffs_ell_h, 2 * org_hyp_h + org_ell_h + astigH)
        if option_axial:
            # coeffs_ell_h = rotate_y(coeffs_ell_h, bufreflangle1_y, center_hyp_v[:, 0])
            coeffs_ell_h, rotation_matrix = rotate_general_axis(coeffs_ell_h, axis4_z, -theta1_h, [0, 0, 0])
            axis4_x, axis4_y, axis4_z = rotatematrix(rotation_matrix, axis4_x, axis4_y, axis4_z)

        center_ell_h = mirr_ray_intersection(coeffs_ell_h, bufreflect3, center_hyp_h)
        if not np.isreal(center_ell_h).all():
            return np.inf
        if option_alignment:
            if not optin_axialrotation:
                coeffs_ell_h = rotate_y(coeffs_ell_h, -bufreflangle3_y, np.mean(center_hyp_v[:, 1:],axis=1))
            if optin_axialrotation:
                if option_rotateLocal:
                    coeffs_ell_h, rotation_matrix = rotate_general_axis(coeffs_ell_h, axis4_y, omega_v, np.mean(center_ell_v[:, 1:],axis=1))
                    axis4_x, axis4_y, axis4_z = rotatematrix(rotation_matrix, axis4_x, axis4_y, axis4_z)
                else:
                    coeffs_ell_h = rotate_y(coeffs_ell_h, -bufreflangle3_y, np.mean(center_ell_h[:, 1:],axis=1))
                center_ell_h = mirr_ray_intersection(coeffs_ell_h, bufreflect3, center_hyp_h)
        bufreflect4 = reflect_ray(bufreflect3, norm_vector(coeffs_ell_h, center_ell_h))
        bufreflangle4_z = np.arctan(np.mean(bufreflect4[1, 1:]) / np.mean(bufreflect4[0, 1:]))
        bufreflangle4_y = np.arctan(np.mean(bufreflect4[2, 1:]) / np.mean(bufreflect4[0, 1:]))

        s2f_H = 2 * org_hyp_h + 2 * org_ell_h
        s2f_V = 2 * org_hyp_v + 2 * org_ell_v
        # print(s2f_H)
        # print(s2f_V)
        s2f_middle = (s2f_H + s2f_V) / 2
        coeffs_det = np.zeros(10)
        coeffs_det[6] = 1.
        coeffs_det[9] = -(s2f_middle + defocus)

        # if option_axial:
        #     coeffs_det = rotate_y(coeffs_det, theta1_v, [0, 0, 0])
        #     coeffs_det = rotate_y(coeffs_det, bufreflangle1_y, center_hyp_v[:, 0])
        #     coeffs_det = rotate_z(coeffs_det, -theta1_h, [0, 0, 0])

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
        if center_hyp_h[0,0]<center_ell_v[0,0]:
            print('conflict 2nd 3rd')
            return np.inf
        if center_ell_h[0,0]<center_hyp_h[0,0]:
            print('conflict 3rd 4th')
            return np.inf

        detcenter = plane_ray_intersection(coeffs_det, bufreflect4, center_ell_h)
        detcenter = detcenter[:, 0]

        # if pitch_ell_v != 0:
        #     coeffs_ell_v = rotate_y(coeffs_ell_v, pitch_ell_v, center_ell_v[:, 0])
        # if roll_ell_v != 0:
        #     coeffs_ell_v = rotate_x(coeffs_ell_v, roll_ell_v, center_ell_v[:, 0])
        # if yaw_ell_v != 0:
        #     coeffs_ell_v = rotate_z(coeffs_ell_v, yaw_ell_v, center_ell_v[:, 0])
        #
        # if pitch_ell_h != 0:
        #     coeffs_ell_h = rotate_y(coeffs_ell_h, pitch_ell_h, center_ell_h[:, 0])
        # if roll_ell_h != 0:
        #     coeffs_ell_h = rotate_x(coeffs_ell_h, roll_ell_h, center_ell_h[:, 0])
        # if yaw_ell_h != 0:
        #     coeffs_ell_h = rotate_z(coeffs_ell_h, yaw_ell_h, center_ell_h[:, 0])
        #
        # if pitch_hyp_v != 0:
        #     coeffs_hyp_v = rotate_y(coeffs_hyp_v, pitch_hyp_v, center_hyp_v[:, 0])
        # if roll_hyp_v != 0:
        #     coeffs_hyp_v = rotate_x(coeffs_hyp_v, roll_hyp_v, center_hyp_v[:, 0])
        # if yaw_hyp_v != 0:
        #     coeffs_hyp_v = rotate_z(coeffs_hyp_v, yaw_hyp_v, center_hyp_v[:, 0])
        #
        # if pitch_hyp_h != 0:
        #     coeffs_hyp_h = rotate_y(coeffs_hyp_h, pitch_hyp_h, center_hyp_h[:, 0])
        # if roll_hyp_h != 0:
        #     coeffs_hyp_h = rotate_x(coeffs_hyp_h, roll_hyp_h, center_hyp_h[:, 0])
        # if yaw_hyp_h != 0:
        #     coeffs_hyp_h = rotate_z(coeffs_hyp_h, yaw_hyp_h, center_hyp_h[:, 0])
        if option_rotateLocal:
            if True:
                center_wolter_v = (np.mean(center_ell_v[:, 1:],axis=1) + np.mean(center_hyp_v[:, 1:],axis=1)) / 2
                
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

                center_wolter_h = (np.mean(center_ell_h[:, 1:],axis=1) + np.mean(center_hyp_h[:, 1:],axis=1)) / 2
                if pitch_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis4_y, pitch_ell_h, center_wolter_h)
                if yaw_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis4_z, yaw_ell_h, center_wolter_h)
                if roll_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis4_x, roll_ell_h, center_wolter_h)
                if pitch_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis3_y, pitch_hyp_h, center_wolter_h)
                if yaw_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis3_z, yaw_hyp_h, center_wolter_h)
                if roll_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis3_x, roll_hyp_h, center_wolter_h)
            else:
                if yaw_ell_v != 0:
                    coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis3_z, yaw_ell_v, np.mean(center_ell_v[:, 1:],axis=1))
                if pitch_ell_v != 0:
                    coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis3_y, pitch_ell_v, np.mean(center_ell_v[:, 1:],axis=1))
                if roll_ell_v != 0:
                    coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis3_x, roll_ell_v, np.mean(center_ell_v[:, 1:],axis=1))
                
                if yaw_hyp_v != 0:
                    coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis2_z, yaw_hyp_v, np.mean(center_hyp_v[:, 1:],axis=1))
                if pitch_hyp_v != 0:
                    coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis2_y, pitch_hyp_v, np.mean(center_hyp_v[:, 1:],axis=1))
                if roll_hyp_v != 0:
                    coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis2_x, roll_hyp_v, np.mean(center_hyp_v[:, 1:],axis=1))

                if pitch_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis3_y, pitch_ell_h, np.mean(center_ell_h[:, 1:],axis=1))
                if yaw_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis3_z, yaw_ell_h, np.mean(center_ell_h[:, 1:],axis=1))
                if roll_ell_h != 0:
                    coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis3_x, roll_ell_h, np.mean(center_ell_h[:, 1:],axis=1))
                if pitch_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis3_y, pitch_hyp_h, np.mean(center_hyp_h[:, 1:],axis=1))
                if yaw_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis3_z, yaw_hyp_h, np.mean(center_hyp_h[:, 1:],axis=1))
                if roll_hyp_h != 0:
                    coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis3_x, roll_hyp_h, np.mean(center_hyp_h[:, 1:],axis=1))

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
            source = np.zeros((3, ray_num * ray_num))
            source[0, :] =+ source_shift[0] 
            source[1, :] =+ source_shift[1]
            source[2, :] =+ source_shift[2]
            # source[1, :] =+ source_shift[1] + np.random.uniform(-1e-4, 1e-4, ray_num_H * ray_num_V)
            # source[2, :] =+ source_shift[2] + np.random.uniform(-1e-4, 1e-4, ray_num_H * ray_num_V)
            if option_axial:
                rand_p0h = np.linspace(np.arctan((y1_h-source_shift[1]) / (x1_h-source_shift[0])), np.arctan((y2_h-source_shift[1]) / (x2_h-source_shift[0])), ray_num)
                rand_p0v = np.linspace(np.arctan((y1_v-source_shift[2]) / (x1_v-source_shift[0])), np.arctan((y2_v-source_shift[2]) / (x2_v-source_shift[0])), ray_num)
                rand_p0h = rand_p0h - np.mean(rand_p0h)
                rand_p0v = rand_p0v - np.mean(rand_p0v)
            if not option_axial:
                rand_p0h = np.linspace(np.arctan(y1_h / x1_h), np.arctan(y2_h / x2_h), ray_num)
                rand_p0v = np.linspace(np.arctan(y1_v / x1_v), np.arctan(y2_v / x2_v), ray_num)
            # rand_p0h = create_non_uniform_distribution(np.arctan(y1_h / x1_h), np.arctan(y2_h / x2_h), ray_num)
            # rand_p0v = create_non_uniform_distribution(np.arctan(y1_v / x1_v), np.arctan(y2_v / x2_v), ray_num)
            # rand_p0h = rand_p0h*0.1
            # rand_p0h = rand_p0v*0.1

            phai0 = np.zeros((3, ray_num * ray_num))
            for i in range(ray_num):
                rand_p0v_here = rand_p0v[i]
                phai0[1, ray_num * i:ray_num * (i + 1)] = np.tan(rand_p0h)
                phai0[2, ray_num * i:ray_num * (i + 1)] = np.tan(rand_p0v_here)
                phai0[0, ray_num * i:ray_num * (i + 1)] = 1.

            phai0 = normalize_vector(phai0)

            # plt.figure()
            # plt.scatter(phai0[1, :], phai0[2, :])
            # plt.title('angle')
            # plt.xlabel('Horizontal Angle (rad)')
            # plt.ylabel('Vertical Angle (rad)')
            # plt.show()

            vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
            reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))


            vmirr_ell = mirr_ray_intersection(coeffs_ell_v, reflect1, vmirr_hyp)
            reflect2 = reflect_ray(reflect1, norm_vector(coeffs_ell_v, vmirr_ell))
            # if option == 'ray':
            #     plot_ray_sideview(75,85,2,reflect2,vmirr_ell,ray_num)

            hmirr_hyp = mirr_ray_intersection(coeffs_hyp_h, reflect2, vmirr_ell)
            reflect3 = reflect_ray(reflect2, norm_vector(coeffs_hyp_h, hmirr_hyp))

            hmirr_ell = mirr_ray_intersection(coeffs_ell_h, reflect3, hmirr_hyp)
            reflect4 = reflect_ray(reflect3, norm_vector(coeffs_ell_h, hmirr_ell))

            # if option == 'ray':
            #     plot_ray_sideview(146,1,2,reflect4,hmirr_hyp,ray_num)

            mean_reflect4 = np.mean(reflect4,1)
            # print(mean_reflect2)
            # print(np.sum(mean_reflect2*mean_reflect2))
            mean_reflect4 = normalize_vector(mean_reflect4)
            # print(mean_reflect2)
            # print(np.sum(mean_reflect2*mean_reflect2))

            if option == 'sep_direct':
                defocus = find_defocus(reflect4, hmirr_hyp, s2f_middle,defocus,ray_num)

            coeffs_det = np.zeros(10)
            coeffs_det[6] = 1
            coeffs_det[9] = -(s2f_middle + defocus)
            detcenter = plane_ray_intersection(coeffs_det, reflect4, hmirr_ell)
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
                angle_h = np.arctan(angle[1, :]/angle[0, :])
                angle_v = np.arctan(angle[2, :]/angle[0, :])

                angle_v_sep_y = angle_v[thinned_array_v_y]
                angle_h_sep_y = angle_h[thinned_array_h_y]

                output_equal_v = np.linspace(angle_v_sep_y[0],angle_v_sep_y[-1],len(angle_v_sep_y))
                output_equal_h = np.linspace(angle_h_sep_y[0],angle_h_sep_y[-1],len(angle_h_sep_y))

                interp_func_v = interp1d(angle_v_sep_y, rand_p0v, kind='linear')
                interp_func_h = interp1d(angle_h_sep_y, rand_p0h, kind='linear')

                rand_p0v_new = interp_func_v(output_equal_v)
                rand_p0h_new = interp_func_h(output_equal_h)

                phai0 = np.zeros((3, ray_num * ray_num))
                for i in range(ray_num):
                    rand_p0v_here = rand_p0v_new[i]
                    phai0[1, ray_num * i:ray_num * (i + 1)] = np.tan(rand_p0h_new)
                    phai0[2, ray_num * i:ray_num * (i + 1)] = np.tan(rand_p0v_here)
                    phai0[0, ray_num * i:ray_num * (i + 1)] = 1.

                phai0 = normalize_vector(phai0)

                vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
                reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))

                dist0to1 = np.linalg.norm(vmirr_hyp - source,axis=0)


                vmirr_ell = mirr_ray_intersection(coeffs_ell_v, reflect1, vmirr_hyp)
                reflect2 = reflect_ray(reflect1, norm_vector(coeffs_ell_v, vmirr_ell))

                dist1to2 = np.linalg.norm(vmirr_ell - vmirr_hyp,axis=0)
                hmirr_hyp = mirr_ray_intersection(coeffs_hyp_h, reflect2, vmirr_ell)
                reflect3 = reflect_ray(reflect2, norm_vector(coeffs_hyp_h, hmirr_hyp))
                dist2to3 = np.linalg.norm(hmirr_hyp - vmirr_ell,axis=0)

                hmirr_ell = mirr_ray_intersection(coeffs_ell_h, reflect3, hmirr_ell)
                reflect4 = reflect_ray(reflect3, norm_vector(coeffs_ell_h, hmirr_ell))
                dist3to4 = np.linalg.norm(hmirr_ell - hmirr_hyp,axis=0)

                if option == 'sep_direct':
                    defocus = find_defocus(reflect4, hmirr_ell, s2f_middle,defocus,ray_num)

                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)
                detcenter = plane_ray_intersection(coeffs_det, reflect4, hmirr_ell)

                if option == 'ray':
                    from scipy.spatial import cKDTree
                    def mindist(A,B):
                        tree = cKDTree(B.T)
                        dist, idx = tree.query(A.T, k=1)  # Aの各点からBへの最近点距離
                        min_dist = np.min(dist)
                        return min_dist
                    print('======================')
                    print('workX srs 1st',np.min(vmirr_hyp[0,:]) - np.max(source[0,:]))
                    print('workX 1st 2nd',np.min(vmirr_ell[0,:]) - np.max(vmirr_hyp[0,:]))
                    print('workX 2nd 3rd',np.min(hmirr_hyp[0,:]) - np.max(vmirr_ell[0,:]))
                    print('workX 3rd 4th',np.min(hmirr_ell[0,:]) - np.max(hmirr_hyp[0,:]))
                    print('workX 4th fcs',np.min(detcenter[0,:]) - np.max(hmirr_ell[0,:]))
                    print('======================')
                    print('workAbs srs 1st',mindist(source,vmirr_hyp))
                    print('workAbs 1st 2nd',mindist(vmirr_hyp,vmirr_ell))
                    print('workAbs 2nd 3rd',mindist(vmirr_ell,hmirr_hyp))
                    print('workAbs 3rd 4th',mindist(hmirr_hyp,hmirr_ell))
                    print('workAbs 4th fcs',mindist(hmirr_ell,detcenter))

                    print('1st W upper',np.linalg.norm(vmirr_hyp[:,0] - vmirr_hyp[:,ray_num-1]))
                    print('1st W lower',np.linalg.norm(vmirr_hyp[:,-1] - vmirr_hyp[:,-ray_num]))
                    print('2nd W upper',np.linalg.norm(vmirr_ell[:,0] - vmirr_ell[:,ray_num-1]))
                    print('2nd W lower',np.linalg.norm(vmirr_ell[:,-1] - vmirr_ell[:,-ray_num]))
                    print('3rd W lower',np.linalg.norm(hmirr_hyp[:,0] - hmirr_hyp[:,-ray_num]))
                    print('3rd W upper',np.linalg.norm(hmirr_hyp[:,ray_num-1] - hmirr_hyp[:,-1]))
                    print('4th W upper',np.linalg.norm(hmirr_ell[:,0] - hmirr_ell[:,-ray_num]))
                    print('4th W lower',np.linalg.norm(hmirr_ell[:,ray_num-1] - hmirr_ell[:,-1]))

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
                    plt.show()

                angle = reflect4

            if option == 'wave':
                print('diverg angle H',np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))
                print('diverg angle V',np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))
                # 全データからランダムに10%だけを選択
                sample_indices = np.random.choice(detcenter.shape[1], size=int(detcenter.shape[1]*0.001), replace=False)

                theta_y = -np.mean(np.arctan(angle[2, :]/angle[0, :]))
                theta_z = np.mean(np.arctan(angle[1, :]/angle[0, :]))
                source = np.zeros((3,1))
                if option_rotate==True:
                    reflect4_rotated = rotate_vectors(reflect4, -theta_y, -theta_z)
                    focus_apprx = np.mean(detcenter,axis=1)
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
                coeffs_det = np.zeros(10)
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

                if np.abs(defocusForWave) > 1e-9:
                    coeffs_det2 = np.zeros(10)
                    coeffs_det2[6] = 1
                    coeffs_det2[9] = -(s2f_middle + defocus+defocusForWave)
                    detcenter2 = plane_ray_intersection(coeffs_det2, reflect4_rotated, hmirr_hyp_points_rotated)
                    return source_rotated, vmirr_hyp_points_rotated_grid, hmirr_hyp_points_rotated_grid, vmirr_ell_points_rotated_grid, hmirr_ell_points_rotated_grid, detcenter, detcenter2, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vmirr2_norm, hmirr2_norm, vec0to1, vec1to2, vec2to3, vec3to4
                else:
                    return source_rotated, vmirr_hyp_points_rotated_grid, hmirr_hyp_points_rotated_grid, vmirr_ell_points_rotated_grid, hmirr_ell_points_rotated_grid, detcenter, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vmirr2_norm, hmirr2_norm, vec0to1, vec1to2, vec2to3, vec3to4

            
            hmirr_ell0 = hmirr_ell.copy()
            if option_tilt:

                if option == 'ray':
                    angles_between_rad, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect4,norm_vector(coeffs_det, detcenter))
                    theta_z = (np.max(angles_yx_rad) + np.min(angles_yx_rad))/2
                    theta_y = -(np.max(angles_zx_rad) + np.min(angles_zx_rad))/2
                    print('NA_h')
                    print(np.sin((np.max(angles_yx_rad) - np.min(angles_yx_rad))/2))
                    print('angles_yx_rad',np.sort(angles_yx_rad)[:5][::-1])
                    print('angles_yx_rad',np.sort(angles_yx_rad)[-5:][::-1])
                    print('NA_v')
                    print(np.sin((np.max(angles_zx_rad) - np.min(angles_zx_rad))/2))
                    print('angles_zx_rad',np.sort(angles_zx_rad)[:5][::-1])
                    print('angles_zx_rad',np.sort(angles_zx_rad)[-5:][::-1])
                    print('type(detcenter[0,0])',type(detcenter[0,0]))
                    print('theta_y',theta_y)
                    print('theta_z',theta_z)
                else:
                    theta_y = -np.mean(np.arctan(angle[2, :]/angle[0, :]))
                    # if option == 'ray':
                    #     plt.figure()
                    #     plt.plot(np.arctan(angle[2, :]/angle[0, :]))
                    #     plt.show()
                    theta_z = np.mean(np.arctan(angle[1, :]/angle[0, :]))

                reflect4_rotated = rotate_vectors(reflect4, -theta_y, -theta_z)
                focus_apprx = np.mean(detcenter,axis=1)
                hmirr_ell_points_rotated = rotate_points(hmirr_ell, focus_apprx, -theta_y, -theta_z)
                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)
                detcenter = plane_ray_intersection(coeffs_det, reflect4_rotated, hmirr_ell_points_rotated)


                hmirr_ell = hmirr_ell_points_rotated.copy()
                reflect4 = reflect4_rotated.copy()
                angle = reflect4.copy()


            if option == 'sep':
                focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2 = compare_sep(reflect4_rotated, hmirr_ell_points_rotated, coeffs_det,ray_num_H,1e-4)
                return focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2
            if option == 'sep_direct':
                focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2 = compare_sep(reflect4_rotated, hmirr_ell_points_rotated, coeffs_det,ray_num_H,1e-4)
                return focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2

            if option == 'ray_wave':
                if option_HighNA == True:
                    defocusWave = 1e-2
                    lambda_ = 13.5
                else:
                    defocusWave = 1e-3
                    lambda_ = 1.35
                coeffs_det2 = np.zeros(10)
                coeffs_det2[6] = 1
                coeffs_det2[9] = -(s2f_middle + defocus + defocusWave)
                detcenter2 = plane_ray_intersection(coeffs_det2, reflect4, hmirr_ell)

                dist4tofocus = np.linalg.norm(detcenter - hmirr_ell, axis=0)
                vector4tofocus = (detcenter - hmirr_ell) / dist4tofocus
                totalDist = dist0to1 + dist1to2 + dist2to3 + dist3to4 + dist4tofocus
                DistError = (totalDist - np.mean(totalDist))*1e9



                dist4tofocus2 = np.linalg.norm(detcenter2 - hmirr_ell, axis=0)
                vector4tofocus2 = (detcenter2 - hmirr_ell) / dist4tofocus2
                totalDist2 = dist0to1 + dist1to2 + dist2to3 + dist3to4 + dist4tofocus2
                DistError2 = (totalDist2 - np.mean(totalDist2))*1e9
                print('detcenter',np.mean(detcenter,axis=1))
                print('detcenter2',np.mean(detcenter2,axis=1))
                print('dist0to1',np.mean(dist0to1))
                print('dist1to2',np.mean(dist1to2))
                print('dist2to3',np.mean(dist2to3))
                print('dist3to4',np.mean(dist3to4))
                print('totalDist',np.mean(totalDist))
                print('dist4tofocus std',np.mean(dist4tofocus))
                print('dist0to1 std',np.std(dist0to1))
                print('dist1to2 std',np.std(dist1to2))
                print('dist2to3 std',np.std(dist2to3))
                print('dist3to4 std',np.std(dist3to4))
                print('dist4tofocus std',np.std(dist4tofocus))
                print('totalDist std',np.std(totalDist))
                print('np.std(totalDist)',np.std(totalDist))
                print('np.mean(totalDist)',np.mean(totalDist))
                print('np.mean(totalDist2)',np.mean(totalDist2))
                print('np.mean(totalDist2-totalDist)',np.mean(totalDist2-totalDist))
                print('np.std(totalDist2-totalDist)',np.std(totalDist2-totalDist))
                # 補間するグリッドを作成
                grid_H, grid_V = np.meshgrid(
                    np.linspace(detcenter2[1, :].min(), detcenter2[1, :].max(), ray_num_H),
                    np.linspace(detcenter2[2, :].min(), detcenter2[2, :].max(), ray_num_V)
                )

                CosAngle = angle[0,:]
                # グリッド上にデータを補間 (method: 'linear', 'nearest', 'cubic' から選択)
                if False:
                    matrixDistError2 = griddata((detcenter2[1, :], detcenter2[2, :]), DistError2, (grid_H, grid_V), method='cubic')
                    meanFocus = np.mean(detcenter,axis=1)
                    Sph = np.linalg.norm(detcenter2 - meanFocus[:, np.newaxis], axis=0) * 1e9
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
                    print('np.mean(DistError2)',np.mean(DistError2))
                    print('np.std(DistError2)',np.std(DistError2))
                    print('np.mean(Sph)',np.mean(Sph))
                    print('np.std(Sph)',np.std(Sph))
                    print('np.mean(Wave2)',np.mean(Wave2))
                    print('np.std(Wave2)',np.std(Wave2))
                    print('grid_H.shape',grid_H.shape)
                    print('Wave2.shape',Wave2.shape)
                    print('detcenter2.shape',detcenter2.shape)

                    matrixWave2 = griddata((detcenter2[1, :], detcenter2[2, :]), Wave2, (grid_H, grid_V), method='cubic')
                    matrixWave2 = matrixWave2 - np.nanmean(matrixWave2)

                np.savetxt('matrixWave2(nm).txt',matrixWave2)
                tifffile.imwrite('matrixWave2(nm).tiff', matrixWave2)

                matrixWave2_Corrected = plane_correction_with_nan_and_outlier_filter(matrixWave2)
                matrixDistError2_Corrected = plane_correction_with_nan_and_outlier_filter(matrixDistError2)
                print('PV',np.nanmax(matrixWave2_Corrected)-np.nanmin(matrixWave2_Corrected))
                
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
                # plt.show()
                # rectified_img = extract_affine_square_region(matrixWave2_Corrected/lambda_, target_size=256)
                rectified_img = extract_affine_square_region(matrixWave2/lambda_, target_size=256)

                # plt.figure()
                # plt.imshow(rectified_img[1:-2, 1:-2], cmap='jet',vmin = -1/4,vmax = 1/4)
                # plt.colorbar(label='\u03BB')
                # plt.title("Affine-Corrected Square Cutout")
                # # plt.show()
                

                assesorder = 5
                fit_datas, inner_products, orders = lf.match_legendre_multi(rectified_img[1:-2, 1:-2], assesorder)
                length = len(inner_products)
                fig, axes = plt.subplots(assesorder, assesorder, figsize=(16, 16))
                for i in range(length):
                    ny = orders[i][0]
                    nx = orders[i][1]
                    print(f"ny: {ny}, nx: {nx}, Inner Product: {inner_products[i]:.3e}")
                    axes[ny, nx].imshow(fit_datas[i], cmap='jet', vmin=-1/4, vmax=1/4)
                    axes[ny, nx].set_title(f"ny: {ny}, nx: {nx}, Inner Product: {inner_products[i]:.3e}")
                    ### set colorbar for each subplot
                    # cbar = plt.colorbar(axes[ny, nx].images[0], ax=axes[ny, nx], fraction=0.046, pad=0.04)
                axes[-1, -1].imshow(rectified_img[1:-2, 1:-2], cmap='jet', vmin=-1/4, vmax=1/4)
                cbar = plt.colorbar(axes[-1, -1].images[0], ax=axes[-1, -1], fraction=0.046, pad=0.04)
                fit_sum = np.sum(fit_datas, axis=0)
                axes[-2, -1].imshow(fit_sum, cmap='jet', vmin=-1/4, vmax=1/4)
                # cbar = plt.colorbar(axes[-2, -1].images[0], ax=axes[-2, -1], fraction=0.046, pad=0.04)
                axes[-1, -2].imshow(rectified_img[1:-2, 1:-2]-fit_sum, cmap='jet', vmin=-1/4, vmax=1/4)
                # cbar = plt.colorbar(axes[-1, -2].images[0], ax=axes[-1, -2], fraction=0.046, pad=0.04)

                plt.tight_layout()
                if option_legendre:
                    plt.close()
                    print('inner_products',inner_products)
                    print('orders',orders)
                    return inner_products, orders
                else:
                    plt.show()



                if False:
                    # psf, x_out, y_out = fresnel_psf(matrixWave2_Corrected, lambda_=lambda_, z=-defocusWave, grid_x=grid_H, grid_y=grid_V)
                    calcrange=1.e-6
                    psf, x_out, y_out = fresnel_integral(
                        phi=matrixWave2_Corrected*1e-9,
                        grid_x=grid_H-np.mean(grid_H),
                        grid_y=grid_V-np.mean(grid_V),
                        lambda_=lambda_*1e-9,
                        z=-defocusWave,
                        x_out_range=(-calcrange, calcrange),
                        y_out_range=(-calcrange, calcrange),
                        dx_out=calcrange/65,
                        dy_out=calcrange/65,
                    )

                    def compute_fwhm(x, intensity_1d):
                        """1次元の強度分布から FWHM を計算"""
                        dx = np.abs(x[1] - x[0])
                        num_over_half_max = np.sum(intensity_1d >= 0.5 * np.max(intensity_1d))
                        fwhm = (num_over_half_max-1) * dx
                        return fwhm
                    psf_x = psf[psf.shape[0] // 2, :]
                    psf_y = psf[:, psf.shape[1] // 2]
                    half_max = 0.5 * np.max(psf)
                    mask = psf >= half_max
                    dx = np.abs(x_out[1] - x_out[0])
                    dy = np.abs(y_out[1] - y_out[0])
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
                print(np.cos(theta1_v)*s2f_middle)
                print(theta1_h)
                print(np.mean(detcenter[0,:]))
                print(np.mean(detcenter[1,:]))
                print(np.mean(detcenter[2,:]))
                # print(np.mean(detcenter[0,:]))
                print(coeffs_det)
                print('s2f_H',s2f_H)
                print('s2f_V',s2f_V)
                mabiki = round(np.sqrt(ray_num_H*ray_num_V)/50)
                mabiki =  1
                defocussize = 4e-5
                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus) + defocussize
                detcenter1 = plane_ray_intersection(coeffs_det, reflect4, hmirr_ell)

                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus) - defocussize
                detcenter2 = plane_ray_intersection(coeffs_det, reflect4, hmirr_ell)
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[::ray_num+1]
                thinned_array = first_thinned_array[::mabiki]

                print('Oblique1',thinned_array)

                obl_1 = (detcenter1[1,thinned_array] + detcenter1[2,thinned_array])/np.sqrt(2)
                obl_2 = (detcenter2[1,thinned_array] + detcenter2[2,thinned_array])/np.sqrt(2)
                axs[0,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[obl_1,obl_2],c='y')
                axs[0,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(obl_2),np.max(obl_1)],color='k')
                axs[0,0].set_title('Oblique1 aperture 0')
                axs[0,0].set_xlabel('Axial (m)')
                axs[0,0].set_ylabel('Oblique1 Position (m)')

                obl_1 = (-detcenter1[1,thinned_array] + detcenter1[2,thinned_array])/np.sqrt(2)
                obl_2 = (-detcenter2[1,thinned_array] + detcenter2[2,thinned_array])/np.sqrt(2)
                axs[0,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[obl_1,obl_2],c='y')
                axs[0,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(obl_2),np.max(obl_1)],color='k')
                axs[0,1].set_title('Oblique1 aperture 0')
                axs[0,1].set_xlabel('Axial (m)')
                axs[0,1].set_ylabel('Oblique2 Position (m)')

                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[ray_num-1::ray_num-1][:-1]
                # first_thinned_array = first_thinned_array[:-1]
                thinned_array = first_thinned_array[::mabiki]

                print('Oblique2',thinned_array)

                obl_1 = (detcenter1[1,thinned_array] + detcenter1[2,thinned_array])/np.sqrt(2)
                obl_2 = (detcenter2[1,thinned_array] + detcenter2[2,thinned_array])/np.sqrt(2)
                axs[1,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[obl_1,obl_2],c='y')
                axs[1,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(obl_2),np.max(obl_1)],color='k')
                axs[1,0].set_title('Oblique2 aperture 0')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Oblique1 Position (m)')

                obl_1 = (-detcenter1[1,thinned_array] + detcenter1[2,thinned_array])/np.sqrt(2)
                obl_2 = (-detcenter2[1,thinned_array] + detcenter2[2,thinned_array])/np.sqrt(2)
                axs[1,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[obl_1,obl_2],c='y')
                axs[1,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(obl_2),np.max(obl_1)],color='k')
                axs[1,1].set_title('Oblique2 aperture 0')
                axs[1,1].set_xlabel('Axial (m)')
                axs[1,1].set_ylabel('Oblique2 Position (m)')
                plt.savefig('multiple_plots_ray_oblique.png', dpi=300)
                # plt.show()
                plt.close()
                fig, axs = plt.subplots(3, 2, figsize=(10, 15))
                # 範囲内の値を間引く
                start = 0
                end = ray_num
                thinned_array = crop(start, end, mabiki)



                axs[0,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='r')
                axs[0,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[0,0].set_title('Ray from V')
                axs[0,0].set_xlabel('Axial (m)')
                axs[0,0].set_ylabel('Horizontal Position (m)')

                axs[0,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='r')
                axs[0,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[0,1].set_title('Ray from H')
                axs[0,1].set_xlabel('Axial (m)')
                axs[0,1].set_ylabel('Vertical Position (m)')
                # plt.show()

                # 範囲内の値を間引く
                start = round(ray_num*(ray_num-1)/2)
                end = round(ray_num*(ray_num+1)/2)
                thinned_array = crop(start, end, mabiki)

                axs[1,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='y')
                axs[1,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[1,0].set_title('Ray from V')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Horizontal Position (m)')

                axs[1,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='y')
                axs[1,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[1,1].set_title('Ray from H')
                axs[1,1].set_xlabel('Axial (m)')
                axs[1,1].set_ylabel('Vertical Position (m)')

                # 範囲内の値を間引く
                start = ray_num**2 - ray_num
                end = ray_num**2
                thinned_array = crop(start, end, mabiki)

                axs[2,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='g')
                axs[2,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[2,0].set_title('Ray from V')
                axs[2,0].set_xlabel('Axial (m)')
                axs[2,0].set_ylabel('Horizontal Position (m)')

                axs[2,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='g')
                axs[2,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[2,1].set_title('Ray from H')
                axs[2,1].set_xlabel('Axial (m)')
                axs[2,1].set_ylabel('Vertical Position (m)')
                fig.suptitle('V aperture 0')
                plt.savefig('multiple_plots_ray_v.png', dpi=300)
                # plt.show()
                plt.close()


                fig, axs = plt.subplots(3, 2, figsize=(10, 15))
                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[::ray_num]
                thinned_array = first_thinned_array[::mabiki]
                fig.suptitle('H aperture 0')

                axs[0,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='r')
                axs[0,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[0,0].set_title('Ray from V')
                axs[0,0].set_xlabel('Axial (m)')
                axs[0,0].set_ylabel('Horizontal Position (m)')

                axs[0,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='r')
                axs[0,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[0,1].set_title('Ray from H')
                axs[0,1].set_xlabel('Axial (m)')
                axs[0,1].set_ylabel('Vertical Position (m)')


                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[round((ray_num-1)/2)::ray_num]

                # さらに間引き (skip_rateでさらに間引く)
                thinned_array = first_thinned_array[::mabiki]

                axs[1,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='y')
                axs[1,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[1,0].set_title('Ray from V')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Horizontal Position (m)')

                axs[1,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='y')
                axs[1,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[1,1].set_title('Ray from H')
                axs[1,1].set_xlabel('Axial (m)')
                axs[1,1].set_ylabel('Vertical Position (m)')

                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                # 最初の間引き (ray_num-1から始めてray_numごとに要素を取得)
                first_thinned_array = original_array[ray_num-1::ray_num]

                # さらに間引き (skip_rateでさらに間引く)
                thinned_array = first_thinned_array[::mabiki]

                axs[2,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='g')
                axs[2,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[2,0].set_title('Ray from V')
                axs[2,0].set_xlabel('Axial (m)')
                axs[2,0].set_ylabel('Horizontal Position (m)')

                axs[2,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='g')
                axs[2,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[2,1].set_title('Ray from H')
                axs[2,1].set_xlabel('Axial (m)')
                axs[2,1].set_ylabel('Vertical Position (m)')
                plt.savefig('multiple_plots_ray_h.png', dpi=300)
                # plt.show()
                plt.close()

                # # # plot_ray_sideview(8,10,mabiki,reflect1,vmirr_hyp,ray_num)
                # plot_ray_sideview(-5,35,mabiki,reflect2,vmirr_ell,ray_num)
                # # # plot_ray_sideview(8,10,mabiki,reflect3,vmirr_ell,ray_num)
                # # # plot_ray_sideview(0.2,0.2,mabiki,reflect3,vmirr_ell,ray_num)

                phai0 = normalize_vector(phai0)

                plt.figure()
                plt.scatter(phai0[1, :], phai0[2, :])
                plt.scatter(phai0[1, ::ray_num], phai0[2, ::ray_num],color='r')
                plt.scatter(phai0[1, round((ray_num-1)/2)::ray_num], phai0[2, round((ray_num-1)/2)::ray_num],color='y')
                plt.scatter(phai0[1, ray_num-1::ray_num], phai0[2, ray_num-1::ray_num],color='g')
                plt.title('angle')
                plt.xlabel('Horizontal Angle (rad)')
                plt.ylabel('Vertical Angle (rad)')
                plt.axis('equal')
                # plt.show()
                plt.close()

                plt.figure()
                plt.scatter(phai0[1, :], phai0[2, :])
                plt.scatter((phai0[1, :ray_num]), (phai0[2, :ray_num]),color='r')
                plt.scatter(phai0[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)], phai0[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)],color='y')
                plt.scatter((phai0[1, -ray_num:]), (phai0[2, -ray_num:]),color='g')
                plt.title('angle')
                plt.xlabel('Horizontal Angle (rad)')
                plt.ylabel('Vertical Angle (rad)')
                plt.axis('equal')
                # plt.show()
                plt.close()

                vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
                reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))

                angle_1st, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect1,norm_vector(coeffs_hyp_v, vmirr_hyp))



                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)

                detcenter0 = plane_ray_intersection(coeffs_det, reflect4, hmirr_ell)

                # angles_between_rad, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect_ray(reflect4, norm_vector(coeffs_det, detcenter0)),norm_vector(coeffs_det, detcenter0))
                angles_between_rad, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect4,norm_vector(coeffs_det, detcenter0))
                print('NA_h')
                print(np.sin((np.max(angles_yx_rad) - np.min(angles_yx_rad))/2))
                print('angles_yx_rad',np.sort(angles_yx_rad)[:5][::-1])
                print('angles_yx_rad',np.sort(angles_yx_rad)[-5:][::-1])
                print('NA_v')
                print(np.sin((np.max(angles_zx_rad) - np.min(angles_zx_rad))/2))
                print('angles_zx_rad',np.sort(angles_zx_rad)[:5][::-1])
                print('angles_zx_rad',np.sort(angles_zx_rad)[-5:][::-1])
                print('type(detcenter[0,0])',type(detcenter[0,0]))
                if option_tilt:
                    reflect4_rotated = rotate_vectors(reflect4, -theta_y, -theta_z)
                    focus_apprx = np.mean(detcenter,axis=1)
                    hmirr_ell_points_rotated = rotate_points(hmirr_ell, focus_apprx, -theta_y, -theta_z)
                    coeffs_det = np.zeros(10)
                    coeffs_det[6] = 1
                    coeffs_det[9] = -(s2f_middle + defocus)
                    detcenter = plane_ray_intersection(coeffs_det, reflect4_rotated, hmirr_ell_points_rotated)
                    hmirr_ell = hmirr_ell_points_rotated
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
                axs[0, 0].plot([detcenter1[0, thinned_array_h_r[:third_r]], detcenter2[0, thinned_array_h_r[:third_r]]],
                               [detcenter1[1, thinned_array_h_r[:third_r]], detcenter2[1, thinned_array_h_r[:third_r]]], c='r')

                # 後ろ1/3のプロット（ピンク）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_r[-third_r:]], detcenter2[0, thinned_array_h_r[-third_r:]]],
                               [detcenter1[1, thinned_array_h_r[-third_r:]], detcenter2[1, thinned_array_h_r[-third_r:]]], c='purple')

                # thinned_array_h_y の前1/3のプロット（darkyellow）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_y[:third_y]], detcenter2[0, thinned_array_h_y[:third_y]]],
                               [detcenter1[1, thinned_array_h_y[:third_y]], detcenter2[1, thinned_array_h_y[:third_y]]], c='y')

                # thinned_array_h_y の後ろ1/3のプロット（purple）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_y[-third_y:]], detcenter2[0, thinned_array_h_y[-third_y:]]],
                               [detcenter1[1, thinned_array_h_y[-third_y:]], detcenter2[1, thinned_array_h_y[-third_y:]]], c='#B8860B')

                # thinned_array_h_g の前1/3のプロット（緑）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_g[:third_g]], detcenter2[0, thinned_array_h_g[:third_g]]],
                               [detcenter1[1, thinned_array_h_g[:third_g]], detcenter2[1, thinned_array_h_g[:third_g]]], c='g')

                # thinned_array_h_g の後ろ1/3のプロット（薄緑）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_g[-third_g:]], detcenter2[0, thinned_array_h_g[-third_g:]]],
                               [detcenter1[1, thinned_array_h_g[-third_g:]], detcenter2[1, thinned_array_h_g[-third_g:]]], c='lightgreen')
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
                axs[1,1].scatter(np.mean(detcenter[1, ::ray_num]), np.mean(detcenter[2, ::ray_num]),color='r',marker='x',s=100)
                axs[1,1].scatter(np.mean(detcenter[1, round((ray_num-1)/2)::ray_num]), np.mean(detcenter[2, round((ray_num-1)/2)::ray_num]),color='y',marker='x',s=100)
                axs[1,1].scatter(np.mean(detcenter[1, ray_num-1::ray_num]), np.mean(detcenter[2, ray_num-1::ray_num]),color='g',marker='x',s=100)
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
                axs[1, 0].plot([detcenter1[0, thinned_array_v_r[:third_r]], detcenter2[0, thinned_array_v_r[:third_r]]],
                               [detcenter1[2, thinned_array_v_r[:third_r]], detcenter2[2, thinned_array_v_r[:third_r]]], c='r')

                # 後ろ1/3のプロット（ピンク）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_r[-third_r:]], detcenter2[0, thinned_array_v_r[-third_r:]]],
                               [detcenter1[2, thinned_array_v_r[-third_r:]], detcenter2[2, thinned_array_v_r[-third_r:]]], c='purple')

                # thinned_array_v_y の前1/3のプロット（darkyellow）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_y[:third_y]], detcenter2[0, thinned_array_v_y[:third_y]]],
                               [detcenter1[2, thinned_array_v_y[:third_y]], detcenter2[2, thinned_array_v_y[:third_y]]], c='y')

                # thinned_array_v_y の後ろ1/3のプロット（purple）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_y[-third_y:]], detcenter2[0, thinned_array_v_y[-third_y:]]],
                               [detcenter1[2, thinned_array_v_y[-third_y:]], detcenter2[2, thinned_array_v_y[-third_y:]]], c='#B8860B')

                # thinned_array_v_g の前1/3のプロット（緑）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_g[:third_g]], detcenter2[0, thinned_array_v_g[:third_g]]],
                               [detcenter1[2, thinned_array_v_g[:third_g]], detcenter2[2, thinned_array_v_g[:third_g]]], c='g')

                # thinned_array_v_g の後ろ1/3のプロット（薄緑）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_g[-third_g:]], detcenter2[0, thinned_array_v_g[-third_g:]]],
                               [detcenter1[2, thinned_array_v_g[-third_g:]], detcenter2[2, thinned_array_v_g[-third_g:]]], c='lightgreen')

                axs[1,0].plot([input_val, input_val],
                            [np.min(detcenter2[2, :]), np.max(detcenter1[2, :])], color='k')
                axs[1,0].set_title('H aperture 0')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Vertical Position (m)')

                axs[0,1].scatter(detcenter[1, :], detcenter[2, :])
                axs[0,1].scatter(detcenter[1, :ray_num], detcenter[2, :ray_num],color='r')
                axs[0,1].scatter(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)], detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)],color='y')
                axs[0,1].scatter(detcenter[1, -ray_num:], detcenter[2, -ray_num:],color='g')
                axs[0,1].scatter(detcenter[1, ray_num-1::ray_num-1][:-1], detcenter[2, ray_num-1::ray_num-1][:-1],color='k')
                axs[0,1].scatter(detcenter[1, ::ray_num+1], detcenter[2, ::ray_num+1],color='gray')
                axs[0,1].scatter(np.mean(detcenter[1, :ray_num]), np.mean(detcenter[2, :ray_num]),color='r',marker='x',s=100)
                axs[0,1].scatter(np.mean(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]), np.mean(detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]),color='y',marker='x',s=100)
                axs[0,1].scatter(np.mean(detcenter[1, -ray_num:]), np.mean(detcenter[2, -ray_num:]),color='g',marker='x',s=100)
                axs[0,1].set_title('focus @V aperture 0')
                axs[0,1].set_xlabel('Horizontal (m)')
                axs[0,1].set_ylabel('Vertical (m)')
                axs[0,1].axis('equal')

                # タイトル用の新しいサイズ計算
                size_v = np.max(detcenter[2,:]) - np.min(detcenter[2,:])
                size_h = np.max(detcenter[1,:]) - np.min(detcenter[1,:])

                # タイトルの更新
                title1 = f'Params 0-1: {params[0:2]}'
                title2 = f'Params 2-7: {params[2:8]}'
                title3 = f'Params 8-13: {params[8:14]}'
                title4 = f'Params 14-19: {params[14:20]}'
                title5 = f'Params 20-25: {params[20:26]}'
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
                        detcenter = plane_ray_intersection(coeffs_det, reflect4, hmirr_ell)
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
                        axs[1,1].scatter(np.mean(detcenter[1, ::ray_num]), np.mean(detcenter[2, ::ray_num]),color='r',marker='x',s=100)
                        axs[1,1].scatter(np.mean(detcenter[1, round((ray_num-1)/2)::ray_num]), np.mean(detcenter[2, round((ray_num-1)/2)::ray_num]),color='y',marker='x',s=100)
                        axs[1,1].scatter(np.mean(detcenter[1, ray_num-1::ray_num]), np.mean(detcenter[2, ray_num-1::ray_num]),color='g',marker='x',s=100)
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
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_r[:third_r]], detcenter2[0, thinned_array_h_r[:third_r]]],
                                       [detcenter1[1, thinned_array_h_r[:third_r]], detcenter2[1, thinned_array_h_r[:third_r]]], c='r')

                        # 後ろ1/3のプロット（ピンク）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_r[-third_r:]], detcenter2[0, thinned_array_h_r[-third_r:]]],
                                       [detcenter1[1, thinned_array_h_r[-third_r:]], detcenter2[1, thinned_array_h_r[-third_r:]]], c='purple')

                        # thinned_array_h_y の前1/3のプロット（darkyellow）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_y[:third_y]], detcenter2[0, thinned_array_h_y[:third_y]]],
                                       [detcenter1[1, thinned_array_h_y[:third_y]], detcenter2[1, thinned_array_h_y[:third_y]]], c='y')

                        # thinned_array_h_y の後ろ1/3のプロット（purple）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_y[-third_y:]], detcenter2[0, thinned_array_h_y[-third_y:]]],
                                       [detcenter1[1, thinned_array_h_y[-third_y:]], detcenter2[1, thinned_array_h_y[-third_y:]]], c='#B8860B')

                        # thinned_array_h_g の前1/3のプロット（緑）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_g[:third_g]], detcenter2[0, thinned_array_h_g[:third_g]]],
                                       [detcenter1[1, thinned_array_h_g[:third_g]], detcenter2[1, thinned_array_h_g[:third_g]]], c='g')

                        # thinned_array_h_g の後ろ1/3のプロット（薄緑）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_g[-third_g:]], detcenter2[0, thinned_array_h_g[-third_g:]]],
                                       [detcenter1[1, thinned_array_h_g[-third_g:]], detcenter2[1, thinned_array_h_g[-third_g:]]], c='lightgreen')
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
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_r[:third_r]], detcenter2[0, thinned_array_v_r[:third_r]]],
                                       [detcenter1[2, thinned_array_v_r[:third_r]], detcenter2[2, thinned_array_v_r[:third_r]]], c='r')

                        # 後ろ1/3のプロット（ピンク）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_r[-third_r:]], detcenter2[0, thinned_array_v_r[-third_r:]]],
                                       [detcenter1[2, thinned_array_v_r[-third_r:]], detcenter2[2, thinned_array_v_r[-third_r:]]], c='purple')

                        # thinned_array_v_y の前1/3のプロット（darkyellow）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_y[:third_y]], detcenter2[0, thinned_array_v_y[:third_y]]],
                                       [detcenter1[2, thinned_array_v_y[:third_y]], detcenter2[2, thinned_array_v_y[:third_y]]], c='y')

                        # thinned_array_v_y の後ろ1/3のプロット（purple）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_y[-third_y:]], detcenter2[0, thinned_array_v_y[-third_y:]]],
                                       [detcenter1[2, thinned_array_v_y[-third_y:]], detcenter2[2, thinned_array_v_y[-third_y:]]], c='#B8860B')

                        # thinned_array_v_g の前1/3のプロット（緑）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_g[:third_g]], detcenter2[0, thinned_array_v_g[:third_g]]],
                                       [detcenter1[2, thinned_array_v_g[:third_g]], detcenter2[2, thinned_array_v_g[:third_g]]], c='g')

                        # thinned_array_v_g の後ろ1/3のプロット（薄緑）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_g[-third_g:]], detcenter2[0, thinned_array_v_g[-third_g:]]],
                                       [detcenter1[2, thinned_array_v_g[-third_g:]], detcenter2[2, thinned_array_v_g[-third_g:]]], c='lightgreen')

                        axs[1,0].plot([input_val, input_val],
                                    [np.min(detcenter2[2, :]), np.max(detcenter1[2, :])], color='k')
                        axs[1,0].set_title('H aperture 0')
                        axs[1,0].set_xlabel('Axial (m)')
                        axs[1,0].set_ylabel('Vertical Position (m)')

                        axs[0,1].cla()  # 右側プロットをクリア
                        axs[0,1].scatter(detcenter[1, :], detcenter[2, :])
                        axs[0,1].scatter(detcenter[1, :ray_num], detcenter[2, :ray_num],color='r')
                        axs[0,1].scatter(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)], detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)],color='y')
                        axs[0,1].scatter(detcenter[1, -ray_num:], detcenter[2, -ray_num:],color='g')
                        axs[0,1].scatter(detcenter[1, ray_num-1::ray_num-1][:-1], detcenter[2, ray_num-1::ray_num-1][:-1],color='k')
                        axs[0,1].scatter(detcenter[1, ::ray_num+1], detcenter[2, ::ray_num+1],color='gray')
                        axs[0,1].scatter(np.mean(detcenter[1, :ray_num]), np.mean(detcenter[2, :ray_num]),color='r',marker='x',s=100)
                        axs[0,1].scatter(np.mean(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]), np.mean(detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]),color='y',marker='x',s=100)
                        axs[0,1].scatter(np.mean(detcenter[1, -ray_num:]), np.mean(detcenter[2, -ray_num:]),color='g',marker='x',s=100)
                        axs[0,1].set_title('focus @V aperture 0')
                        axs[0,1].set_xlabel('Horizontal (m)')
                        axs[0,1].set_ylabel('Vertical (m)')
                        axs[0,1].axis('equal')

                        # axs[0,2].scatter(input_val,np.mean(detcenter[1, :ray_num]),color='r')
                        # axs[0,2].scatter(input_val,np.mean(detcenter[1, round((ray_num**2)/2) : round((ray_num**2 + ray_num*2)/2)]),color='y')
                        # axs[0,2].scatter(input_val,np.mean(detcenter[1, -ray_num:-1]),color='g')

                        # axs[1,2].scatter(input_val,np.mean(detcenter[2, ::ray_num]),color='r')
                        # axs[1,2].scatter(input_val,np.mean(detcenter[2, round(ray_num/2)-1::ray_num]),color='y')
                        # axs[1,2].scatter(input_val,np.mean(detcenter[2, ray_num-1::ray_num]),color='g')
                        # axs[1,3].scatter(input_val,np.mean(detcenter[1, ::ray_num]),color='r')
                        # axs[1,3].scatter(input_val,np.mean(detcenter[1, round(ray_num/2)-1::ray_num]),color='y')
                        # axs[1,3].scatter(input_val,np.mean(detcenter[1, ray_num-1::ray_num]),color='g')

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
                plt.savefig('multipleAroundFocus.png', dpi=300)
                plt.show()





    # 焦点面での標準偏差を計算
        return vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle
    
else:
    def plot_result_debug(params,option):
        defocus, astigH, \
        pitch_hyp_v, roll_hyp_v, yaw_hyp_v, decenterX_hyp_v, decenterY_hyp_v, decenterZ_hyp_v,\
        pitch_hyp_h, roll_hyp_h, yaw_hyp_h, decenterX_hyp_h, decenterY_hyp_h, decenterZ_hyp_h,\
        pitch_ell_v, roll_ell_v, yaw_ell_v, decenterX_ell_v, decenterY_ell_v, decenterZ_ell_v,\
        pitch_ell_h, roll_ell_h, yaw_ell_h, decenterX_ell_h, decenterY_ell_h, decenterZ_ell_h  = params
        if option_HighNA:
            # Mirror parameters EUV
            # a_hyp_v = np.float64(72.96002945938)
            # b_hyp_v = np.float64(0.134829747201017)
            # a_ell_v = np.float64(0.442)
            # b_ell_v = np.float64(0.0607128830733533)
            # length_hyp_v = np.float64(0.115)
            # length_ell_v = np.float64(0.229790269646258)
            # theta1_v = np.float64(4.73536529533549E-05)

            # a_hyp_h = np.float64(73.018730871665)
            # b_hyp_h = np.float64(0.0970536727319812)
            # a_ell_h = np.float64(0.38125)
            # b_ell_h = np.float64(0.0397791317992322)
            # length_hyp_h = np.float64(0.25)
            # length_ell_h = np.float64(0.0653872838592807)
            # theta1_h = np.float64(5.6880350884129E-05)


            a_hyp_v = np.float64(72.968)
            b_hyp_v = np.float64(0.250376439249605)
            a_ell_v = np.float64(0.2195)
            b_ell_v = np.float64(0.0512440210953413)
            length_hyp_v = np.float64(0.061)
            length_ell_v = np.float64(0.103058980190576)
            theta1_v = np.float64(7.15995481547383E-055)


            a_hyp_h = np.float64(72.985)
            b_hyp_h = np.float64(0.183699830190467)
            a_ell_h = np.float64(0.203)
            b_ell_h = np.float64(0.0354925652013309)
            length_hyp_h = np.float64(0.13)
            length_ell_h = np.float64(0.0500546018611444)
            theta1_h = np.float64(8.32243535501736E-05)


        else:
            # Mirror parameters hardX
            a_hyp_v = np.float64(72.952)
            b_hyp_v = np.float64(0.014226386294077)
            a_ell_v = np.float64(0.448)
            b_ell_v = np.float64(0.00679549637250505)
            length_hyp_v = np.float64(0.115)
            length_ell_v = np.float64(0.223373626775487)
            theta1_v = np.float64(5.00050007220147E-06)

            a_hyp_h = np.float64(73.018730871665)
            b_hyp_h = np.float64(0.012210459376815)
            a_ell_h = np.float64(0.38125)
            b_ell_h = np.float64(0.00494315702001789)
            length_hyp_h = np.float64(0.25)
            length_ell_h = np.float64(0.0663194129478278)
            theta1_h = np.float64(7.15704945387313E-06)


        org_hyp_v = np.sqrt(a_hyp_v**2 + b_hyp_v**2)
        org_hyp_h = np.sqrt(a_hyp_h**2 + b_hyp_h**2)

        org_ell_v = np.sqrt(a_ell_v**2 - b_ell_v**2)
        org_ell_h = np.sqrt(a_ell_h**2 - b_ell_h**2)

        # astig_v = (org_hyp_v - org_hyp_h)/2
        # astig_v_ = (org_hyp_v - org_hyp_h)/2*np.linspace(0,4,10)
        n = 20
        # param_ = np.linspace(-1,1,n)
        std_v = np.full(n, np.nan)
        std_h = np.full(n, np.nan)


        # astig_v = astig_v_[j]
        # astig_v = 0.5*-0.16626315789473686
        # astig_v = 0.5
        # print(astig_v_)

        # Input parameters
        ray_num = 53
        # defocus = 0.2*0.03157894736842107
        # defocus = 0
        # print(defocus)
        # astig_v = 0
        # pitch_hyp_h = 0
        # roll_hyp_h = 0
        # yaw_hyp_h = 0
        # pitch_ell_h = 0
        # roll_ell_h = 0
        # yaw_ell_h = 0
        # pitch_hyp_v = 0
        # roll_hyp_v = 0
        # yaw_hyp_v = 0.1*param_[j]
        # pitch_ell_v = 0
        # roll_ell_v = 0
        # yaw_ell_v = 0.1*param_[j]
        # option_2mirror = True
        # Input parameters
        ray_num_H = 53
        ray_num_V = 53
        ray_num = 53
        if option == 'ray':
            ray_num_H = 53
            ray_num_V = 53
            # ray_num = 1
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
        c_v = np.zeros(10)
        c_v[0] = 1 / a_hyp_v**2
        c_v[2] = -1 / b_hyp_v**2
        c_v[9] = -1.
        org_v = np.sqrt(a_hyp_v**2 + b_hyp_v**2)
        c_v = shift_x(c_v, org_v)

        center_v = mirr_ray_intersection(c_v, np.array([[np.cos(theta1_v)], [0.], [np.sin(theta1_v)]]), np.array([[0.], [0.], [0.]]))
        if not np.isreal(center_v).all():
            return np.inf
        x1_v = center_v[0, 0] - length_hyp_v / 2
        x2_v = center_v[0, 0] + length_hyp_v / 2
        # y1_v = np.sqrt((-1 + (x1_v / a_hyp_v)**2) * b_hyp_v**2)
        # y2_v = np.sqrt((-1 + (x2_v / a_hyp_v)**2) * b_hyp_v**2)
        y1_v = calc_Y_hyp(a_hyp_v, b_hyp_v, x1_v)
        y2_v = calc_Y_hyp(a_hyp_v, b_hyp_v, x2_v)
        # print(np.arctan(y1_v/x1_v))
        # print(np.arctan(y2_v/x2_v))
        accept_v = np.abs(y2_v - y1_v)
        l1v = np.linalg.norm(c_v)

        c_h = np.zeros(10)
        c_h[0] = 1 / a_hyp_h**2
        c_h[1] = -1 / b_hyp_h**2
        c_h[9] = -1.
        org_h = np.sqrt(a_hyp_h**2 + b_hyp_h**2)
        c_h = shift_x(c_h, org_h)
        center_h = mirr_ray_intersection(c_h, np.array([[np.cos(theta1_h)], [np.sin(theta1_h)], [0.]]), np.array([[0.], [0.], [0.]]))
        if not np.isreal(center_h).all():
            return np.inf
        x1_h = center_h[0, 0] - length_hyp_h / 2
        x2_h = center_h[0, 0] + length_hyp_h / 2
        # y1_h = np.sqrt((-1 + (x1_h / a_hyp_h)**2) * b_hyp_h**2)
        # y2_h = np.sqrt((-1 + (x2_h / a_hyp_h)**2) * b_hyp_h**2)
        y1_h = calc_Y_hyp(a_hyp_h, b_hyp_h, x1_h)
        y2_h = calc_Y_hyp(a_hyp_h, b_hyp_h, x2_h)
        # print(np.arctan(y1_h/x1_h))
        # print(np.arctan(y2_h/x2_h))
        accept_h = np.abs(y2_h - y1_h)
        l1h = np.linalg.norm(c_h)

        # Raytrace (X = x-ray direction)

        # V hyp mirror set (1st)
        axis_x = np.array([1., 0., 0.])
        axis_y = np.array([0., 1., 0.])
        axis_z = np.array([0., 0., 1.])
        coeffs_hyp_v = np.zeros(10)
        coeffs_hyp_v[0] = 1 / a_hyp_v**2
        coeffs_hyp_v[2] = -1 / b_hyp_v**2
        coeffs_hyp_v[9] = -1.
        coeffs_hyp_v = shift_x(coeffs_hyp_v, org_hyp_v)
        if option_axial:
            # coeffs_hyp_v = rotate_y(coeffs_hyp_v, theta1_v, [0, 0, 0])
            coeffs_hyp_v, rotation_matrix = rotate_general_axis(coeffs_hyp_v, axis_y, theta1_v, [0, 0, 0])
            axis_x, axis_y, axis_z = rotatematrix(rotation_matrix, axis_x, axis_y, axis_z)

        if option_alignment and option_axial:
            bufray = np.zeros((3, 5))
            ### 4隅の光線
            theta_cntr_h = (np.arctan(y2_h / x2_h) + np.arctan(y1_h / x1_h))/2.
            theta_cntr_v = (np.arctan(y2_v / x2_v) + np.arctan(y1_v / x1_v))/2.
            if option == 'ray':

                def print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h):
                    l2_h = (4*a_hyp_h**2 + (org_hyp_h*2)**2 -4*a_hyp_h*(org_hyp_h*2)*np.cos(theta1_h))/(4*org_hyp_h - 4*a_hyp_h)
                    # l2_v = (4*a_hyp_v**2 + (org_hyp_v*2)**2 -4*a_hyp_v*(org_hyp_v*2)*np.cos(theta1_v))/(4*org_hyp_v - 4*a_hyp_v)

                    l1_h = 2*a_hyp_h +l2_h
                    # l1_v = 2*a_hyp_v +l2_v

                    theta2_h = np.arcsin(org_hyp_h*2*np.sin(theta1_h)/l2_h)/2
                    # theta2_v = np.arcsin(org_hyp_v*2*np.sin(theta1_v)/l2_v)/2

                    theta3_h = np.arcsin(l1_h*np.sin(theta1_h)/l2_h)
                    # theta3_v = np.arcsin(l1_v*np.sin(theta1_v)/l2_v)

                    l4_h = ((org_ell_h)**2 - 2*org_ell_h*a_ell_h*np.cos(theta3_h) + a_ell_h**2)/(a_ell_h - org_ell_h*np.cos(theta3_h))
                    # l4_v = ((org_ell_v)**2 - 2*org_ell_v*a_ell_v*np.cos(theta3_v) + a_ell_v**2)/(a_ell_v - org_ell_v*np.cos(theta3_v))

                    l3_h = 2*a_ell_h - l2_h - l4_h
                    # l3_v = 2*a_ell_v - l2_v - l4_v

                    theta5_h = np.arcsin((2*a_ell_h - l4_h)*np.sin(theta3_h)/l4_h)
                    # theta5_v = np.arcsin((2*a_ell_v - l4_v)*np.sin(theta3_v)/l4_v)

                    theta4_h = (theta5_h+theta3_h)/2.
                    # theta4_v = (theta5_v+theta3_v)/2.
                    print('theta1',theta1_h)
                    print('l2',l2_h)
                    # print('l2_v',l2_v)
                    print('l1',l1_h)
                    # print('l1_v',l1_v)
                    print('theta2 incidence hyp',theta2_h)
                    # print('theta2_v incidence hyp',theta2_v)
                    print('theta3',theta3_h)
                    # print('theta3_v',theta3_v)
                    print('l3',l3_h)
                    # print('l3_v',l3_v)
                    print('l4',l4_h)
                    # print('l4_v',l4_v)
                    print('hyp to ell',np.cos(theta3_h)*(l3_h))
                    # print('hyp to ell v',np.cos(theta3_v)*(l3_v))
                    print('theta4 incidence ell',theta4_h)
                    # print('theta4_v incidence ell',theta4_v)
                    theta4_h = np.arcsin(2*org_ell_h*np.sin(theta3_h)/l4_h)/2
                    # theta4_v = np.arcsin(2*org_ell_v*np.sin(theta3_v)/l4_v)/2
                    print('theta4 incidence ell',theta4_h)
                    # print('theta4_v incidence ell',theta4_v)
                    print('theta5 focal',theta5_h)
                    # print('theta5_v focal',theta5_v)
                    print('width1',l1_h*np.cos(theta1_h))
                    print('width2',l3_h*np.cos(theta3_h))
                    print('width3',l4_h*np.cos(theta5_h))
                    print('')
                    return theta3_h, theta5_h

                print('===== Horizontal center =====')
                theta3_h, theta5_h = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h)
                print('===== Horizontal edge1 =====')
                theta3_h1, theta5_h1 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y1_h / x1_h))
                print('===== Horizontal edge2 =====')
                theta3_h2, theta5_h2 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y2_h / x2_h))

                print('===== Vertical center =====')
                theta3_v, theta5_v = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,theta1_v)
                print('===== Vertical edge1 =====')
                theta3_v1, theta5_v1 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y1_v / x1_v))
                print('===== Vertical edge2 =====')
                theta3_v2, theta5_v2 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y2_v / x2_v))

                print('===== ===== =====')
                omegav1 = (theta3_v1 + theta3_v2 - np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))/2
                print('omegav1',omegav1)
                omegah1 = (theta3_h1 + theta3_h2 - np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))/2
                print('omegah1',omegah1)
                omegav2 = (np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v) + theta5_v1 + theta5_v2)/2
                print('omegav2',omegav2)

                print('===== ===== =====')
                print('na_h',np.sin((theta5_h1 - theta5_h2))/2.)
                print('na_v',np.sin((theta5_v1 - theta5_v2))/2.)
                print('div_h',(theta5_h1 - theta5_h2))
                print('div_v',(theta5_v1 - theta5_v2))
                print('div0_h',np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))
                print('div0_v',np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))
                print('div_int_h',theta3_h1-theta3_h2)
                print('div_int_v',theta3_v1-theta3_v2)
                print('===== ===== =====')
                # print('theta_edge1_h1', theta_cntr_h)
                # print('theta_cntr_h1', theta_cntr_h)
                # print('theta_edge1_h1', np.arctan(y2_h / x2_h))
            else:
                def print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h):
                    l2_h = (4*a_hyp_h**2 + (org_hyp_h*2)**2 -4*a_hyp_h*(org_hyp_h*2)*np.cos(theta1_h))/(4*org_hyp_h - 4*a_hyp_h)
                    # l2_v = (4*a_hyp_v**2 + (org_hyp_v*2)**2 -4*a_hyp_v*(org_hyp_v*2)*np.cos(theta1_v))/(4*org_hyp_v - 4*a_hyp_v)

                    l1_h = 2*a_hyp_h +l2_h
                    # l1_v = 2*a_hyp_v +l2_v

                    theta2_h = np.arcsin(org_hyp_h*2*np.sin(theta1_h)/l2_h)/2
                    # theta2_v = np.arcsin(org_hyp_v*2*np.sin(theta1_v)/l2_v)/2

                    theta3_h = np.arcsin(l1_h*np.sin(theta1_h)/l2_h)
                    # theta3_v = np.arcsin(l1_v*np.sin(theta1_v)/l2_v)

                    l4_h = ((org_ell_h)**2 - 2*org_ell_h*a_ell_h*np.cos(theta3_h) + a_ell_h**2)/(a_ell_h - org_ell_h*np.cos(theta3_h))
                    # l4_v = ((org_ell_v)**2 - 2*org_ell_v*a_ell_v*np.cos(theta3_v) + a_ell_v**2)/(a_ell_v - org_ell_v*np.cos(theta3_v))

                    l3_h = 2*a_ell_h - l2_h - l4_h
                    # l3_v = 2*a_ell_v - l2_v - l4_v

                    theta5_h = np.arcsin((2*a_ell_h - l4_h)*np.sin(theta3_h)/l4_h)
                    # theta5_v = np.arcsin((2*a_ell_v - l4_v)*np.sin(theta3_v)/l4_v)

                    theta4_h = (theta5_h+theta3_h)/2.
                    # theta4_v = (theta5_v+theta3_v)/2.
                    # print('theta4_v incidence ell',theta4_v)
                    theta4_h = np.arcsin(2*org_ell_h*np.sin(theta3_h)/l4_h)/2
                    # theta4_v = np.arcsin(2*org_ell_v*np.sin(theta3_v)/l4_v)/2
                    return theta3_h, theta5_h

                theta3_h, theta5_h = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h)
                theta3_h1, theta5_h1 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y1_h / x1_h))
                theta3_h2, theta5_h2 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y2_h / x2_h))
                theta3_v, theta5_v = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,theta1_v)
                theta3_v1, theta5_v1 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y1_v / x1_v))
                theta3_v2, theta5_v2 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y2_v / x2_v))
                omegav1 = (theta3_v1 + theta3_v2 - np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))/2
                omegah1 = (theta3_h1 + theta3_h2 - np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))/2
                omegav2 = (np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v) + theta5_v1 + theta5_v2)/2
            theta_source_1_h = np.arctan(y1_h / x1_h) - theta_cntr_h
            theta_source_2_h = np.arctan(y2_h / x2_h) - theta_cntr_h
            theta_source_1_v = np.arctan(y1_v / x1_v) - theta_cntr_v
            theta_source_2_v = np.arctan(y2_v / x2_v) - theta_cntr_v

            bufray[0, 0] = 1.
            bufray[1, 0] = np.tan(theta1_h)
            bufray[2, 0] = np.tan(theta1_v)

            bufray[0, 1] = 1.
            bufray[1, 1] = np.tan(theta_source_1_h)
            bufray[2, 1] = np.tan(theta_source_1_v)

            bufray[0, 2] = 1.
            bufray[1, 2] = np.tan(theta_source_2_h)
            bufray[2, 2] = np.tan(theta_source_1_v)

            bufray[0, 3] = 1.
            bufray[1, 3] = np.tan(theta_source_2_h)
            bufray[2, 3] = np.tan(theta_source_1_v)

            bufray[0, 4] = 1.
            bufray[1, 4] = np.tan(theta_source_2_h)
            bufray[2, 4] = np.tan(theta_source_2_v)

            source = np.zeros((3, 5))

        else:
            bufray = np.zeros((3, 2))
            if option_axial:
                bufray[0, :] = 1.
                bufray[1, :] = np.tan(theta1_h)
                bufray[2, :] = np.tan(theta1_v)
            def print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h):
                l2_h = (4*a_hyp_h**2 + (org_hyp_h*2)**2 -4*a_hyp_h*(org_hyp_h*2)*np.cos(theta1_h))/(4*org_hyp_h - 4*a_hyp_h)
                # l2_v = (4*a_hyp_v**2 + (org_hyp_v*2)**2 -4*a_hyp_v*(org_hyp_v*2)*np.cos(theta1_v))/(4*org_hyp_v - 4*a_hyp_v)

                l1_h = 2*a_hyp_h +l2_h
                # l1_v = 2*a_hyp_v +l2_v

                theta2_h = np.arcsin(org_hyp_h*2*np.sin(theta1_h)/l2_h)/2
                # theta2_v = np.arcsin(org_hyp_v*2*np.sin(theta1_v)/l2_v)/2

                theta3_h = np.arcsin(l1_h*np.sin(theta1_h)/l2_h)
                # theta3_v = np.arcsin(l1_v*np.sin(theta1_v)/l2_v)

                l4_h = ((org_ell_h)**2 - 2*org_ell_h*a_ell_h*np.cos(theta3_h) + a_ell_h**2)/(a_ell_h - org_ell_h*np.cos(theta3_h))
                # l4_v = ((org_ell_v)**2 - 2*org_ell_v*a_ell_v*np.cos(theta3_v) + a_ell_v**2)/(a_ell_v - org_ell_v*np.cos(theta3_v))

                l3_h = 2*a_ell_h - l2_h - l4_h
                # l3_v = 2*a_ell_v - l2_v - l4_v

                theta5_h = np.arcsin((2*a_ell_h - l4_h)*np.sin(theta3_h)/l4_h)
                # theta5_v = np.arcsin((2*a_ell_v - l4_v)*np.sin(theta3_v)/l4_v)

                theta4_h = (theta5_h+theta3_h)/2.
                # theta4_v = (theta5_v+theta3_v)/2.
                print('theta1',theta1_h)
                print('l2',l2_h)
                # print('l2_v',l2_v)
                print('l1',l1_h)
                # print('l1_v',l1_v)
                print('theta2 incidence hyp',theta2_h)
                # print('theta2_v incidence hyp',theta2_v)
                print('theta3',theta3_h)
                # print('theta3_v',theta3_v)
                print('l3',l3_h)
                # print('l3_v',l3_v)
                print('l4',l4_h)
                # print('l4_v',l4_v)
                print('hyp to ell',np.cos(theta3_h)*(l3_h))
                # print('hyp to ell v',np.cos(theta3_v)*(l3_v))
                print('theta4 incidence ell',theta4_h)
                # print('theta4_v incidence ell',theta4_v)
                theta4_h = np.arcsin(2*org_ell_h*np.sin(theta3_h)/l4_h)/2
                # theta4_v = np.arcsin(2*org_ell_v*np.sin(theta3_v)/l4_v)/2
                print('theta4 incidence ell',theta4_h)
                # print('theta4_v incidence ell',theta4_v)
                print('theta5 focal',theta5_h)
                # print('theta5_v focal',theta5_v)
                print('width1',l1_h*np.cos(theta1_h))
                print('width2',l3_h*np.cos(theta3_h))
                print('width3',l4_h*np.cos(theta5_h))
                print('')
                return theta3_h, theta5_h


            theta3_h, theta5_h = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,theta1_h)
            theta3_h1, theta5_h1 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y1_h / x1_h))
            theta3_h2, theta5_h2 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,a_ell_h,b_ell_h,org_ell_h,np.arctan(y2_h / x2_h))
            theta3_v, theta5_v = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,theta1_v)
            theta3_v1, theta5_v1 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y1_v / x1_v))
            theta3_v2, theta5_v2 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,a_ell_v,b_ell_v,org_ell_v,np.arctan(y2_v / x2_v))
            omegav1 = (theta3_v1 + theta3_v2 - np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))/2
            omegah1 = (theta3_h1 + theta3_h2 - np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))/2
            omegav2 = (np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v) + theta5_v1 + theta5_v2)/2
            source = np.zeros((3, 2))


        bufray = normalize_vector(bufray)

        center_hyp_v = mirr_ray_intersection(coeffs_hyp_v, bufray, source)
        if not np.isreal(center_hyp_v).all():
            return np.inf
        bufreflect1 = reflect_ray(bufray, norm_vector(coeffs_hyp_v, center_hyp_v))

        bufreflangle1_y = np.arctan(np.mean(bufreflect1[2, 1:]) / np.mean(bufreflect1[0, 1:]))

        if option == 'ray':
            print('coeffs_hyp_v',coeffs_hyp_v)
            print('center_hyp_v',center_hyp_v)
            print('bufray',bufray)
            print('np.mean(bufreflect1[2, 1:])',np.mean(bufreflect1[2, 1:]))
            print('np.mean(bufreflect1[0, 1:])',np.mean(bufreflect1[0, 1:]))
            print('angle_y 1st to 2nd',bufreflangle1_y)

        # print(center_hyp_v)
        # print(bufreflect1)
        # print(np.arctan(bufreflect1[2, 0] / bufreflect1[0, 0]))
        # print('1st')

        # H hyp mirror set (2nd)
        axis2_x = np.array([1., 0., 0.])
        axis2_y = np.array([0., 1., 0.])
        axis2_z = np.array([0., 0., 1.])
        coeffs_hyp_h = np.zeros(10)
        coeffs_hyp_h[0] = 1 / a_hyp_h**2
        coeffs_hyp_h[1] = -1 / b_hyp_h**2
        coeffs_hyp_h[9] = -1.
        # coeffs_hyp_h = shift_x(coeffs_hyp_h, org_hyp_h + astigH)
        org_hyp_h1 = org_hyp_h + astigH
        coeffs_hyp_h = shift_x(coeffs_hyp_h, org_hyp_h1)
        if option_axial:
            # coeffs_hyp_h = rotate_y(coeffs_hyp_h, bufreflangle1_y, center_hyp_v[:, 0])
            # coeffs_hyp_h = rotate_z(coeffs_hyp_h, -theta1_h, [0, 0, 0])
            coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis2_z, -theta1_h, [0, 0, 0])
            axis2_x, axis2_y, axis2_z = rotatematrix(rotation_matrix, axis2_x, axis2_y, axis2_z)
        mean_bufreflect1 = np.mean(bufreflect1[:, 1:],axis = 1)
        if option == 'ray':
            roty_local2 = np.arctan(np.dot(mean_bufreflect1,axis2_z) / np.dot(mean_bufreflect1,axis2_x))
            print('rot localy2',roty_local2)

        center_hyp_h = mirr_ray_intersection(coeffs_hyp_h, bufreflect1, center_hyp_v)
        if not np.isreal(center_hyp_h).all():
            return np.inf

        if option_alignment:
            if not optin_axialrotation:
                coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle1_y, np.mean(center_hyp_v[:, 1:],axis=1))
            if optin_axialrotation:
                if option_rotateLocal:
                    coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis2_y, -omegav1, np.mean(center_hyp_h[:, 1:],axis=1))
                    axis2_x, axis2_y, axis2_z = rotatematrix(rotation_matrix, axis2_x, axis2_y, axis2_z)
                else:
                    coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle1_y, np.mean(center_hyp_h[:, 1:],axis=1))
                center_hyp_h = mirr_ray_intersection(coeffs_hyp_h, bufreflect1, center_hyp_v)
        bufreflect2 = reflect_ray(bufreflect1, norm_vector(coeffs_hyp_h, center_hyp_h))
        bufreflangle2_z = np.arctan(np.mean(bufreflect2[1, 1:]) / np.mean(bufreflect2[0, 1:]))
        bufreflangle2_y = np.arctan(np.mean(bufreflect2[2, 1:]) / np.mean(bufreflect2[0, 1:]))

        if option == 'ray':
            print('angle_y 2nd to 3rd',bufreflangle2_y)
            print('angle_z 2nd to 3rd',bufreflangle2_z)
        # print(bufreflect2)
        # print(bufreflangle2_z)

        # print('2nd')
        # print(center_hyp_h)


        # Set ellipse mirror in the vert set (3rd)
        axis3_x = np.array([1., 0., 0.])
        axis3_y = np.array([0., 1., 0.])
        axis3_z = np.array([0., 0., 1.])
        coeffs_ell_v = np.zeros(10)
        coeffs_ell_v[0] = 1 / a_ell_v**2
        coeffs_ell_v[2] = 1 / b_ell_v**2
        coeffs_ell_v[9] = -1.

        coeffs_ell_v = shift_x(coeffs_ell_v, 2 * org_hyp_v + org_ell_v)

        if option_axial:
            coeffs_ell_v, rotation_matrix = rotate_general_axis(coeffs_ell_v, axis3_y, theta1_v, [0, 0, 0])
            axis3_x, axis3_y, axis3_z = rotatematrix(rotation_matrix, axis3_x, axis3_y, axis3_z)

        center_ell_v = mirr_ray_intersection(coeffs_ell_v, bufreflect2, center_hyp_h)
        if not np.isreal(center_ell_v).all():
            return np.inf
        if option_alignment:
            if not optin_axialrotation:
                coeffs_ell_v = rotate_z(coeffs_ell_v, bufreflangle2_z, np.mean(center_hyp_h[:, 1:],axis=1))
            if optin_axialrotation:
                if option_rotateLocal:
                    coeffs_ell_v, rotation_matrix = rotate_general_axis(coeffs_ell_v, axis3_z, omegah1, np.mean(center_ell_v[:, 1:],axis=1))
                    axis3_x, axis3_y, axis3_z = rotatematrix(rotation_matrix, axis3_x, axis3_y, axis3_z)
                else:
                    coeffs_ell_v = rotate_z(coeffs_ell_v, bufreflangle2_z, center_ell_v[:, 0])
                center_ell_v = mirr_ray_intersection(coeffs_ell_v, bufreflect2, center_hyp_h)
        bufreflect3 = reflect_ray(bufreflect2, norm_vector(coeffs_ell_v, center_ell_v))
        bufreflangle3_y = np.arctan(np.mean(bufreflect3[2, 1:]) / np.mean(bufreflect3[0, 1:]))
        bufreflangle3_z = np.arctan(np.mean(bufreflect3[1, 1:]) / np.mean(bufreflect3[0, 1:]))

        if option == 'ray':
            print('angle_y 3rd to 4th',bufreflangle3_y)
            print('angle_z 3rd to 4th',bufreflangle3_z)
        # print(bufreflangle3_y)
        # print('3rd')

        # Set ellipse mirror in the horiz set (4th)
        axis4_x = np.array([1., 0., 0.])
        axis4_y = np.array([0., 1., 0.])
        axis4_z = np.array([0., 0., 1.])
        coeffs_ell_h = np.zeros(10)
        coeffs_ell_h[0] = 1 / a_ell_h**2
        coeffs_ell_h[1] = 1 / b_ell_h**2
        coeffs_ell_h[9] = -1.

        coeffs_ell_h = shift_x(coeffs_ell_h, 2 * org_hyp_h + org_ell_h + astigH)
        if option_axial:
            # coeffs_ell_h = rotate_y(coeffs_ell_h, bufreflangle1_y, center_hyp_v[:, 0])
            coeffs_ell_h, rotation_matrix = rotate_general_axis(coeffs_ell_h, axis4_z, -theta1_h, [0, 0, 0])
            axis4_x, axis4_y, axis4_z = rotatematrix(rotation_matrix, axis4_x, axis4_y, axis4_z)

        center_ell_h = mirr_ray_intersection(coeffs_ell_h, bufreflect3, center_ell_v)
        if not np.isreal(center_ell_h).all():
            return np.inf
        if option_alignment:
            if not optin_axialrotation:
                coeffs_ell_h = rotate_y(coeffs_ell_h, -bufreflangle3_y, np.mean(center_ell_v[:, 1:],axis=1))
            if optin_axialrotation:
                if option_rotateLocal:
                    coeffs_ell_h, rotation_matrix = rotate_general_axis(coeffs_ell_h, axis4_y, omegav2, np.mean(center_ell_h[:, 1:],axis=1))
                    axis4_x, axis4_y, axis4_z = rotatematrix(rotation_matrix, axis4_x, axis4_y, axis4_z)
                else:
                    coeffs_ell_h = rotate_y(coeffs_ell_h, -bufreflangle3_y, np.mean(center_ell_h[:, 1:],axis=1))
                center_ell_h = mirr_ray_intersection(coeffs_ell_h, bufreflect3, center_ell_v)
        bufreflect4 = reflect_ray(bufreflect3, norm_vector(coeffs_ell_h, center_ell_h))
        bufreflangle4_z = np.arctan(np.mean(bufreflect4[1, 1:]) / np.mean(bufreflect4[0, 1:]))
        bufreflangle4_y = np.arctan(np.mean(bufreflect4[2, 1:]) / np.mean(bufreflect4[0, 1:]))

        s2f_H = 2 * org_hyp_h + 2 * org_ell_h
        s2f_V = 2 * org_hyp_v + 2 * org_ell_v
        # print(s2f_H)
        # print(s2f_V)
        s2f_middle = (s2f_H + s2f_V) / 2
        coeffs_det = np.zeros(10)
        coeffs_det[6] = 1.
        coeffs_det[9] = -(s2f_middle + defocus)

        # if option_axial:
        #     coeffs_det = rotate_y(coeffs_det, theta1_v, [0, 0, 0])
        #     coeffs_det = rotate_y(coeffs_det, bufreflangle1_y, center_hyp_v[:, 0])
        #     coeffs_det = rotate_z(coeffs_det, -theta1_h, [0, 0, 0])
        detcenter = plane_ray_intersection(coeffs_det, bufreflect4, center_ell_h)
        detcenter = detcenter[:, 0]

        # if pitch_ell_v != 0:
        #     coeffs_ell_v = rotate_y(coeffs_ell_v, pitch_ell_v, center_ell_v[:, 0])
        # if roll_ell_v != 0:
        #     coeffs_ell_v = rotate_x(coeffs_ell_v, roll_ell_v, center_ell_v[:, 0])
        # if yaw_ell_v != 0:
        #     coeffs_ell_v = rotate_z(coeffs_ell_v, yaw_ell_v, center_ell_v[:, 0])
        #
        # if pitch_ell_h != 0:
        #     coeffs_ell_h = rotate_y(coeffs_ell_h, pitch_ell_h, center_ell_h[:, 0])
        # if roll_ell_h != 0:
        #     coeffs_ell_h = rotate_x(coeffs_ell_h, roll_ell_h, center_ell_h[:, 0])
        # if yaw_ell_h != 0:
        #     coeffs_ell_h = rotate_z(coeffs_ell_h, yaw_ell_h, center_ell_h[:, 0])
        #
        # if pitch_hyp_v != 0:
        #     coeffs_hyp_v = rotate_y(coeffs_hyp_v, pitch_hyp_v, center_hyp_v[:, 0])
        # if roll_hyp_v != 0:
        #     coeffs_hyp_v = rotate_x(coeffs_hyp_v, roll_hyp_v, center_hyp_v[:, 0])
        # if yaw_hyp_v != 0:
        #     coeffs_hyp_v = rotate_z(coeffs_hyp_v, yaw_hyp_v, center_hyp_v[:, 0])
        #
        # if pitch_hyp_h != 0:
        #     coeffs_hyp_h = rotate_y(coeffs_hyp_h, pitch_hyp_h, center_hyp_h[:, 0])
        # if roll_hyp_h != 0:
        #     coeffs_hyp_h = rotate_x(coeffs_hyp_h, roll_hyp_h, center_hyp_h[:, 0])
        # if yaw_hyp_h != 0:
        #     coeffs_hyp_h = rotate_z(coeffs_hyp_h, yaw_hyp_h, center_hyp_h[:, 0])
        if option_rotateLocal:
            if yaw_ell_v != 0:
                coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis3_z, yaw_ell_v, np.mean(center_ell_v[:, 1:],axis=1))
            if pitch_ell_v != 0:
                coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis3_y, pitch_ell_v, np.mean(center_ell_v[:, 1:],axis=1))
            if roll_ell_v != 0:
                coeffs_ell_v, _ = rotate_general_axis(coeffs_ell_v, axis3_x, roll_ell_v, np.mean(center_ell_v[:, 1:],axis=1))


            if pitch_ell_h != 0:
                coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis4_y, pitch_ell_h, np.mean(center_ell_h[:, 1:],axis=1))
            if yaw_ell_h != 0:
                coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis4_z, yaw_ell_h, np.mean(center_ell_h[:, 1:],axis=1))
            if roll_ell_h != 0:
                coeffs_ell_h, _ = rotate_general_axis(coeffs_ell_h, axis4_x, roll_ell_h, np.mean(center_ell_h[:, 1:],axis=1))

            if yaw_hyp_v != 0:
                coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_z, yaw_hyp_v, np.mean(center_hyp_v[:, 1:],axis=1))
            if pitch_hyp_v != 0:
                coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_y, pitch_hyp_v, np.mean(center_hyp_v[:, 1:],axis=1))
            if roll_hyp_v != 0:
                coeffs_hyp_v, _ = rotate_general_axis(coeffs_hyp_v, axis_x, roll_hyp_v, np.mean(center_hyp_v[:, 1:],axis=1))

            if pitch_hyp_h != 0:
                coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis2_y, pitch_hyp_h, np.mean(center_hyp_h[:, 1:],axis=1))
            if yaw_hyp_h != 0:
                coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis2_z, yaw_hyp_h, np.mean(center_hyp_h[:, 1:],axis=1))
            if roll_hyp_h != 0:
                coeffs_hyp_h, _ = rotate_general_axis(coeffs_hyp_h, axis2_x, roll_hyp_h, np.mean(center_hyp_h[:, 1:],axis=1))

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
            source = np.zeros((3, ray_num * ray_num))
            if option_axial:
                rand_p0h = np.linspace(np.arctan(y1_h / x1_h), np.arctan(y2_h / x2_h), ray_num)
                rand_p0v = np.linspace(np.arctan(y1_v / x1_v), np.arctan(y2_v / x2_v), ray_num)
                rand_p0h = rand_p0h - np.mean(rand_p0h)
                rand_p0v = rand_p0v - np.mean(rand_p0v)
            if not option_axial:
                rand_p0h = np.linspace(np.arctan(y1_h / x1_h), np.arctan(y2_h / x2_h), ray_num)
                rand_p0v = np.linspace(np.arctan(y1_v / x1_v), np.arctan(y2_v / x2_v), ray_num)
            # rand_p0h = create_non_uniform_distribution(np.arctan(y1_h / x1_h), np.arctan(y2_h / x2_h), ray_num)
            # rand_p0v = create_non_uniform_distribution(np.arctan(y1_v / x1_v), np.arctan(y2_v / x2_v), ray_num)
            # rand_p0h = rand_p0h*0.1
            # rand_p0h = rand_p0v*0.1

            phai0 = np.zeros((3, ray_num * ray_num))
            for i in range(ray_num):
                rand_p0v_here = rand_p0v[i]
                phai0[1, ray_num * i:ray_num * (i + 1)] = np.tan(rand_p0h)
                phai0[2, ray_num * i:ray_num * (i + 1)] = np.tan(rand_p0v_here)
                phai0[0, ray_num * i:ray_num * (i + 1)] = 1.

            phai0 = normalize_vector(phai0)

            # plt.figure()
            # plt.scatter(phai0[1, :], phai0[2, :])
            # plt.title('angle')
            # plt.xlabel('Horizontal Angle (rad)')
            # plt.ylabel('Vertical Angle (rad)')
            # plt.show()

            vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
            reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))


            hmirr_hyp = mirr_ray_intersection(coeffs_hyp_h, reflect1, vmirr_hyp)
            reflect2 = reflect_ray(reflect1, norm_vector(coeffs_hyp_h, hmirr_hyp))


            vmirr_ell = mirr_ray_intersection(coeffs_ell_v, reflect2, hmirr_hyp)
            reflect3 = reflect_ray(reflect2, norm_vector(coeffs_ell_v, vmirr_ell))

            hmirr_ell = mirr_ray_intersection(coeffs_ell_h, reflect3, vmirr_ell)
            reflect4 = reflect_ray(reflect3, norm_vector(coeffs_ell_h, hmirr_ell))

            mean_reflect4 = np.mean(reflect4,1)
            # print(mean_reflect2)
            # print(np.sum(mean_reflect2*mean_reflect2))
            mean_reflect4 = normalize_vector(mean_reflect4)
            # print(mean_reflect2)
            # print(np.sum(mean_reflect2*mean_reflect2))

            if option == 'sep_direct':
                defocus = find_defocus(reflect4, hmirr_ell, s2f_middle,defocus,ray_num)

            coeffs_det = np.zeros(10)
            coeffs_det[6] = 1
            coeffs_det[9] = -(s2f_middle + defocus)
            detcenter = plane_ray_intersection(coeffs_det, reflect4, hmirr_ell)

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
                angle_h = np.arctan(angle[1, :]/angle[0, :])
                angle_v = np.arctan(angle[2, :]/angle[0, :])

                angle_v_sep_y = angle_v[thinned_array_v_y]
                angle_h_sep_y = angle_h[thinned_array_h_y]

                output_equal_v = np.linspace(angle_v_sep_y[0],angle_v_sep_y[-1],len(angle_v_sep_y))
                output_equal_h = np.linspace(angle_h_sep_y[0],angle_h_sep_y[-1],len(angle_h_sep_y))

                interp_func_v = interp1d(angle_v_sep_y, rand_p0v, kind='linear')
                interp_func_h = interp1d(angle_h_sep_y, rand_p0h, kind='linear')

                rand_p0v_new = interp_func_v(output_equal_v)
                rand_p0h_new = interp_func_h(output_equal_h)

                phai0 = np.zeros((3, ray_num * ray_num))
                for i in range(ray_num):
                    rand_p0v_here = rand_p0v_new[i]
                    phai0[1, ray_num * i:ray_num * (i + 1)] = np.tan(rand_p0h_new)
                    phai0[2, ray_num * i:ray_num * (i + 1)] = np.tan(rand_p0v_here)
                    phai0[0, ray_num * i:ray_num * (i + 1)] = 1.

                phai0 = normalize_vector(phai0)

                vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
                reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))

                dist0to1 = np.linalg.norm(vmirr_hyp - source,axis=0)

                if option_2mirror ==True:
                    hmirr_hyp = mirr_ray_intersection(coeffs_hyp_h, reflect1, vmirr_hyp)
                    reflect2 = reflect_ray(reflect1, norm_vector(coeffs_hyp_h, hmirr_hyp))
                else:
                    hmirr_hyp = vmirr_hyp
                    reflect2 = reflect1

                dist1to2 = np.linalg.norm(hmirr_hyp - vmirr_hyp,axis=0)
                vmirr_ell = mirr_ray_intersection(coeffs_ell_v, reflect2, hmirr_hyp)
                reflect3 = reflect_ray(reflect2, norm_vector(coeffs_ell_v, vmirr_ell))
                dist2to3 = np.linalg.norm(vmirr_ell - hmirr_hyp,axis=0)

                hmirr_ell = mirr_ray_intersection(coeffs_ell_h, reflect3, vmirr_ell)
                reflect4 = reflect_ray(reflect3, norm_vector(coeffs_ell_h, hmirr_ell))
                dist3to4 = np.linalg.norm(hmirr_ell - vmirr_ell,axis=0)

                if option == 'sep_direct':
                    defocus = find_defocus(reflect4, hmirr_ell, s2f_middle,defocus,ray_num)

                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)
                detcenter = plane_ray_intersection(coeffs_det, reflect4, hmirr_ell)
                if option == 'ray':
                    from scipy.spatial import cKDTree
                    def mindist(A,B):
                        tree = cKDTree(B.T)
                        dist, idx = tree.query(A.T, k=1)  # Aの各点からBへの最近点距離
                        min_dist = np.min(dist)
                        return min_dist
                    print('======================')
                    print('workX srs 1st',np.min(vmirr_hyp[0,:]) - np.max(source[0,:]))
                    print('workX 1st 2nd',np.min(hmirr_hyp[0,:]) - np.max(vmirr_hyp[0,:]))
                    print('workX 2nd 3rd',np.min(vmirr_ell[0,:]) - np.max(hmirr_hyp[0,:]))
                    print('workX 3rd 4th',np.min(hmirr_ell[0,:]) - np.max(vmirr_ell[0,:]))
                    print('workX 4th fcs',np.min(detcenter[0,:]) - np.max(hmirr_ell[0,:]))
                    print('======================')
                    print('workAbs srs 1st',mindist(source,vmirr_hyp))
                    print('workAbs 1st 2nd',mindist(vmirr_hyp,hmirr_hyp))
                    print('workAbs 2nd 3rd',mindist(hmirr_hyp,vmirr_ell))
                    print('workAbs 3rd 4th',mindist(vmirr_ell,hmirr_ell))
                    print('workAbs 4th fcs',mindist(hmirr_ell,detcenter))

                    print('1st W upper',np.linalg.norm(vmirr_hyp[:,0] - vmirr_hyp[:,ray_num-1]))
                    print('1st W lower',np.linalg.norm(vmirr_hyp[:,-1] - vmirr_hyp[:,-ray_num]))
                    print('2nd W lower',np.linalg.norm(hmirr_hyp[:,0] - hmirr_hyp[:,-ray_num]))
                    print('2nd W upper',np.linalg.norm(hmirr_hyp[:,ray_num-1] - hmirr_hyp[:,-1]))
                    print('3rd W upper',np.linalg.norm(vmirr_ell[:,0] - vmirr_ell[:,ray_num-1]))
                    print('3rd W lower',np.linalg.norm(vmirr_ell[:,-1] - vmirr_ell[:,-ray_num]))
                    print('4th W upper',np.linalg.norm(hmirr_ell[:,0] - hmirr_ell[:,-ray_num]))
                    print('4th W lower',np.linalg.norm(hmirr_ell[:,ray_num-1] - hmirr_ell[:,-1]))

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
                    plt.show()

                    

                angle = reflect4

            if option == 'wave':
                print('diverg angle H',np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))
                print('diverg angle V',np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))
                # 全データからランダムに10%だけを選択
                sample_indices = np.random.choice(detcenter.shape[1], size=int(detcenter.shape[1]*0.001), replace=False)

                theta_y = -np.mean(np.arctan(angle[2, :]/angle[0, :]))
                theta_z = np.mean(np.arctan(angle[1, :]/angle[0, :]))
                source = np.zeros((3,1))
                if option_rotate==True:
                    reflect4_rotated = rotate_vectors(reflect4, -theta_y, -theta_z)
                    focus_apprx = np.mean(detcenter,axis=1)
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
                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)
                detcenter = plane_ray_intersection(coeffs_det, reflect4_rotated, hmirr_ell_points_rotated)

                vec0to1 = normalize_vector(vmirr_hyp_points_rotated_grid - source_rotated)
                vec1to2 = normalize_vector(hmirr_hyp_points_rotated_grid - vmirr_hyp_points_rotated_grid)
                vec2to3 = normalize_vector(vmirr_ell_points_rotated_grid - hmirr_hyp_points_rotated_grid)
                vec3to4 = normalize_vector(hmirr_ell_points_rotated_grid - vmirr_ell_points_rotated_grid)
                vec4to5 = normalize_vector(detcenter - hmirr_ell_points_rotated_grid)

                vmirr_norm = normalize_vector( (-vec1to2 + vec0to1) / 2 )
                hmirr_norm = normalize_vector( (-vec2to3 + vec1to2) / 2 )
                vmirr2_norm = normalize_vector( (-vec3to4 + vec2to3) / 2 )
                hmirr2_norm = normalize_vector( (-vec4to5 + vec3to4) / 2 )

                if np.abs(defocusForWave) > 1e-9:
                    coeffs_det2 = np.zeros(10)
                    coeffs_det2[6] = 1
                    coeffs_det2[9] = -(s2f_middle + defocus+defocusForWave)
                    detcenter2 = plane_ray_intersection(coeffs_det2, reflect4_rotated, hmirr_ell_points_rotated)
                    return source_rotated, vmirr_hyp_points_rotated_grid, hmirr_hyp_points_rotated_grid, vmirr_ell_points_rotated_grid, hmirr_ell_points_rotated_grid, detcenter, detcenter2, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vmirr2_norm, hmirr2_norm, vec0to1, vec1to2, vec2to3, vec3to4
                else:
                    return source_rotated, vmirr_hyp_points_rotated_grid, hmirr_hyp_points_rotated_grid, vmirr_ell_points_rotated_grid, hmirr_ell_points_rotated_grid, detcenter, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vmirr2_norm, hmirr2_norm, vec0to1, vec1to2, vec2to3, vec3to4

            option_tilt = True
            if option_tilt:
                theta_y = -np.mean(np.arctan(angle[2, :]/angle[0, :]))
                theta_z = np.mean(np.arctan(angle[1, :]/angle[0, :]))
                reflect4_rotated = rotate_vectors(reflect4, -theta_y, -theta_z)
                focus_apprx = np.mean(detcenter,axis=1)
                hmirr_ell_points_rotated = rotate_points(hmirr_ell, focus_apprx, -theta_y, -theta_z)
                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)
                detcenter = plane_ray_intersection(coeffs_det, reflect4_rotated, hmirr_ell_points_rotated)

                hmirr_ell = hmirr_ell_points_rotated
                reflect4 = reflect4_rotated
                angle = reflect4

            if option == 'sep':
                focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2 = compare_sep(reflect4_rotated, hmirr_ell_points_rotated, coeffs_det,ray_num_H,1e-4)
                return focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2
            if option == 'sep_direct':
                focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2 = compare_sep(reflect4_rotated, hmirr_ell_points_rotated, coeffs_det,ray_num_H,1e-4)
                return focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2

            if option == 'ray_wave':
                if option_HighNA == True:
                    defocusWave = 1e-2
                    lambda_ = 13.5
                else:
                    defocusWave = 1e-3
                    lambda_ = 1.35
                coeffs_det2 = np.zeros(10)
                coeffs_det2[6] = 1
                coeffs_det2[9] = -(s2f_middle + defocus + defocusWave)
                detcenter2 = plane_ray_intersection(coeffs_det2, reflect4, hmirr_ell)

                dist4tofocus = np.linalg.norm(detcenter - hmirr_ell, axis=0)
                vector4tofocus = (detcenter - hmirr_ell) / dist4tofocus
                totalDist = dist0to1 + dist1to2 + dist2to3 + dist3to4 + dist4tofocus
                DistError = (totalDist - np.mean(totalDist))*1e9



                dist4tofocus2 = np.linalg.norm(detcenter2 - hmirr_ell, axis=0)
                vector4tofocus2 = (detcenter2 - hmirr_ell) / dist4tofocus2
                totalDist2 = dist0to1 + dist1to2 + dist2to3 + dist3to4 + dist4tofocus2
                DistError2 = (totalDist2 - np.mean(totalDist2))*1e9
                print('detcenter',np.mean(detcenter,axis=1))
                print('detcenter2',np.mean(detcenter2,axis=1))
                print('dist0to1',np.mean(dist0to1))
                print('dist1to2',np.mean(dist1to2))
                print('dist2to3',np.mean(dist2to3))
                print('dist3to4',np.mean(dist3to4))
                print('totalDist',np.mean(totalDist))
                print('dist4tofocus std',np.mean(dist4tofocus))
                print('dist0to1 std',np.std(dist0to1))
                print('dist1to2 std',np.std(dist1to2))
                print('dist2to3 std',np.std(dist2to3))
                print('dist3to4 std',np.std(dist3to4))
                print('dist4tofocus std',np.std(dist4tofocus))
                print('totalDist std',np.std(totalDist))
                print('np.std(totalDist)',np.std(totalDist))
                print('np.mean(totalDist)',np.mean(totalDist))
                print('np.mean(totalDist2)',np.mean(totalDist2))
                print('np.mean(totalDist2-totalDist)',np.mean(totalDist2-totalDist))
                print('np.std(totalDist2-totalDist)',np.std(totalDist2-totalDist))
                from numpy.fft import fft2, fftshift, fftfreq
                def fresnel_psf(phi, lambda_, z, grid_x, grid_y):
                    """
                    フレネル近似により、光路長分布から焦点面のPSFを計算する。

                    Parameters:
                    - phi : 2D array (光路長分布, NaNで開口外をマスク)
                    - lambda_ : float (波長 [m])
                    - z : float (焦点面までの距離 [m])
                    - grid_x, grid_y : 2D array (各ピクセルの物理座標 [m])

                    Returns:
                    - psf : 2D array (正規化されたPSF)
                    - x_out, y_out : 1D array (出力PSFのx, y座標軸 [m])
                    """
                    # 振幅マスクとNaN補完
                    A = ~np.isnan(phi)
                    phi = np.nan_to_num(phi, nan=0.0)
                    
                    # 入力複素波面
                    U = A * np.exp(1j * 2 * np.pi / lambda_ * phi)
                    
                    # ピクセルサイズ
                    dx = np.mean(np.diff(grid_x[0, :]))
                    dy = np.mean(np.diff(grid_y[:, 0]))

                    # 前項の二次位相因子
                    Q1 = np.exp(1j * np.pi / (lambda_ * z) * (grid_x**2 + grid_y**2))
                    U1 = U * Q1

                    # FFT: Fresnel伝搬の中心部分
                    U2 = fftshift(fft2(U1))

                    # 出力座標系（周波数 -> 空間座標）
                    ny, nx = phi.shape
                    fx = fftshift(fftfreq(nx, d=dx))
                    fy = fftshift(fftfreq(ny, d=dy))
                    x_out = fx * lambda_ * z
                    y_out = fy * lambda_ * z

                    # 出力側の二次位相因子（optional, 位相計算には必要だがPSFには不要）
                    # Q2 = np.exp(1j * np.pi / (lambda_ * z) * (grid_x**2 + grid_y**2))

                    # PSFの計算
                    psf = np.abs(U2)**2
                    psf /= np.nanmax(psf)

                    return psf, x_out, y_out

                # 補間するグリッドを作成
                grid_H, grid_V = np.meshgrid(
                    np.linspace(detcenter2[1, :].min(), detcenter2[1, :].max(), ray_num_H),
                    np.linspace(detcenter2[2, :].min(), detcenter2[2, :].max(), ray_num_V)
                )

                CosAngle = angle[0,:]
                # グリッド上にデータを補間 (method: 'linear', 'nearest', 'cubic' から選択)
                if False:
                    matrixDistError2 = griddata((detcenter2[1, :], detcenter2[2, :]), DistError2, (grid_H, grid_V), method='cubic')
                    meanFocus = np.mean(detcenter,axis=1)
                    Sph = np.linalg.norm(detcenter2 - meanFocus[:, np.newaxis], axis=0) * 1e9
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
                    print('np.mean(DistError2)',np.mean(DistError2))
                    print('np.std(DistError2)',np.std(DistError2))
                    print('np.mean(Sph)',np.mean(Sph))
                    print('np.std(Sph)',np.std(Sph))
                    print('np.mean(Wave2)',np.mean(Wave2))
                    print('np.std(Wave2)',np.std(Wave2))
                    print('grid_H.shape',grid_H.shape)
                    print('Wave2.shape',Wave2.shape)
                    print('detcenter2.shape',detcenter2.shape)

                    matrixWave2 = griddata((detcenter2[1, :], detcenter2[2, :]), Wave2, (grid_H, grid_V), method='cubic')
                    matrixWave2 = matrixWave2 - np.nanmean(matrixWave2)

                np.savetxt('matrixWave2(nm).txt',matrixWave2)
                tifffile.imwrite('matrixWave2(nm).tiff', matrixWave2)


                plt.figure()
                plt.pcolormesh(grid_H, grid_V, matrixWave2, cmap='jet', shading='auto',vmin = -1e-2,vmax = 1e-2)
                # plt.colorbar(label='\u03BB')
                plt.colorbar(label='wavefront error (nm)')
                plt.savefig('waveRaytrace.png')
                plt.show()

                matrixWave2_Corrected = plane_correction_with_nan_and_outlier_filter(matrixWave2)
                print('PV',np.nanmax(matrixWave2_Corrected)-np.nanmin(matrixWave2_Corrected))
                
                plt.figure()
                plt.pcolormesh(grid_H, grid_V, matrixWave2_Corrected, cmap='jet', shading='auto',vmin = -1e-2,vmax = 1e-2)
                # plt.colorbar(label='\u03BB')
                plt.colorbar(label='wavefront error (nm)')
                plt.title(f'PV={np.nanmax(matrixWave2_Corrected)-np.nanmin(matrixWave2_Corrected)}')
                plt.savefig('waveRaytrace_Corrected.png')
                plt.show()

                psf, x_out, y_out = fresnel_psf(matrixWave2_Corrected, lambda_=lambda_, z=defocusWave, grid_x=grid_H, grid_y=grid_V)
                plt.imshow(psf, extent=[x_out[0], x_out[-1], y_out[0], y_out[-1]], origin='lower', cmap='hot')
                plt.xlabel("x [m]")
                plt.ylabel("y [m]")
                plt.title("PSF (Fresnel Approx.)")
                plt.colorbar()
                plt.show()



                plt.figure()
                sample_detcenter = detcenter2.copy()
                sample_DistError = DistError2.copy()
                sample = np.vstack([sample_detcenter, sample_DistError])
                sizeh_here = ray_num_H
                sizev_here = ray_num_V
                while sizeh_here > 33:
                    sample, sizev_here, sizeh_here = downsample_array_any_n(sample, sizev_here, sizeh_here, 2, 2)
                scatter = plt.scatter(sample[1, :], sample[2, :],c=sample[3,:], cmap='jet')
                # カラーバーを追加
                plt.colorbar(scatter, label='OPL error (nm)')
                plt.axis('equal')
                plt.show()

                return

            if option == 'ray':
                print(theta1_v)
                print(np.cos(theta1_v)*s2f_middle)
                print(theta1_h)
                print(np.mean(detcenter[0,:]))
                print(np.mean(detcenter[1,:]))
                print(np.mean(detcenter[2,:]))
                # print(np.mean(detcenter[0,:]))
                print(coeffs_det)
                print('s2f_H',s2f_H)
                print('s2f_V',s2f_V)
                mabiki = round(np.sqrt(ray_num_H*ray_num_V)/50)
                defocussize = 4e-6
                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus) + defocussize
                detcenter1 = plane_ray_intersection(coeffs_det, reflect4, hmirr_ell)

                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus) - defocussize
                detcenter2 = plane_ray_intersection(coeffs_det, reflect4, hmirr_ell)
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[::ray_num+1]
                thinned_array = first_thinned_array[::mabiki]

                print('Oblique1',thinned_array)

                obl_1 = (detcenter1[1,thinned_array] + detcenter1[2,thinned_array])/np.sqrt(2)
                obl_2 = (detcenter2[1,thinned_array] + detcenter2[2,thinned_array])/np.sqrt(2)
                axs[0,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[obl_1,obl_2],c='y')
                axs[0,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(obl_2),np.max(obl_1)],color='k')
                axs[0,0].set_title('Oblique1 aperture 0')
                axs[0,0].set_xlabel('Axial (m)')
                axs[0,0].set_ylabel('Oblique1 Position (m)')

                obl_1 = (-detcenter1[1,thinned_array] + detcenter1[2,thinned_array])/np.sqrt(2)
                obl_2 = (-detcenter2[1,thinned_array] + detcenter2[2,thinned_array])/np.sqrt(2)
                axs[0,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[obl_1,obl_2],c='y')
                axs[0,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(obl_2),np.max(obl_1)],color='k')
                axs[0,1].set_title('Oblique1 aperture 0')
                axs[0,1].set_xlabel('Axial (m)')
                axs[0,1].set_ylabel('Oblique2 Position (m)')

                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[ray_num-1::ray_num-1][:-1]
                # first_thinned_array = first_thinned_array[:-1]
                thinned_array = first_thinned_array[::mabiki]

                print('Oblique2',thinned_array)

                obl_1 = (detcenter1[1,thinned_array] + detcenter1[2,thinned_array])/np.sqrt(2)
                obl_2 = (detcenter2[1,thinned_array] + detcenter2[2,thinned_array])/np.sqrt(2)
                axs[1,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[obl_1,obl_2],c='y')
                axs[1,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(obl_2),np.max(obl_1)],color='k')
                axs[1,0].set_title('Oblique2 aperture 0')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Oblique1 Position (m)')

                obl_1 = (-detcenter1[1,thinned_array] + detcenter1[2,thinned_array])/np.sqrt(2)
                obl_2 = (-detcenter2[1,thinned_array] + detcenter2[2,thinned_array])/np.sqrt(2)
                axs[1,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[obl_1,obl_2],c='y')
                axs[1,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(obl_2),np.max(obl_1)],color='k')
                axs[1,1].set_title('Oblique2 aperture 0')
                axs[1,1].set_xlabel('Axial (m)')
                axs[1,1].set_ylabel('Oblique2 Position (m)')
                plt.savefig('multiple_plots_ray_oblique.png', dpi=300)
                # plt.show()
                plt.close()
                fig, axs = plt.subplots(3, 2, figsize=(10, 15))
                # 範囲内の値を間引く
                start = 0
                end = ray_num
                thinned_array = crop(start, end, mabiki)



                axs[0,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='r')
                axs[0,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[0,0].set_title('Ray from V')
                axs[0,0].set_xlabel('Axial (m)')
                axs[0,0].set_ylabel('Horizontal Position (m)')

                axs[0,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='r')
                axs[0,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[0,1].set_title('Ray from H')
                axs[0,1].set_xlabel('Axial (m)')
                axs[0,1].set_ylabel('Vertical Position (m)')
                # plt.show()

                # 範囲内の値を間引く
                start = round(ray_num*(ray_num-1)/2)
                end = round(ray_num*(ray_num+1)/2)
                thinned_array = crop(start, end, mabiki)

                axs[1,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='y')
                axs[1,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[1,0].set_title('Ray from V')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Horizontal Position (m)')

                axs[1,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='y')
                axs[1,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[1,1].set_title('Ray from H')
                axs[1,1].set_xlabel('Axial (m)')
                axs[1,1].set_ylabel('Vertical Position (m)')

                # 範囲内の値を間引く
                start = ray_num**2 - ray_num
                end = ray_num**2
                thinned_array = crop(start, end, mabiki)

                axs[2,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='g')
                axs[2,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[2,0].set_title('Ray from V')
                axs[2,0].set_xlabel('Axial (m)')
                axs[2,0].set_ylabel('Horizontal Position (m)')

                axs[2,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='g')
                axs[2,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[2,1].set_title('Ray from H')
                axs[2,1].set_xlabel('Axial (m)')
                axs[2,1].set_ylabel('Vertical Position (m)')
                fig.suptitle('V aperture 0')
                plt.savefig('multiple_plots_ray_v.png', dpi=300)
                # plt.show()
                plt.close()


                fig, axs = plt.subplots(3, 2, figsize=(10, 15))
                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[::ray_num]
                thinned_array = first_thinned_array[::mabiki]
                fig.suptitle('H aperture 0')

                axs[0,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='r')
                axs[0,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[0,0].set_title('Ray from V')
                axs[0,0].set_xlabel('Axial (m)')
                axs[0,0].set_ylabel('Horizontal Position (m)')

                axs[0,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='r')
                axs[0,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[0,1].set_title('Ray from H')
                axs[0,1].set_xlabel('Axial (m)')
                axs[0,1].set_ylabel('Vertical Position (m)')


                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                first_thinned_array = original_array[round((ray_num-1)/2)::ray_num]

                # さらに間引き (skip_rateでさらに間引く)
                thinned_array = first_thinned_array[::mabiki]

                axs[1,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='y')
                axs[1,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[1,0].set_title('Ray from V')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Horizontal Position (m)')

                axs[1,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='y')
                axs[1,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[1,1].set_title('Ray from H')
                axs[1,1].set_xlabel('Axial (m)')
                axs[1,1].set_ylabel('Vertical Position (m)')

                # 範囲内の値を間引く
                original_array = list(range(len(detcenter1[0,:])))
                # 最初の間引き (ray_num-1から始めてray_numごとに要素を取得)
                first_thinned_array = original_array[ray_num-1::ray_num]

                # さらに間引き (skip_rateでさらに間引く)
                thinned_array = first_thinned_array[::mabiki]

                axs[2,0].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[1,thinned_array],detcenter2[1,thinned_array]],c='g')
                axs[2,0].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[1,:]),np.max(detcenter1[1,:])],color='k')
                # axs[2,0].set_title('Ray from V')
                axs[2,0].set_xlabel('Axial (m)')
                axs[2,0].set_ylabel('Horizontal Position (m)')

                axs[2,1].plot([detcenter1[0,thinned_array],detcenter2[0,thinned_array]],[detcenter1[2,thinned_array],detcenter2[2,thinned_array]],c='g')
                axs[2,1].plot([np.mean(detcenter[0,:]),np.mean(detcenter[0,:])],[np.min(detcenter2[2,:]),np.max(detcenter1[2,:])],color='k')
                # axs[2,1].set_title('Ray from H')
                axs[2,1].set_xlabel('Axial (m)')
                axs[2,1].set_ylabel('Vertical Position (m)')
                plt.savefig('multiple_plots_ray_h.png', dpi=300)
                # plt.show()
                plt.close()

                # plot_ray_sideview(8,10,mabiki,reflect1,vmirr_hyp,ray_num)
                # plot_ray_sideview(-5,35,mabiki,reflect1,vmirr_hyp,ray_num)
                # plot_ray_sideview(8,10,mabiki,reflect3,vmirr_ell,ray_num)
                # plot_ray_sideview(0.2,0.2,mabiki,reflect3,vmirr_ell,ray_num)

                phai0 = normalize_vector(phai0)

                plt.figure()
                plt.scatter(phai0[1, :], phai0[2, :])
                plt.scatter(phai0[1, ::ray_num], phai0[2, ::ray_num],color='r')
                plt.scatter(phai0[1, round((ray_num-1)/2)::ray_num], phai0[2, round((ray_num-1)/2)::ray_num],color='y')
                plt.scatter(phai0[1, ray_num-1::ray_num], phai0[2, ray_num-1::ray_num],color='g')
                plt.title('angle')
                plt.xlabel('Horizontal Angle (rad)')
                plt.ylabel('Vertical Angle (rad)')
                plt.axis('equal')
                # plt.show()
                plt.close()

                plt.figure()
                plt.scatter(phai0[1, :], phai0[2, :])
                plt.scatter((phai0[1, :ray_num]), (phai0[2, :ray_num]),color='r')
                plt.scatter(phai0[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)], phai0[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)],color='y')
                plt.scatter((phai0[1, -ray_num:]), (phai0[2, -ray_num:]),color='g')
                plt.title('angle')
                plt.xlabel('Horizontal Angle (rad)')
                plt.ylabel('Vertical Angle (rad)')
                plt.axis('equal')
                # plt.show()
                plt.close()

                vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
                reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))

                angle_1st, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect1,norm_vector(coeffs_hyp_v, vmirr_hyp))



                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)

                detcenter0 = plane_ray_intersection(coeffs_det, reflect4, hmirr_ell)

                angles_between_rad, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect_ray(reflect4, norm_vector(coeffs_det, detcenter0)),norm_vector(coeffs_det, detcenter0))

                if option_tilt:
                    hmirr_hyp0 = hmirr_ell
                    reflect4_rotated = rotate_vectors(reflect4, -theta_y, -theta_z)
                    focus_apprx = np.mean(detcenter,axis=1)
                    hmirr_ell_points_rotated = rotate_points(hmirr_ell, focus_apprx, -theta_y, -theta_z)
                    coeffs_det = np.zeros(10)
                    coeffs_det[6] = 1
                    coeffs_det[9] = -(s2f_middle + defocus)
                    detcenter = plane_ray_intersection(coeffs_det, reflect4_rotated, hmirr_ell_points_rotated)
                    hmirr_ell = hmirr_ell_points_rotated
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
                axs[0, 0].plot([detcenter1[0, thinned_array_h_r[:third_r]], detcenter2[0, thinned_array_h_r[:third_r]]],
                               [detcenter1[1, thinned_array_h_r[:third_r]], detcenter2[1, thinned_array_h_r[:third_r]]], c='r')

                # 後ろ1/3のプロット（ピンク）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_r[-third_r:]], detcenter2[0, thinned_array_h_r[-third_r:]]],
                               [detcenter1[1, thinned_array_h_r[-third_r:]], detcenter2[1, thinned_array_h_r[-third_r:]]], c='purple')

                # thinned_array_h_y の前1/3のプロット（darkyellow）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_y[:third_y]], detcenter2[0, thinned_array_h_y[:third_y]]],
                               [detcenter1[1, thinned_array_h_y[:third_y]], detcenter2[1, thinned_array_h_y[:third_y]]], c='y')

                # thinned_array_h_y の後ろ1/3のプロット（purple）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_y[-third_y:]], detcenter2[0, thinned_array_h_y[-third_y:]]],
                               [detcenter1[1, thinned_array_h_y[-third_y:]], detcenter2[1, thinned_array_h_y[-third_y:]]], c='#B8860B')

                # thinned_array_h_g の前1/3のプロット（緑）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_g[:third_g]], detcenter2[0, thinned_array_h_g[:third_g]]],
                               [detcenter1[1, thinned_array_h_g[:third_g]], detcenter2[1, thinned_array_h_g[:third_g]]], c='g')

                # thinned_array_h_g の後ろ1/3のプロット（薄緑）
                axs[0, 0].plot([detcenter1[0, thinned_array_h_g[-third_g:]], detcenter2[0, thinned_array_h_g[-third_g:]]],
                               [detcenter1[1, thinned_array_h_g[-third_g:]], detcenter2[1, thinned_array_h_g[-third_g:]]], c='lightgreen')
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
                axs[1,1].scatter(np.mean(detcenter[1, ::ray_num]), np.mean(detcenter[2, ::ray_num]),color='r',marker='x',s=100)
                axs[1,1].scatter(np.mean(detcenter[1, round((ray_num-1)/2)::ray_num]), np.mean(detcenter[2, round((ray_num-1)/2)::ray_num]),color='y',marker='x',s=100)
                axs[1,1].scatter(np.mean(detcenter[1, ray_num-1::ray_num]), np.mean(detcenter[2, ray_num-1::ray_num]),color='g',marker='x',s=100)
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
                axs[1, 0].plot([detcenter1[0, thinned_array_v_r[:third_r]], detcenter2[0, thinned_array_v_r[:third_r]]],
                               [detcenter1[2, thinned_array_v_r[:third_r]], detcenter2[2, thinned_array_v_r[:third_r]]], c='r')

                # 後ろ1/3のプロット（ピンク）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_r[-third_r:]], detcenter2[0, thinned_array_v_r[-third_r:]]],
                               [detcenter1[2, thinned_array_v_r[-third_r:]], detcenter2[2, thinned_array_v_r[-third_r:]]], c='purple')

                # thinned_array_v_y の前1/3のプロット（darkyellow）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_y[:third_y]], detcenter2[0, thinned_array_v_y[:third_y]]],
                               [detcenter1[2, thinned_array_v_y[:third_y]], detcenter2[2, thinned_array_v_y[:third_y]]], c='y')

                # thinned_array_v_y の後ろ1/3のプロット（purple）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_y[-third_y:]], detcenter2[0, thinned_array_v_y[-third_y:]]],
                               [detcenter1[2, thinned_array_v_y[-third_y:]], detcenter2[2, thinned_array_v_y[-third_y:]]], c='#B8860B')

                # thinned_array_v_g の前1/3のプロット（緑）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_g[:third_g]], detcenter2[0, thinned_array_v_g[:third_g]]],
                               [detcenter1[2, thinned_array_v_g[:third_g]], detcenter2[2, thinned_array_v_g[:third_g]]], c='g')

                # thinned_array_v_g の後ろ1/3のプロット（薄緑）
                axs[1, 0].plot([detcenter1[0, thinned_array_v_g[-third_g:]], detcenter2[0, thinned_array_v_g[-third_g:]]],
                               [detcenter1[2, thinned_array_v_g[-third_g:]], detcenter2[2, thinned_array_v_g[-third_g:]]], c='lightgreen')

                axs[1,0].plot([input_val, input_val],
                            [np.min(detcenter2[2, :]), np.max(detcenter1[2, :])], color='k')
                axs[1,0].set_title('H aperture 0')
                axs[1,0].set_xlabel('Axial (m)')
                axs[1,0].set_ylabel('Vertical Position (m)')

                axs[0,1].scatter(detcenter[1, :], detcenter[2, :])
                axs[0,1].scatter(detcenter[1, :ray_num], detcenter[2, :ray_num],color='r')
                axs[0,1].scatter(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)], detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)],color='y')
                axs[0,1].scatter(detcenter[1, -ray_num:], detcenter[2, -ray_num:],color='g')
                axs[0,1].scatter(detcenter[1, ray_num-1::ray_num-1][:-1], detcenter[2, ray_num-1::ray_num-1][:-1],color='k')
                axs[0,1].scatter(detcenter[1, ::ray_num+1], detcenter[2, ::ray_num+1],color='gray')
                axs[0,1].scatter(np.mean(detcenter[1, :ray_num]), np.mean(detcenter[2, :ray_num]),color='r',marker='x',s=100)
                axs[0,1].scatter(np.mean(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]), np.mean(detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]),color='y',marker='x',s=100)
                axs[0,1].scatter(np.mean(detcenter[1, -ray_num:]), np.mean(detcenter[2, -ray_num:]),color='g',marker='x',s=100)
                axs[0,1].set_title('focus @V aperture 0')
                axs[0,1].set_xlabel('Horizontal (m)')
                axs[0,1].set_ylabel('Vertical (m)')
                axs[0,1].axis('equal')

                # タイトル用の新しいサイズ計算
                size_v = np.max(detcenter[2,:]) - np.min(detcenter[2,:])
                size_h = np.max(detcenter[1,:]) - np.min(detcenter[1,:])

                # タイトルの更新
                title1 = f'Params 0-1: {params[0:2]}'
                title2 = f'Params 2-7: {params[2:8]}'
                title3 = f'Params 8-13: {params[8:14]}'
                title4 = f'Params 14-19: {params[14:20]}'
                title5 = f'Params 20-25: {params[20:26]}'
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
                        detcenter = plane_ray_intersection(coeffs_det, reflect4, hmirr_ell)
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
                        axs[1,1].scatter(np.mean(detcenter[1, ::ray_num]), np.mean(detcenter[2, ::ray_num]),color='r',marker='x',s=100)
                        axs[1,1].scatter(np.mean(detcenter[1, round((ray_num-1)/2)::ray_num]), np.mean(detcenter[2, round((ray_num-1)/2)::ray_num]),color='y',marker='x',s=100)
                        axs[1,1].scatter(np.mean(detcenter[1, ray_num-1::ray_num]), np.mean(detcenter[2, ray_num-1::ray_num]),color='g',marker='x',s=100)
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
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_r[:third_r]], detcenter2[0, thinned_array_h_r[:third_r]]],
                                       [detcenter1[1, thinned_array_h_r[:third_r]], detcenter2[1, thinned_array_h_r[:third_r]]], c='r')

                        # 後ろ1/3のプロット（ピンク）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_r[-third_r:]], detcenter2[0, thinned_array_h_r[-third_r:]]],
                                       [detcenter1[1, thinned_array_h_r[-third_r:]], detcenter2[1, thinned_array_h_r[-third_r:]]], c='purple')

                        # thinned_array_h_y の前1/3のプロット（darkyellow）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_y[:third_y]], detcenter2[0, thinned_array_h_y[:third_y]]],
                                       [detcenter1[1, thinned_array_h_y[:third_y]], detcenter2[1, thinned_array_h_y[:third_y]]], c='y')

                        # thinned_array_h_y の後ろ1/3のプロット（purple）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_y[-third_y:]], detcenter2[0, thinned_array_h_y[-third_y:]]],
                                       [detcenter1[1, thinned_array_h_y[-third_y:]], detcenter2[1, thinned_array_h_y[-third_y:]]], c='#B8860B')

                        # thinned_array_h_g の前1/3のプロット（緑）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_g[:third_g]], detcenter2[0, thinned_array_h_g[:third_g]]],
                                       [detcenter1[1, thinned_array_h_g[:third_g]], detcenter2[1, thinned_array_h_g[:third_g]]], c='g')

                        # thinned_array_h_g の後ろ1/3のプロット（薄緑）
                        axs[0, 0].plot([detcenter1[0, thinned_array_h_g[-third_g:]], detcenter2[0, thinned_array_h_g[-third_g:]]],
                                       [detcenter1[1, thinned_array_h_g[-third_g:]], detcenter2[1, thinned_array_h_g[-third_g:]]], c='lightgreen')
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
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_r[:third_r]], detcenter2[0, thinned_array_v_r[:third_r]]],
                                       [detcenter1[2, thinned_array_v_r[:third_r]], detcenter2[2, thinned_array_v_r[:third_r]]], c='r')

                        # 後ろ1/3のプロット（ピンク）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_r[-third_r:]], detcenter2[0, thinned_array_v_r[-third_r:]]],
                                       [detcenter1[2, thinned_array_v_r[-third_r:]], detcenter2[2, thinned_array_v_r[-third_r:]]], c='purple')

                        # thinned_array_v_y の前1/3のプロット（darkyellow）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_y[:third_y]], detcenter2[0, thinned_array_v_y[:third_y]]],
                                       [detcenter1[2, thinned_array_v_y[:third_y]], detcenter2[2, thinned_array_v_y[:third_y]]], c='y')

                        # thinned_array_v_y の後ろ1/3のプロット（purple）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_y[-third_y:]], detcenter2[0, thinned_array_v_y[-third_y:]]],
                                       [detcenter1[2, thinned_array_v_y[-third_y:]], detcenter2[2, thinned_array_v_y[-third_y:]]], c='#B8860B')

                        # thinned_array_v_g の前1/3のプロット（緑）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_g[:third_g]], detcenter2[0, thinned_array_v_g[:third_g]]],
                                       [detcenter1[2, thinned_array_v_g[:third_g]], detcenter2[2, thinned_array_v_g[:third_g]]], c='g')

                        # thinned_array_v_g の後ろ1/3のプロット（薄緑）
                        axs[1, 0].plot([detcenter1[0, thinned_array_v_g[-third_g:]], detcenter2[0, thinned_array_v_g[-third_g:]]],
                                       [detcenter1[2, thinned_array_v_g[-third_g:]], detcenter2[2, thinned_array_v_g[-third_g:]]], c='lightgreen')

                        axs[1,0].plot([input_val, input_val],
                                    [np.min(detcenter2[2, :]), np.max(detcenter1[2, :])], color='k')
                        axs[1,0].set_title('H aperture 0')
                        axs[1,0].set_xlabel('Axial (m)')
                        axs[1,0].set_ylabel('Vertical Position (m)')

                        axs[0,1].cla()  # 右側プロットをクリア
                        axs[0,1].scatter(detcenter[1, :], detcenter[2, :])
                        axs[0,1].scatter(detcenter[1, :ray_num], detcenter[2, :ray_num],color='r')
                        axs[0,1].scatter(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)], detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)],color='y')
                        axs[0,1].scatter(detcenter[1, -ray_num:], detcenter[2, -ray_num:],color='g')
                        axs[0,1].scatter(detcenter[1, ray_num-1::ray_num-1][:-1], detcenter[2, ray_num-1::ray_num-1][:-1],color='k')
                        axs[0,1].scatter(detcenter[1, ::ray_num+1], detcenter[2, ::ray_num+1],color='gray')
                        axs[0,1].scatter(np.mean(detcenter[1, :ray_num]), np.mean(detcenter[2, :ray_num]),color='r',marker='x',s=100)
                        axs[0,1].scatter(np.mean(detcenter[1, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]), np.mean(detcenter[2, round(ray_num*(ray_num-1)/2) : round(ray_num*(ray_num+1)/2)]),color='y',marker='x',s=100)
                        axs[0,1].scatter(np.mean(detcenter[1, -ray_num:]), np.mean(detcenter[2, -ray_num:]),color='g',marker='x',s=100)
                        axs[0,1].set_title('focus @V aperture 0')
                        axs[0,1].set_xlabel('Horizontal (m)')
                        axs[0,1].set_ylabel('Vertical (m)')
                        axs[0,1].axis('equal')

                        # axs[0,2].scatter(input_val,np.mean(detcenter[1, :ray_num]),color='r')
                        # axs[0,2].scatter(input_val,np.mean(detcenter[1, round((ray_num**2)/2) : round((ray_num**2 + ray_num*2)/2)]),color='y')
                        # axs[0,2].scatter(input_val,np.mean(detcenter[1, -ray_num:-1]),color='g')

                        # axs[1,2].scatter(input_val,np.mean(detcenter[2, ::ray_num]),color='r')
                        # axs[1,2].scatter(input_val,np.mean(detcenter[2, round(ray_num/2)-1::ray_num]),color='y')
                        # axs[1,2].scatter(input_val,np.mean(detcenter[2, ray_num-1::ray_num]),color='g')
                        # axs[1,3].scatter(input_val,np.mean(detcenter[1, ::ray_num]),color='r')
                        # axs[1,3].scatter(input_val,np.mean(detcenter[1, round(ray_num/2)-1::ray_num]),color='y')
                        # axs[1,3].scatter(input_val,np.mean(detcenter[1, ray_num-1::ray_num]),color='g')

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
                plt.savefig('multipleAroundFocus.png', dpi=300)
                plt.show()

                print('NA_h')
                print(np.sin((np.max(angles_yx_rad) - np.min(angles_yx_rad))/2))
                print('NA_v')
                print(np.sin((np.max(angles_zx_rad) - np.min(angles_zx_rad))/2))


    # 焦点面での標準偏差を計算
        return vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle


#####################################
def find_defocus(rays, points, s2f_middle,defocus,ray_num):
    # 初期のパラメータ設定
    initial_a_min = -0.3   # a の最初の最小値
    initial_a_max = 0.3   # a の最初の最大値
    shrink_factor = 0.1      # 各ループで範囲を縮小する割合
    num_adj_astg = 50        # 各範囲でのステップ数
    max_loops = 10            # ループ回数 (収束条件により調整)

    # 初期の範囲を設定
    a_min = initial_a_min
    a_max = initial_a_max
    # if option_param == 'D':
    #     distance_ = np.linspace(-0.01, 0.01,num_adj_astg);
    # ループ処理を繰り返す
    for loop in range(max_loops):
        # print(f"ループ {loop+1} 回目: a の範囲 [{a_min}, {a_max}]")
        # a のリストを生成
        a = np.linspace(a_min, a_max, num_adj_astg)
        # size_v_ と size_h_ を初期化
        size_v_ = np.zeros(num_adj_astg)
        size_h_ = np.zeros(num_adj_astg)
        # size_v_ と size_h_ を初期化
        size_v_sep1 = np.zeros((3,num_adj_astg))
        size_h_sep1 = np.zeros((3,num_adj_astg))
        size_v_sep2 = np.zeros((3,num_adj_astg))
        size_h_sep2 = np.zeros((3,num_adj_astg))
        # 各 a の値に対して計算
        for i in range(len(a)):
            coeffs_det = np.zeros(10)
            coeffs_det[6] = 1
            coeffs_det[9] = -(s2f_middle + a[i])
            detcenter = plane_ray_intersection(coeffs_det, rays, points)
            size_v_[i] = np.std(detcenter[2, :])
            size_h_[i] = np.std(detcenter[1, :])

            size_v_sep2[0,i] = np.std(detcenter[2, ::ray_num])
            size_v_sep2[1,i] = np.std(detcenter[2, round(ray_num/2)-1::ray_num])
            size_v_sep2[2,i] = np.std(detcenter[2, ray_num-1::ray_num])
            size_h_sep2[0,i] = np.std(detcenter[1, ::ray_num])
            size_h_sep2[1,i] = np.std(detcenter[1, round(ray_num/2)-1::ray_num])
            size_h_sep2[2,i] = np.std(detcenter[1, ray_num-1::ray_num])

            size_v_sep1[0,i] = np.std(detcenter[2, :ray_num])
            size_v_sep1[1,i] = np.std(detcenter[2, round((ray_num**2)/2) : round((ray_num**2 + ray_num*2)/2)])
            size_v_sep1[2,i] = np.std(detcenter[2, -ray_num:-1])
            size_h_sep1[0,i] = np.std(detcenter[2, :ray_num])
            size_h_sep1[1,i] = np.std(detcenter[2, round((ray_num**2)/2) : round((ray_num**2 + ray_num*2)/2)])
            size_h_sep1[2,i] = np.std(detcenter[2, -ray_num:-1])

        # # 最小値を取得
        # size_v_param = np.min(size_v_)
        # size_h_param = np.min(size_h_)

        # a の範囲を更新
        best_a = (a[np.argmin(size_h_)] + a[np.argmin(size_v_)])/2
        delta_a = (a_max - a_min) * shrink_factor
        a_min = best_a - delta_a / 2
        a_max = best_a + delta_a / 2

        # fig, axs = plt.subplots(3, 2, figsize=(10, 15))
        # axs[0,0].plot(a,size_v_sep1[0,:],c='r')
        # axs[0,0].plot(a,size_v_sep1[1,:],c='y')
        # axs[0,0].plot(a,size_v_sep1[2,:],c='g')
        # axs[0,0].plot(a,size_v_,c='k',linestyle='--')
        # axs[0,0].set_ylabel('Vertical @V aperture 0')
        #
        # axs[0,1].plot(a,size_h_sep1[0,:],c='r')
        # axs[0,1].plot(a,size_h_sep1[1,:],c='y')
        # axs[0,1].plot(a,size_h_sep1[2,:],c='g')
        # axs[0,1].plot(a,size_h_,c='k',linestyle='--')
        # axs[0,1].set_ylabel('Horizontal @V aperture 0')
        #
        # axs[1,0].plot(a,size_v_sep2[0,:],c='r')
        # axs[1,0].plot(a,size_v_sep2[1,:],c='y')
        # axs[1,0].plot(a,size_v_sep2[2,:],c='g')
        # axs[1,0].plot(a,size_v_,c='k',linestyle='--')
        # axs[1,0].set_ylabel('Vertical @H aperture 0')
        #
        # axs[1,1].plot(a,size_h_sep2[0,:],c='r')
        # axs[1,1].plot(a,size_h_sep2[1,:],c='y')
        # axs[1,1].plot(a,size_h_sep2[2,:],c='g')
        # axs[1,1].plot(a,size_h_,c='k',linestyle='--')
        # axs[1,1].set_ylabel('Horizontal @H aperture 0')
        # plt.show()


    return best_a

def optimize_min_index(func, x_min, x_max, num_steps=100, shrink_factor=0.1, max_attempts=20, tolerance=1e-13):
    """
    指定された関数の最小値のインデックスを探索します。

    Parameters:
        func (callable): 入力xに対して出力yを返す関数f(x)。
        x_min (float): xの初期範囲の最小値。
        x_max (float): xの初期範囲の最大値。
        num_steps (int): 各範囲内で分割するステップ数。
        shrink_factor (float): 範囲を縮小する割合 (0.0 < shrink_factor < 1.0)。
        max_attempts (int): 最大試行回数。
        tolerance (float): 収束条件とする最小範囲の幅。

    Returns:
        best_x (float): 最適なxの値（最小値のインデックス）。
        min_y (float): 最適なxでの関数値。
    """
    attempt = 0
    while attempt < max_attempts:
        # xの候補値を生成
        x_values = np.linspace(x_min, x_max, num_steps)

        # 関数値を計算
        y_values = np.array([func(x) for x in x_values])  # スカラーごとに評価

        # 最小値のインデックスを取得
        min_index = np.argmin(y_values)
        best_x = x_values[min_index]
        min_y = y_values[min_index]

        # 範囲を更新
        delta_x = (x_max - x_min) * shrink_factor
        x_min = best_x - delta_x / 2
        x_max = best_x + delta_x / 2

        # 終了条件
        if (x_max - x_min) < tolerance and x_max - x_min > 1e-16:
            break

        attempt += 1
    if (x_max - x_min) > tolerance:
        print('x_max - x_min',x_max - x_min)
    # 結果を返す
    return best_x, min_y

def create_func_to_minimize(coeffs_det, rays, points, evaluation_fn, thinned_array):
    """
    func_to_minimize を生成する工場関数。

    Parameters:
        coeffs_det (array): 最適化対象の係数配列。
        rays (array): 光線のデータ。
        points (array): 点のデータ。
        evaluation_fn (callable): detcenter_2r から評価値を計算する関数。
        thinned_array_v_r (array): 評価に使うインデックスの配列。

    Returns:
        callable: 最適化用の目的関数。
    """
    def func_to_minimize(a_value):
        coeffs_det[6] = 1.
        coeffs_det[9] = a_value  # 最適化パラメータ
        detcenter = plane_ray_intersection(coeffs_det, rays[:, thinned_array], points[:, thinned_array])
        return evaluation_fn(detcenter)  # 外部で指定した評価関数を適用

    return func_to_minimize

def create_evaluation_fn():
    """
    `i` 番目のインデックスに対して評価関数を生成。

    Parameters:
        i (int): 評価するインデックス。
        mode (str): 'std' または 'sep1_r'。'std' は標準偏差最小化、'sep1_r' は `size_sep1_r` 最小化。

    Returns:
        callable: 引数 `detcenter_1r` の指定インデックス `i` を評価する関数。
    """
    def evaluate_fn(detcenter):
        size_v = np.std(detcenter[2, :])
        size_h = np.std(detcenter[1, :])
        size_sep1_r = np.sqrt(size_v**2 + size_h**2)
        return size_sep1_r
        # if mode == 'std':
        #     return size_v  # 標準偏差を返す
        # elif mode == 'sep1_r':
        #     size_sep1_r = np.sqrt(size_v**2 + size_h**2)
        #     return size_sep1_r  # size_sep1_rを返す
        # else:
        #     raise ValueError("mode must be 'std' or 'sep1_r'")

    return evaluate_fn

def compare_sep(rays, points,coeffs_det0,ray_num,region):
    # print(coeffs_det0)
    num = 10000
    x_min = coeffs_det0[9] - 1e-2
    x_max = coeffs_det0[9] + 1e-2


    # 範囲内の値を間引く
    original_array = list(range(ray_num**2))
    thinned_array_v_r = original_array[::ray_num]

    # 範囲内の値を間引く
    original_array = list(range(ray_num**2))
    thinned_array_v_y = original_array[round((ray_num-1)/2)::ray_num]

    # 範囲内の値を間引く
    original_array = list(range(ray_num**2))
    thinned_array_v_g = original_array[ray_num-1::ray_num]

    third_v_r = len(thinned_array_v_r) *2 // 3
    third_v_y = len(thinned_array_v_y) *2 // 3
    third_v_g = len(thinned_array_v_g) *2 // 3

    # 範囲内の値を間引く
    start = 0
    end = ray_num
    thinned_array_h_r = crop(start, end, 1)
    # 範囲内の値を間引く
    start = round(ray_num*(ray_num-1)/2)
    end = round(ray_num*(ray_num+1)/2)
    thinned_array_h_y = crop(start, end, 1)
    # 範囲内の値を間引く
    start = ray_num**2 - ray_num
    end = ray_num**2
    thinned_array_h_g = crop(start, end, 1)

    third_h_r = len(thinned_array_h_r) *2 // 3
    third_h_y = len(thinned_array_h_y) *2 // 3
    third_h_g = len(thinned_array_h_g) *2 // 3

    y_values1 = np.empty(ray_num)
    y_values2 = np.empty(ray_num)

    angle_h = np.arctan(rays[1, :]/rays[0, :])
    angle_v = np.arctan(rays[2, :]/rays[0, :])
    angle_v_sep_g = angle_v[thinned_array_v_g]
    angle_v_sep_y = angle_v[thinned_array_v_y]
    angle_v_sep_r = angle_v[thinned_array_v_r]

    angle_h_sep_g = angle_h[thinned_array_h_g]
    angle_h_sep_y = angle_h[thinned_array_h_y]
    angle_h_sep_r = angle_h[thinned_array_h_r]

    # 範囲内の値を間引く
    original_array = list(range(ray_num**2))
    thinned_array_v_y = original_array[round((ray_num-1)/2)::ray_num]
    # 範囲内の値を間引く
    start = round(ray_num*(ray_num-1)/2)
    end = round(ray_num*(ray_num+1)/2)
    thinned_array_h_y = crop(start, end, 1)



    # 目的関数を生成
    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_h_r
    )
    focus_v01, std_v01 = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_h_y
    )
    focus_v02, std_v02 = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_h_g
    )
    focus_v03, std_v03 = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_v_r
    )
    focus_h01, std_h01 = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_v_y
    )
    focus_h02, std_h02 = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_v_g
    )
    focus_h03, std_h03 = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    focus_v0 = np.array([focus_v01,focus_v02,focus_v03])
    focus_h0 = np.array([focus_h01,focus_h02,focus_h03])
    std_v0 = np.array([std_v01,std_v02,std_v03])
    std_h0 = np.array([std_h01,std_h02,std_h03])

    # 目的関数を生成
    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_h_r[:third_h_r]
    )
    focus_v01_l, std_v01_l = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_h_y[:third_h_y]
    )
    focus_v02_l, std_v02_l = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_h_g[:third_h_g]
    )
    focus_v03_l, std_v03_l = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_v_r[:third_v_r]
    )
    focus_h01_l, std_h01_l = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_v_y[:third_v_y]
    )
    focus_h02_l, std_h02_l = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_v_g[:third_v_g]
    )
    focus_h03_l, std_h03_l = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    focus_v0_l = np.array([focus_v01_l,focus_v02_l,focus_v03_l])
    focus_h0_l = np.array([focus_h01_l,focus_h02_l,focus_h03_l])

    # 目的関数を生成
    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_h_r[:-third_h_r]
    )
    focus_v01_u, std_v01_u = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_h_y[:-third_h_y]
    )
    focus_v02_u, std_v02_u = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_h_g[:-third_h_g]
    )
    focus_v03_u, std_v03_u = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_v_r[:-third_v_r]
    )
    focus_h01_u, std_h01_u = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_v_y[:-third_v_y]
    )
    focus_h02_u, std_h02_u = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=thinned_array_v_g[:-third_v_g]
    )
    focus_h03_u, std_h03_u = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)

    focus_v0_u = np.array([focus_v01_u,focus_v02_u,focus_v03_u])
    focus_h0_u = np.array([focus_h01_u,focus_h02_u,focus_h03_u])

    num_rays = rays.shape[1]  # 列数（光線の数）

    # ray_num-1 ごとのインデックスを作成
    indices = np.arange(ray_num-1, num_rays, ray_num-1)

    # 最後の要素を削除
    indices = indices[:-1]  # 最後のインデックスを削除

    # スライスされた配列を取得
    sliced_rays = rays[:, indices]

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=indices
    )
    focus_obl1, std_obl1 = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)
    focus_std_obl1 = focus_obl1
    num_rays = rays.shape[1]  # 列数（光線の数）

    # 0 から num_rays まで、ray_num+1 ごとのインデックスを生成
    indices = np.arange(0, num_rays, ray_num+1)

    func_to_minimize = create_func_to_minimize(
        coeffs_det=coeffs_det0,
        rays=rays,
        points=points,
        evaluation_fn=create_evaluation_fn(),  # 動的に指定された評価関数
        thinned_array=indices
    )
    focus_obl2, std_obl2 = optimize_min_index(func_to_minimize, x_min=x_min, x_max=x_max)
    focus_std_obl2 = focus_obl2


    detcenter = plane_ray_intersection(coeffs_det0, rays, points)

    detcenter_2r = detcenter[:,::ray_num]
    detcenter_2y = detcenter[:,round((ray_num-1)/2)::ray_num]
    detcenter_2g = detcenter[:,ray_num-1::ray_num]

    detcenter_1r = detcenter[:,:ray_num]
    detcenter_1y = detcenter[:,round(ray_num*(ray_num-1)/2): round(ray_num*(ray_num+1)/2)]
    detcenter_1g = detcenter[:,-ray_num:]

    detcenter_c = np.mean(detcenter,axis=1)

    pos_v0 = np.array([[np.mean(detcenter_1r,axis=1)],[np.mean(detcenter_1y,axis=1)],[np.mean(detcenter_1g,axis=1)]])
    pos_h0 = np.array([[np.mean(detcenter_2r,axis=1)],[np.mean(detcenter_2y,axis=1)],[np.mean(detcenter_2g,axis=1)]])

    return focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2

#####################################
def apply_rotation_to_conic(coeffs, R):
    a, b, c, d, e, f, g, h, i, j = coeffs

    # 回転行列を使って新しい係数を計算
    # 新しい x', y', z' に関する計算

    # x^2, y^2, z^2 項の変換
    # x^2 の新しい係数 a'
    a_prime = a * R[0, 0]**2 + b * R[0, 1]**2 + c * R[0, 2]**2 + d*R[0,0]*R[0,1] + e*R[0,0]*R[0,2] + f*R[0,1]*R[0,2]
    b_prime = a * (R[0, 0] * R[0, 1] + R[1, 0] * R[1, 1] + R[2, 0] * R[2, 1]) + b * (R[0, 1]**2 + R[1, 1]**2 + R[2, 1]**2) + c * (R[0, 2] * R[0, 1] + R[1, 2] * R[1, 1] + R[2, 2] * R[2, 1])
    c_prime = a * (R[0, 0] * R[0, 2] + R[1, 0] * R[1, 2] + R[2, 0] * R[2, 2]) + b * (R[0, 1] * R[0, 2] + R[1, 1] * R[1, 2] + R[2, 1] * R[2, 2]) + c * (R[0, 2]**2 + R[1, 2]**2 + R[2, 2]**2)

    # x * y, x * z, y * z 項の変換
    d_prime = d * (R[0, 0] * R[1, 0] + R[0, 1] * R[1, 1] + R[0, 2] * R[1, 2])
    e_prime = e * (R[0, 0] * R[2, 0] + R[0, 1] * R[2, 1] + R[0, 2] * R[2, 2])
    f_prime = f * (R[1, 0] * R[2, 0] + R[1, 1] * R[2, 1] + R[1, 2] * R[2, 2])

    # x, y, z 項の変換
    g_prime = g * (R[0, 0] + R[0, 1] + R[0, 2])
    h_prime = h * (R[1, 0] + R[1, 1] + R[1, 2])
    i_prime = i * (R[2, 0] + R[2, 1] + R[2, 2])

    # 定数項の変換
    j_prime = j  # 定数項は変わりません

    return [a_prime, b_prime, c_prime, d_prime, e_prime, f_prime, g_prime, h_prime, i_prime, j_prime]
def axis_rotation(coeffs, axis, theta, center):
    """
    曲面を指定した軸(axis)周りに回転させる関数
    coeffs: 曲面の係数
    axis: 回転軸（3次元ベクトル）
    theta: 回転角度（ラジアン）
    center: 回転の中心
    """
    # 軸の正規化
    axis = axis / np.linalg.norm(axis)

    # 回転行列を計算するためのスカラー値とベクトル成分
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    # ux, uy, uz = axis
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]

    # 回転行列（ロドリゲスの回転公式）
    rotation_matrix = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ]).T
    print('rotation_matrix',rotation_matrix)
    # 回転を行うために中心を原点に移動させてから回転を適用
    coeffs = shift_x(coeffs, -center[0])
    coeffs = shift_y(coeffs, -center[1])
    coeffs = shift_z(coeffs, -center[2])

    # 回転を適用
    new_coords = apply_rotation_to_conic(coeffs, rotation_matrix)

    # 元の位置に戻す

    coeffs = shift_x(coeffs, center[0])
    coeffs = shift_y(coeffs, center[1])
    coeffs = shift_z(coeffs, center[2])

    return coeffs, rotation_matrix
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

class KBDesignManager:
    def __init__(self):
        self.Ell1 = None
        self.Ell2 = None

    def set_design(self, l_i1, l_o1, theta_g1, na_o_sin, target_l_o2, target_gap, ast):
        self.l_i1 = np.float64(l_i1)
        self.l_o1 = np.float64(l_o1)
        self.theta_g1 = np.float64(theta_g1)
        self.na_o_sin = np.float64(na_o_sin)
        self.na_o_sin_v = np.float64(na_o_sin)
        self.na_o_sin_h = np.float64(na_o_sin)
        self.target_l_o2 = np.float64(target_l_o2)  # WD
        self.target_gap = np.float64(target_gap)
        self.ast = np.float64(ast)

    def get_design(self):
        if self.Ell1 is None or self.Ell2 is None:
            # self.l_i1 = np.float64(145.7500024376426)
            # self.l_o1 = np.float64(1.0499975623574187)
            # # self.theta_g1 = np.float64(0.026207694380368848)*10
            # self.theta_g1 = np.float64(0.21093460929217367)
            # self.na_o_sin = np.float64(0.01)*8.2
            # self.target_l_o2 = np.float64(0.3199999032106779)  # WD
            # self.target_gap = np.float64(0.22868328628430845)
            # # self.target_gap = np.float64(0.3658793853926454)
            # self.ast = np.float64(0.)
            # self.l_i1 = np.float64(145.750155679735)
            # self.l_o1 = np.float64(1.049844320265004)
            # self.theta_g1 = np.float64(0.21093460929217367)
            # self.na_o_sin = np.float64(0.082)
            # self.target_l_o2 = np.float64(0.3199948756376758)  # WD
            # self.target_gap = np.float64(0.3658793853926454)
            # self.ast = np.float64(0.)
            # ファイルに保存
            with open("ParamsDesign.txt", "w") as file:
                file.write(f"l_i1 = {self.l_i1}\n")
                file.write(f"l_o1 = {self.l_o1}\n")
                file.write(f"theta_g1 = {self.theta_g1}\n")
                file.write(f"na_o_sin = {self.na_o_sin}\n")
                file.write(f"target_l_o2 = {self.target_l_o2}\n")
                file.write(f"target_gap = {self.target_gap}\n")
                file.write(f"ast = {self.ast}\n")
            # self.Ell1, self.Ell2 = KB_design_NAbased.KB_design(self.l_i1, self.l_o1, self.theta_g1, self.na_o_sin, self.target_l_o2, self.target_gap, self.ast)
            self.Ell1, self.Ell2 = KB_design_NAbased.KB_design(self.l_i1, self.l_o1, self.theta_g1, self.na_o_sin_v, self.na_o_sin_h, self.target_l_o2, self.target_gap, self.ast)
        return self.Ell1, self.Ell2
kb_manager = KBDesignManager()
def KB_debug(params,na_ratio_h,na_ratio_v,option):
    defocus, astigH, \
    pitch_hyp_v, roll_hyp_v, yaw_hyp_v, decenterX_hyp_v, decenterY_hyp_v, decenterZ_hyp_v,\
    pitch_hyp_h, roll_hyp_h, yaw_hyp_h, decenterX_hyp_h, decenterY_hyp_h, decenterZ_hyp_h,\
    pitch_ell_v, roll_ell_v, yaw_ell_v, decenterX_ell_v, decenterY_ell_v, decenterZ_ell_v,\
    pitch_ell_h, roll_ell_h, yaw_ell_h, decenterX_ell_h, decenterY_ell_h, decenterZ_ell_h  = params
    if optKBdesign:
        # print('KB　最適化')
        # l_i1 = np.float64(145.7500024376426)
        # l_o1 = np.float64(1.0499975623574187)
        # theta_g1 = np.float64(0.211)
        # na_o_sin = np.float64(0.082)
        # target_l_o2 = np.float64(0.04) ### WD
        # target_gap = np.float64(0.02)
        # ast = np.float64(0.)
        # Ell1, Ell2 = KB_design_NAbased.KB_design(l_i1, l_o1, theta_g1, na_o_sin,target_l_o2, target_gap, ast)
        Ell1, Ell2 = kb_manager.get_design()
        a_hyp_v = Ell1.a
        b_hyp_v = Ell1.b
        a_hyp_h = Ell2.a
        b_hyp_h = Ell2.b

        org_hyp_v = np.sqrt(a_hyp_v**2 - b_hyp_v**2)
        org_hyp_h = np.sqrt(a_hyp_h**2 - b_hyp_h**2)

        y1_v = Ell1.y_1
        x1_v = Ell1.x_1
        y2_v = Ell1.y_2
        x2_v = Ell1.edge

        y1_h = Ell2.y_1
        x1_h = Ell2.x_1
        y2_h = Ell2.y_2
        x2_h = Ell2.edge
        ### ミラー長き基準中心
        theta1_v = Ell1.theta_i_cnt_m_wid
        theta1_h = Ell2.theta_i_cnt_m_wid
        # ### 入射光基準
        # theta1_v = Ell1.theta_i_cnt_angle
        # theta1_h = Ell2.theta_i_cnt_angle
        # ### 反射光基準
        # theta1_v = Ell1.theta_i_cnt_o_angle
        # theta1_h = Ell2.theta_i_cnt_o_angle

        # omega_V = ((np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v)) + theta5_v1 + theta5_v2)/2
        # omega_V = Ell1.omega_default
        # omega_V = Ell1.omega_cnt_m_wid ### ミラー中心光線基準　昔これ基準で面内回転調整量大きかった
        omega_V = Ell1.omega_cnt_o_angle ### 反射光中心基準
        # omega_V = + 8e-4
        # omega_V = 0.
        NA_h = np.sin(Ell1.na_o/2)
        NA_v = np.sin(Ell2.na_o/2)
        if option == 'ray':

            print('Ell1 diverge angle', Ell1.theta_i1-Ell1.theta_i2)
            print('Ell1 mirror length', Ell1.mirr_length)
            print('Ell1 mirror angle', [(Ell1.theta_i1+Ell1.theta_o1)/2, (Ell1.theta_i2+Ell1.theta_o2)/2])
            print('Ell2 diverge angle', Ell2.theta_i1-Ell2.theta_i2)
            print('Ell2 mirror length', Ell2.mirr_length)
            print('Ell2 mirror angle', [(Ell2.theta_i1+Ell2.theta_o1)/2, (Ell2.theta_i2+Ell2.theta_o2)/2])
            print('===========================')
            print('Ell1 aperture',Ell1.mirr_length*Ell1.theta_centre)
            print('Ell2 aperture',Ell2.mirr_length*Ell2.theta_centre)
            print('Area aperture',Ell1.mirr_length*Ell1.theta_centre*Ell2.mirr_length*Ell2.theta_centre)
            print('Focus distance',Ell1.f-Ell2.f)
            conditions_file_path = os.path.join(directory_name, 'kb_design.txt')

            # テキストファイルに変数の値や計算条件を書き込む
            with open(conditions_file_path, 'w') as file:
                file.write(f'Ell1 diverge angle: {Ell1.theta_i1-Ell1.theta_i2}\n')
                file.write(f'Ell1 mirror length: {Ell1.mirr_length}\n')
                file.write(f'Ell1 mirror angle: {(Ell1.theta_i1+Ell1.theta_o1)/2, (Ell1.theta_i2+Ell1.theta_o2)/2}\n')
                file.write(f'Ell2 diverge angle: {Ell2.theta_i1-Ell2.theta_i2}\n')
                file.write(f'Ell2 mirror length: {Ell2.mirr_length}\n')
                file.write(f'Ell2 mirror angle: {(Ell2.theta_i1+Ell2.theta_o1)/2, (Ell2.theta_i2+Ell2.theta_o2)/2}\n')
                file.write('===========================\n')
                file.write(f'Ell1 aperture: {Ell1.mirr_length*Ell1.theta_centre}\n')
                file.write(f'Ell2 aperture: {Ell2.mirr_length*Ell2.theta_centre}\n')
                file.write(f'Area aperture: {Ell1.mirr_length*Ell1.theta_centre*Ell2.mirr_length*Ell2.theta_centre}\n')
                file.write(f'Focus distance: {Ell1.f-Ell2.f}\n')
            conditions_file_path = os.path.join(directory_name, 'kb_design_details.json')

            with open(conditions_file_path, 'w') as file:
                json.dump(
                    {
                        'Ell1': vars(Ell1),
                        'Ell2': vars(Ell2)
                    },
                    file,
                    indent=4,  # 見やすく整形
                    ensure_ascii=False
                )

            print('NA',NA_h)
            print('NA',NA_v)
            print('theta_g_V1',(Ell1.theta_i1+Ell1.theta_o1)/2)
            print('theta_g_V2',(Ell1.theta_i2+Ell1.theta_o2)/2)
            print('theta_g_H1',(Ell2.theta_i1+Ell2.theta_o1)/2)
            print('theta_g_H2',(Ell2.theta_i2+Ell2.theta_o2)/2)
            print('mirrLenV',Ell1.x_2)
            print('mirrLenH',Ell2.x_2)
            div_V = Ell1.theta_i1-Ell1.theta_i2
            div_H = Ell2.theta_i1-Ell2.theta_i2
            print('div_V',div_V)
            print('div_H',div_H)
            print('Ell1.x_3',Ell1.x_3)
            gap = Ell2.x_1 - Ell1.x_1 - Ell1.x_2
            print('Gap',gap)
            print('WD',Ell2.l_o2)
            P_V1 = 1/Ell1.x_1 + 1/(Ell1.x_2+Ell1.x_3)
            P_V2 = 1/(Ell1.x_1+Ell1.x_2) + 1/Ell1.x_3
            P_H1 = 1/Ell2.x_1 + 1/(Ell2.x_2+Ell2.x_3)
            P_H2 = 1/(Ell2.x_1+Ell2.x_2) + 1/Ell2.x_3
            P_VC = 1/Ell1.x_cent_o_angle + 1/(2*Ell1.f - Ell1.x_cent_o_angle)
            P_HC = 1/Ell2.x_cent_o_angle + 1/(2*Ell2.f - Ell2.x_cent_o_angle)
            print('P_V1',P_V1)
            print('P_V2',P_V2)
            print('P_H1',P_H1)
            print('P_H2',P_H2)
            print('P_VC',P_VC)
            print('P_HC',P_HC)
            def delta_so(P0,P,s_i):
                return 1/(P-1/s_i) - 1/(P0-1/s_i)

            delta_so_H1 = delta_so(P_VC, P_VC*np.cos(div_H/2)**2, Ell1.x_cent_o_angle)
            delta_so_V2 = delta_so(P_HC, P_HC*np.cos(Ell1.na_o/2)**2, Ell2.x_cent_o_angle)

            div_H = div_H
            div_V = div_V
            P_VC1_dash = P_VC*np.cos(div_H/2)**2
            P_VC2_dash = -P_HC*np.sin(Ell1.na_o/2)**2
            P_HC1_dash = -P_VC*np.sin(div_H/2)**2
            P_HC2_dash = P_HC*np.cos(Ell1.na_o/2)**2
            print('P_VC1_dash',P_VC1_dash)
            print('P_VC2_dash',P_VC2_dash)
            print('P_HC1_dash',P_HC1_dash)
            print('P_HC2_dash',P_HC2_dash)
            # delta_so_V1 = 1/(P_VC*np.sin(div_H/2)**2-1/Ell1.x_cent_o_angle) + Ell1.x_cent_o_angle
            print('delta_so_H1',delta_so_H1)
            print('delta_so_V2',delta_so_V2)
            gap_C = Ell2.x_cent_o_angle - Ell1.x_cent_o_angle

            so_1_H = 1/(P_HC1_dash-1/Ell1.x_cent_o_angle) ### Ell1.x_cent_o_angle = Ell2.x_cent_o_angle - gap
            print('so_1_H',so_1_H)
            si_2_H = gap_C - so_1_H
            so_fin_H = 1/(P_HC2_dash - 1/si_2_H)

            so_1_V = 1/(P_VC1_dash-1/Ell1.x_cent_o_angle)
            print('so_1_V',so_1_V)
            si_2_V = gap_C - so_1_V
            so_fin_V = gap_C + 1/(P_VC2_dash - 1/si_2_V)

            def calc_so_V(si_1_V, gap_C, P_VC1_dash, P_VC2_dash):
                so_1_V = 1/(P_VC1_dash-1/si_1_V)
                si_2_V = gap_C - so_1_V
                so_fin_V = gap_C + 1/(P_VC2_dash - 1/si_2_V)
                return so_fin_V

            div_i_H_array = np.linspace(-div_H/2, div_H/2, 3)
            # div_o_V_array = np.linspace(-div_V/2, div_V/2, 10)
            div_o_V_array = np.linspace(-Ell1.na_o/2, Ell1.na_o/2, 3)
            div_i_H_matrix, div_o_V_matrix = np.meshgrid(div_i_H_array, div_o_V_array)

            P_VC_array = np.array([P_V1,P_VC,P_V2])
            P_HC_array = np.array([P_H1,P_HC,P_H2])
            P_HC_matrix, P_VC_matrix = np.meshgrid(P_HC_array, P_VC_array)

            P_VC1_dash_matrix = P_VC_matrix*np.cos(div_i_H_matrix)**2
            P_VC2_dash_matrix = -P_HC_matrix*np.sin(div_o_V_matrix)**2
            so_fin_V_matrix = calc_so_V(Ell1.x_cent_o_angle, gap_C, P_VC1_dash_matrix, P_VC2_dash_matrix)

            def calc_so_H(si_1_H, gap_C, P_HC1_dash, P_HC2_dash):
                so_1_H = 1/(P_HC1_dash-1/si_1_H)
                si_2_H = gap_C - so_1_H
                so_fin_H = 1/(P_HC2_dash - 1/si_2_H)
                return so_fin_H

            P_HC1_dash_matrix = -P_VC_matrix*np.sin(div_i_H_matrix)**2
            P_HC2_dash_matrix = P_HC_matrix*np.cos(div_o_V_matrix)**2
            so_fin_H_matrix = calc_so_H(Ell1.x_cent_o_angle, gap_C, P_HC1_dash_matrix, P_HC2_dash_matrix)

            print('si_2_H',si_2_H)
            print('si_2_V',si_2_V)
            print('gap_C',gap_C)
            print('Ell1.x_cent_o_angle',Ell1.x_cent_o_angle)
            print('Ell2.x_cent_o_angle',Ell2.x_cent_o_angle)
            print('2*Ell1.f - Ell1.x_cent_o_angle',2*Ell1.f - Ell1.x_cent_o_angle)
            print('2*Ell2.f - Ell2.x_cent_o_angle',2*Ell2.f - Ell2.x_cent_o_angle)
            print('so_fin_H',so_fin_H)
            delta_fin_H = so_fin_H-(2*Ell2.f - Ell2.x_cent_o_angle)
            print('delta_fin_H',delta_fin_H)
            print('so_fin_V',so_fin_V)
            delta_fin_V = so_fin_V-(2*Ell1.f - Ell1.x_cent_o_angle)
            print('delta_fin_V',delta_fin_V)

    else:
        if option_HighNA == True:
            # l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v = [np.float64(146.), np.float64(0.8),  np.float64(0.25), np.float64(0.5), np.float64(0.1), np.float64(0.13), np.float64(0.22)]
            # a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap] = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v)
            l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v = [np.float64(146.), np.float64(0.8),  np.float64(0.242), np.float64(0.5), np.float64(0.1), np.float64(0.128), np.float64(0.22)]

            # l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v = [np.float64(146.), np.float64(0.8),  np.float64(0.242), np.float64(0.5), np.float64(0.1), np.float64(0.128), np.float64(0.22)]
            a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap] = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v)

            # from scipy.optimize import fsolve
            # # `fsolve` を使用して `inc_h` を調整
            # # def calculate_NA_h(inc_h, target_NA_h, l1h, l2h, mlen_h, wd_v, inc_v, mlen_v):
            # #     # `KB_define` を呼び出して `NA_h` を計算
            # #     a_h, b_h, a_v, b_v, l1v, l2v, params = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v)
            # #     NA_h = params[7]  # `params` の 7 番目の要素が `NA_h`
            # #     return NA_h - target_NA_h  # 目標値との差を返す
            # # l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v = [np.float64(146.), np.float64(0.8),  np.float64(0.242), np.float64(0.5*1.5), np.float64(0.1), np.float64(0.128), np.float64(0.22)]
            # # initial_inc_h = inc_h.copy()
            # # target_NA_h = NA_h.copy()  # 目標とする NA_h の値を設定
            # # a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap] = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v)
            # # optimized_inc_h = fsolve(calculate_NA_h, initial_inc_h, args=(target_NA_h, l1h, l2h, mlen_h, wd_v, inc_v, mlen_v))[0]

            # # print(f"目標 NA_h: {target_NA_h}")
            # # print(f"調整後の inc_h: {optimized_inc_h}")
            # # inc_h = optimized_inc_h.copy()

            # def calculate_NA_v(inc_v, target_NA_v, l1h, l2h, mlen_h, wd_v, inc_h, mlen_v):
            #     # `KB_define` を呼び出して `NA_h` を計算
            #     a_h, b_h, a_v, b_v, l1v, l2v, params = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v)
            #     NA_v = params[15]  # `params` の 15 番目の要素が `NA_v`
            #     return NA_v - target_NA_v  # 目標値との差を返す
            # l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v = [np.float64(146.), np.float64(0.8),  np.float64(0.242), np.float64(0.5), np.float64(0.1), np.float64(0.128), np.float64(0.22/2)]
            # initial_inc_v = inc_v.copy()
            # target_NA_v = NA_v.copy()  # 目標とする NA_h の値を設定
            # a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap] = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v)
            # optimized_inc_v = fsolve(calculate_NA_v, initial_inc_v, args=(target_NA_v, l1h, l2h, mlen_h, wd_v, inc_h, mlen_v))[0]

            # print(f"目標 NA_v: {target_NA_v}")
            # print(f"調整後の inc_v: {optimized_inc_v}")
            # inc_v = optimized_inc_v.copy()
            # a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap] = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v)


        else:
            # gapf = -0.230
            l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v = [np.float64(146.), np.float64(0.8),  np.float64(0.029 * LowNAratio), np.float64(0.5), np.float64(0.1), np.float64(0.016 * LowNAratio), np.float64(0.22)]
            a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap] = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v)
            # gapf = 0.
            # l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v = [np.float64(146.), np.float64(0.8),  np.float64(0.029 * LowNAratio), np.float64(0.5), np.float64(0.1), np.float64(0.016 * LowNAratio), np.float64(0.22)]
            # a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap] = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v, gapf=gapf)

            # NA_h_0 = NA_h.copy()
            # NA_v_0 = NA_v.copy()
            # ratio_wid_h = 0.6
            # ratio_wid_v = 0.6
            # l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v = [np.float64(146.), np.float64(0.8),  np.float64(0.029 * LowNAratio), np.float64(0.5*ratio_wid_h), np.float64(0.1), np.float64(0.016 * LowNAratio), np.float64(0.22 * ratio_wid_v)]
            # a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap] = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v, gapf=gapf)
            # ratio_inc_h = NA_h_0 / NA_h
            # ratio_inc_v = NA_v_0 / NA_v
            #
            # l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v = [np.float64(146.), np.float64(0.8),  np.float64(0.029 * LowNAratio * ratio_inc_h), np.float64(0.5*ratio_wid_h), np.float64(0.1), np.float64(0.016 * LowNAratio * ratio_inc_v), np.float64(0.22 * ratio_wid_v)]
            # a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap] = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v,  gapf=gapf)
        a_hyp_v = a_h
        b_hyp_v = b_h
        a_hyp_h = a_v
        b_hyp_h = b_v

        org_hyp_v = np.sqrt(a_hyp_v**2 - b_hyp_v**2)
        org_hyp_h = np.sqrt(a_hyp_h**2 - b_hyp_h**2)

        y1_v = yh_s
        x1_v = xh_s
        y2_v = yh_e
        x2_v = xh_e

        y1_h = yv_s
        x1_h = xv_s
        y2_h = yv_e
        x2_h = xv_e
        theta1_v = sita1h
        theta1_h = sita1v

    if option == 'ray':
        def RoC(x,a,b):
            c = np.sqrt(a**2-b**2)
            x0 = x-c
            theta = np.arccos(x0/a)
            # print('theta',theta)
            # print(b*np.sin(theta))
            R = ((a*np.sin(theta))**2+(b*np.cos(theta))**2)**(3/2)/(a*b)
            print('R',R)
            # R = (a**4 * (1 - x0**2/a**2) + b**2 * x0**2)/(b * (1 - x0**2/a**2)**(3/2))
            # print('R',R)
            # R = 1/0.014e-3
            # print('rgen',a**2/b)
            return R
        def Focuslength(x,a,b,theta=0.):
            # # R = RoC(x,a,b)
            # # f = x * R* (1/(2*x - R/cos**2))
            #
            # c = np.sqrt(a**2-b**2)
            # y = b*np.sqrt(1 - (x-c)**2/a**2)
            # ix0 = x/np.sqrt(x**2+y**2)
            # iy0 = y/np.sqrt(x**2+y**2)
            # iz0 = ix0.copy() * np.tan(theta)
            # ix = ix0/np.sqrt(ix0**2 + iy0**2 + iz0**2)
            # iy = iy0/np.sqrt(ix0**2 + iy0**2 + iz0**2)
            # iz = iz0/np.sqrt(ix0**2 + iy0**2 + iz0**2)
            # # print('theta',theta)
            # # print('ix0',ix0)
            # # print('iy0',iy0)
            # # print('ix',ix)
            #
            # nx= b**2 * (x-c)/(a**2 * y)/np.sqrt(1 + (b**2 * (x-c)/(a**2 * y))**2)
            # ny= 1/np.sqrt(1 + (b**2 * (x-c)/(a**2 * y))**2)
            # rx = ix-2*(ix*nx+iy*ny)*nx
            # ry = iy-2*(ix*nx+iy*ny)*ny
            # f = x - rx/ry*y
            # fz = - iz/ry*y
            # print('fz',fz)

            f0 = 2*np.sqrt(a**2-b**2)
            s0 = x.copy()
            s0_ = f0 - s0
            f0 = s0*s0_/(s0 + s0_)
            f = s0/(s0*np.cos(theta)**2/f0 - 1.)

            return f
        print('x1_v',x1_v)
        print('x2_v',x2_v)
        print('a_hyp_v',a_hyp_v)
        print('b_hyp_v',b_hyp_v)
        angle_h = (np.arctan(y1_h / x1_h)-np.arctan(y2_h / x2_h))/2
        angle_v = (np.arctan(y1_v / x1_v)-np.arctan(y2_v / x2_v))/2
        print('angle_v',angle_v)
        print('angle_h',angle_h)
        print('design f  ',2*np.sqrt(a_hyp_v**2-b_hyp_v**2))
        f1_v_analy = Focuslength(x1_v,a_hyp_v,b_hyp_v)
        f2_v_analy = Focuslength(x2_v,a_hyp_v,b_hyp_v)
        fc_v_analy = Focuslength((x2_v+x1_v)/2,a_hyp_v,b_hyp_v)
        f1_v_analy_edge = Focuslength(x1_v,a_hyp_v,b_hyp_v,theta=angle_h)
        f2_v_analy_edge = Focuslength(x2_v,a_hyp_v,b_hyp_v,theta=angle_h)
        fc_v_analy_edge = Focuslength((x2_v+x1_v)/2,a_hyp_v,b_hyp_v,theta=angle_h)
        print('f1_v_analy',f1_v_analy)
        print('f2_v_analy',f2_v_analy)
        print('fc_v_analy',fc_v_analy)
        print('dif_f1_v_analy',f1_v_analy_edge-f1_v_analy)
        print('dif_f2_v_analy',f2_v_analy_edge-f2_v_analy)
        print('dif_fc_v_analy',fc_v_analy_edge-fc_v_analy)

        print('x1_h',x1_h)
        print('x2_h',x2_h)
        print('a_hyp_h',a_hyp_h)
        print('b_hyp_h',b_hyp_h)
        print('design f  ',2*np.sqrt(a_hyp_h**2-b_hyp_h**2))
        f1_h_analy = Focuslength(x1_h,a_hyp_h,b_hyp_h)
        f2_h_analy = Focuslength(x2_h,a_hyp_h,b_hyp_h)
        fc_h_analy = Focuslength((x2_h+x1_h)/2,a_hyp_h,b_hyp_h)
        f1_h_analy_edge = Focuslength(x1_h,a_hyp_h,b_hyp_h,theta=angle_v)
        f2_h_analy_edge = Focuslength(x2_h,a_hyp_h,b_hyp_h,theta=angle_v)
        fc_h_analy_edge = Focuslength((x2_h+x1_h)/2,a_hyp_h,b_hyp_h,theta=angle_v)
        print('f1_h_analy',f1_h_analy)
        print('f2_h_analy',f2_h_analy)
        print('fc_h_analy',fc_h_analy)
        print('dif_f1_h_analy',f1_h_analy_edge-f1_h_analy)
        print('dif_f2_h_analy',f2_h_analy_edge-f2_h_analy)
        print('dif_fc_h_analy',fc_h_analy_edge-fc_h_analy)

        delta_xc = 1/np.cos(angle_h)*(np.sqrt(a_hyp_v**2 - b_hyp_v**2) - np.sqrt(a_hyp_v**2 - (b_hyp_v*np.cos(angle_h))**2))
        print('delta_xc',delta_xc)
        # sys.exit()

    # astig_v = (org_hyp_v - org_hyp_h)/2
    # astig_v_ = (org_hyp_v - org_hyp_h)/2*np.linspace(0,4,10)
    n = 20
    # param_ = np.linspace(-1,1,n)
    std_v = np.full(n, np.nan)
    std_h = np.full(n, np.nan)
    # astig_v = astig_v_[j]
    # astig_v = 0.5*-0.16626315789473686
    # astig_v = 0.5
    # print(astig_v_)
    # Input parameters
    ray_num_H = 33
    ray_num_V = 33
    ray_num = 33
    if option == 'ray':
        ray_num_H = 33
        ray_num_V = 33
        # ray_num = 1
    if option == 'wave' or option == 'ray_wave':
        ray_num_H = wave_num_H
        ray_num_V = wave_num_V

    bool_draw = True
    bool_point_source = True
    bool_imaging = False
    option_axial = True
    option_alignment = True ###

    optin_axialrotation = True
    # option_self = True
    option_self = False
    optionLocalRotation = True
    optionLocalRotationonlyAll = False

    # Raytrace (X = x-ray direction)

    # V hyp mirror set (1st)
    axis_x = np.float64([1.,0.,0.])
    axis_y = np.float64([0.,1.,0.])
    axis_z = np.float64([0.,0.,1.])
    coeffs_hyp_v = np.zeros(10)
    coeffs_hyp_v[0] = 1 / a_hyp_v**2
    coeffs_hyp_v[2] = 1 / b_hyp_v**2
    coeffs_hyp_v[9] = -1.
    coeffs_hyp_v = shift_x(coeffs_hyp_v, org_hyp_v)
    coeffs_hyp_v_org = coeffs_hyp_v.copy()
    if option_axial:
        # coeffs_hyp_v = rotate_y(coeffs_hyp_v, theta1_v, [0, 0, 0])
        coeffs_hyp_v, rotation_matrix = rotate_general_axis(coeffs_hyp_v, axis_y, theta1_v, [0., 0., 0.])
        if option == 'ray':
            print('1st mirr rotate axis_y',theta1_v)
        if optionLocalRotation:
            axis_x, axis_y, axis_z = rotatematrix(rotation_matrix, axis_x, axis_y, axis_z)

    if optKBdesign:
        # print('KB design')
        bufray = np.zeros((3, 2))
        if option_axial:
            bufray[0, :] = 1.
            bufray[1, :] = np.tan(theta1_h)
            bufray[2, :] = np.tan(theta1_v)
        source = np.zeros((3, 2))
    else:
        if option_alignment and option_axial:
            bufray = np.zeros((3, 5))
            ### 4隅の光線
            theta_cntr_h = (np.arctan(y2_h / x2_h) + np.arctan(y1_h / x1_h))/2.
            theta_cntr_v = (np.arctan(y2_v / x2_v) + np.arctan(y1_v / x1_v))/2.
            def print_optical_design(a_ell_h,b_ell_h,org_ell_h,theta1_h):

                l4_h = ((org_ell_h)**2 - 2*org_ell_h*a_ell_h*np.cos(theta1_h) + a_ell_h**2)/(a_ell_h - org_ell_h*np.cos(theta1_h))
                # l4_v = ((org_ell_v)**2 - 2*org_ell_v*a_ell_v*np.cos(theta3_v) + a_ell_v**2)/(a_ell_v - org_ell_v*np.cos(theta3_v))

                l1_h = 2*a_ell_h - l4_h
                # l3_v = 2*a_ell_v - l2_v - l4_v

                theta5_h = np.arcsin((2*a_ell_h - l4_h)*np.sin(theta1_h)/l4_h)
                # theta5_v = np.arcsin((2*a_ell_v - l4_v)*np.sin(theta3_v)/l4_v)

                theta4_h = (theta5_h+theta1_h)/2.
                # # theta4_v = (theta5_v+theta3_v)/2.
                # print('theta1',theta1_h)
                # # print('l2_v',l2_v)
                # print('l1',l1_h)
                # # print('l1_v',l1_v)
                # print('l4',l4_h)
                # # print('l4_v',l4_v)
                # print('theta4 incidence ell',theta4_h)
                # # print('theta4_v incidence ell',theta4_v)
                theta4_h = np.arcsin(2*org_ell_h*np.sin(theta1_h)/l4_h)/2
                # # theta4_v = np.arcsin(2*org_ell_v*np.sin(theta3_v)/l4_v)/2
                # print('theta4 incidence ell',theta4_h)
                # # print('theta4_v incidence ell',theta4_v)
                # print('theta5 focal',theta5_h)
                # # print('theta5_v focal',theta5_v)
                # print('width1',l1_h*np.cos(theta1_h))
                # print('width3',l4_h*np.cos(theta5_h))
                # print('')
                return l1_h*np.cos(theta1_h),l4_h*np.cos(theta5_h),theta5_h, l1_h, l4_h,theta4_h
            width1_h,width3_h,theta5_h, l1_h, l4_h,theta4_h = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,theta1_h)
            width1_h1,width3_h1,theta5_h1, l1_h1, l4_h1,theta4_h1 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,np.arctan(y1_h / x1_h))
            width1_h2,width3_h2,theta5_h2, l1_h2, l4_h2,theta4_h2 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,np.arctan(y2_h / x2_h))
            width1_v,width3_v,theta5_v, l1_v, l4_v,theta4_v = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,theta1_v)
            width1_v1,width3_v1,theta5_v1, l1_v1, l4_v1,theta4_v1 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,np.arctan(y1_v / x1_v))
            width1_v2,width3_v2,theta5_v2, l1_v2, l4_v2,theta4_v2 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,np.arctan(y2_v / x2_v))
            omega_V = ((np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v)) + theta5_v1 + theta5_v2)/2
            if option == 'ray':
                def print_optical_design(a_ell_h,b_ell_h,org_ell_h,theta1_h):

                    l4_h = ((org_ell_h)**2 - 2*org_ell_h*a_ell_h*np.cos(theta1_h) + a_ell_h**2)/(a_ell_h - org_ell_h*np.cos(theta1_h))
                    # l4_v = ((org_ell_v)**2 - 2*org_ell_v*a_ell_v*np.cos(theta3_v) + a_ell_v**2)/(a_ell_v - org_ell_v*np.cos(theta3_v))

                    l1_h = 2*a_ell_h - l4_h
                    # l3_v = 2*a_ell_v - l2_v - l4_v

                    theta5_h = np.arcsin((2*a_ell_h - l4_h)*np.sin(theta1_h)/l4_h)
                    # theta5_v = np.arcsin((2*a_ell_v - l4_v)*np.sin(theta3_v)/l4_v)

                    theta4_h = (theta5_h+theta1_h)/2.
                    # theta4_v = (theta5_v+theta3_v)/2.
                    print('theta1',theta1_h)
                    # print('l2_v',l2_v)
                    print('l1',l1_h)
                    # print('l1_v',l1_v)
                    print('l4',l4_h)
                    # print('l4_v',l4_v)
                    print('theta4 incidence ell',theta4_h)
                    # print('theta4_v incidence ell',theta4_v)
                    theta4_h = np.arcsin(2*org_ell_h*np.sin(theta1_h)/l4_h)/2
                    # theta4_v = np.arcsin(2*org_ell_v*np.sin(theta3_v)/l4_v)/2
                    print('theta4 incidence ell',theta4_h)
                    # print('theta4_v incidence ell',theta4_v)
                    print('theta5 focal',theta5_h)
                    # print('theta5_v focal',theta5_v)
                    print('width1',l1_h*np.cos(theta1_h))
                    print('width3',l4_h*np.cos(theta5_h))
                    print('')
                    return l1_h*np.cos(theta1_h),l4_h*np.cos(theta5_h),theta5_h, l1_h, l4_h,theta4_h

                print('===== Horizontal center =====')
                width1_h,width3_h,theta5_h, l1_h, l4_h,theta4_h = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,theta1_h)
                print('===== Horizontal edge1 =====')
                width1_h1,width3_h1,theta5_h1, l1_h1, l4_h1,theta4_h1 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,np.arctan(y1_h / x1_h))
                print('===== Horizontal edge2 =====')
                width1_h2,width3_h2,theta5_h2, l1_h2, l4_h2,theta4_h2 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,np.arctan(y2_h / x2_h))

                print('===== Vertical center =====')
                width1_v,width3_v,theta5_v, l1_v, l4_v,theta4_v = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,theta1_v)
                print('===== Vertical edge1 =====')
                width1_v1,width3_v1,theta5_v1, l1_v1, l4_v1,theta4_v1 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,np.arctan(y1_v / x1_v))
                print('===== Vertical edge2 =====')
                width1_v2,width3_v2,theta5_v2, l1_v2, l4_v2,theta4_v2 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,np.arctan(y2_v / x2_v))

                print('===== ===== =====')
                omega_V = ((np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v)) + theta5_v1 + theta5_v2)/2
                print('omega_V',omega_V)
                print('===== ===== =====')
                print('na_h',np.sin((theta5_h1 - theta5_h2))/2.)
                print('na_v',np.sin((theta5_v1 - theta5_v2))/2.)
                print('div_h',(arctan(y2_h / x2_h) - arctan(y1_h / x1_h)))
                print('div_v',(arctan(y2_v / x2_v) - arctan(y1_v / x1_v)))
                print('1stmrr width',width1_v2 - width1_v1)
                print('2ndmrr width',width1_h2 - width1_h1)
                print('sorce to center 1stmrr',width1_v)
                print('center 1stmrr to focus',width3_v)
                print('sorce to center 2ndmrr',width1_h)
                print('center 2ndmrr to focus',width3_h)
                print('distance 1stmrr 2ndmrr',width1_h-width1_v)
                print('inc 1stmrr',[theta4_v1,theta4_v2])
                print('inc 2ndmrr',[theta4_h1,theta4_h2])
                print('WD',width3_h2)
                print('===== ===== =====')
                na0_h = (np.arctan(y2_h / x2_h) - np.arctan(y1_h / x1_h))/2.
                na0_v = (np.arctan(y2_v / x2_v) - np.arctan(y1_v / x1_v))/2.
                na_h = (theta5_h1 - theta5_h2)/2.
                na_v = (theta5_v1 - theta5_v2)/2.
                dif_l1_h = l1_h * (1/np.cos(na0_v) - 1)
                dif_l4_h = l4_h * (1/np.cos(na_v) - 1)
                dif_l1_v = l1_v * (1/np.cos(na0_h) - 1)
                dif_l4_v = l4_v * (1/np.cos(na_h) - 1)
                print('dif_l1_h',dif_l1_h)
                print('dif_l4_h',dif_l4_h)
                print('dif_l1_v',dif_l1_v)
                print('dif_l4_v',dif_l4_v)

            width1_v1,width3_v1,theta5_v1, l1_v1, l4_v1,_ = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,np.arctan(y1_v / x1_v))
            width1_v2,width3_v2,theta5_v2, l1_v2, l4_v2,_ = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,np.arctan(y2_v / x2_v))
            omega_V = ((np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v)) + theta5_v1 + theta5_v2)/2

            theta_source_1_h = np.arctan(y1_h / x1_h) - theta_cntr_h
            theta_source_2_h = np.arctan(y2_h / x2_h) - theta_cntr_h
            theta_source_1_v = np.arctan(y1_v / x1_v) - theta_cntr_v
            theta_source_2_v = np.arctan(y2_v / x2_v) - theta_cntr_v

            bufray[0, 0] = 1.
            bufray[1, 0] = np.tan(theta1_h)
            bufray[2, 0] = np.tan(theta1_v)

            bufray[0, 1] = 1.
            bufray[1, 1] = np.tan(theta_source_1_h)
            bufray[2, 1] = np.tan(theta_source_1_v)

            bufray[0, 2] = 1.
            bufray[1, 2] = np.tan(theta_source_2_h)
            bufray[2, 2] = np.tan(theta_source_1_v)

            bufray[0, 3] = 1.
            bufray[1, 3] = np.tan(theta_source_2_h)
            bufray[2, 3] = np.tan(theta_source_1_v)

            bufray[0, 4] = 1.
            bufray[1, 4] = np.tan(theta_source_2_h)
            bufray[2, 4] = np.tan(theta_source_2_v)

            source = np.zeros((3, 5))

        else:
            bufray = np.zeros((3, 2))
            if option == 'ray':
                theta_cntr_h = (np.arctan(y2_h / x2_h) + np.arctan(y1_h / x1_h))/2.
                theta_cntr_v = (np.arctan(y2_v / x2_v) + np.arctan(y1_v / x1_v))/2.
                def print_optical_design(a_ell_h,b_ell_h,org_ell_h,theta1_h):

                    l4_h = ((org_ell_h)**2 - 2*org_ell_h*a_ell_h*np.cos(theta1_h) + a_ell_h**2)/(a_ell_h - org_ell_h*np.cos(theta1_h))
                    # l4_v = ((org_ell_v)**2 - 2*org_ell_v*a_ell_v*np.cos(theta3_v) + a_ell_v**2)/(a_ell_v - org_ell_v*np.cos(theta3_v))

                    l1_h = 2*a_ell_h - l4_h
                    # l3_v = 2*a_ell_v - l2_v - l4_v

                    theta5_h = np.arcsin((2*a_ell_h - l4_h)*np.sin(theta1_h)/l4_h)
                    # theta5_v = np.arcsin((2*a_ell_v - l4_v)*np.sin(theta3_v)/l4_v)

                    theta4_h = (theta5_h+theta1_h)/2.
                    # theta4_v = (theta5_v+theta3_v)/2.
                    print('theta1',theta1_h)
                    # print('l2_v',l2_v)
                    print('l1',l1_h)
                    # print('l1_v',l1_v)
                    print('l4',l4_h)
                    # print('l4_v',l4_v)
                    print('theta4 incidence ell',theta4_h)
                    # print('theta4_v incidence ell',theta4_v)
                    theta4_h = np.arcsin(2*org_ell_h*np.sin(theta1_h)/l4_h)/2
                    # theta4_v = np.arcsin(2*org_ell_v*np.sin(theta3_v)/l4_v)/2
                    print('theta4 incidence ell',theta4_h)
                    # print('theta4_v incidence ell',theta4_v)
                    print('theta5 focal',theta5_h)
                    # print('theta5_v focal',theta5_v)
                    print('width1',l1_h*np.cos(theta1_h))
                    print('width3',l4_h*np.cos(theta5_h))
                    print('')
                    return l1_h*np.cos(theta1_h),l4_h*np.cos(theta5_h),theta5_h, l1_h, l4_h,theta4_h

                print('===== Horizontal center =====')
                width1_h,width3_h,theta5_h, l1_h, l4_h,theta4_h = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,theta1_h)
                print('===== Horizontal edge1 =====')
                width1_h1,width3_h1,theta5_h1, l1_h1, l4_h1,theta4_h1 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,np.arctan(y1_h / x1_h))
                print('===== Horizontal edge2 =====')
                width1_h2,width3_h2,theta5_h2, l1_h2, l4_h2,theta4_h2 = print_optical_design(a_hyp_h,b_hyp_h,org_hyp_h,np.arctan(y2_h / x2_h))

                print('===== Vertical center =====')
                width1_v,width3_v,theta5_v, l1_v, l4_v,theta4_v = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,theta1_v)
                print('===== Vertical edge1 =====')
                width1_v1,width3_v1,theta5_v1, l1_v1, l4_v1,theta4_v1 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,np.arctan(y1_v / x1_v))
                print('===== Vertical edge2 =====')
                width1_v2,width3_v2,theta5_v2, l1_v2, l4_v2,theta4_v2 = print_optical_design(a_hyp_v,b_hyp_v,org_hyp_v,np.arctan(y2_v / x2_v))

                print('===== ===== =====')
                omega_V = ((np.arctan(y1_v / x1_v) + np.arctan(y2_v / x2_v)) + theta5_v1 + theta5_v2)/2
                print('omega_V',omega_V)
                print('===== ===== =====')
                print('na_h',np.sin((theta5_h1 - theta5_h2))/2.)
                print('na_v',np.sin((theta5_v1 - theta5_v2))/2.)
                print('div_h',(arctan(y2_h / x2_h) - arctan(y1_h / x1_h)))
                print('div_v',(arctan(y2_v / x2_v) - arctan(y1_v / x1_v)))
                print('1stmrr width',width1_v2 - width1_v1)
                print('2ndmrr width',width1_h2 - width1_h1)
                print('sorce to center 1stmrr',width1_v)
                print('center 1stmrr to focus',width3_v)
                print('sorce to center 2ndmrr',width1_h)
                print('center 2ndmrr to focus',width3_h)
                print('distance 1stmrr 2ndmrr',width1_h-width1_v)
                print('inc 1stmrr',[theta4_v1,theta4_v2])
                print('inc 2ndmrr',[theta4_h1,theta4_h2])
                print('===== ===== =====')
                na0_h = np.abs((np.arctan(y2_h / x2_h) - np.arctan(y1_h / x1_h))/2.)
                na0_v = np.abs((np.arctan(y2_v / x2_v) - np.arctan(y1_v / x1_v))/2.)
                na_h = np.abs((theta5_h1 - theta5_h2)/2.)
                na_v = np.abs((theta5_v1 - theta5_v2)/2.)

                def printAnalyticalWaveerror(na0_h,theta1_v,theta4_v,l1_v):
                    ### analytical waveerror v
                    # print('theta1_v',theta1_v)
                    # print('l1_v * np.sin(theta1_v)',l1_v * np.sin(theta1_v))
                    # print('l4_v * np.sin(theta5_v)',l4_v * np.sin(theta5_v))
                    # print('error',l1_v * np.sin(theta1_v) - l4_v * np.sin(theta5_v))
                    alpha_h = np.arcsin(np.tan(na0_h)/np.sin(theta1_v))
                    # print('na0_h',na0_h)
                    # print('alpha_h',alpha_h)
                    # dwave_v = 2 * np.sin(theta4_v) * l1_v * np.sin(theta1_v) * (1 - np.cos(alpha_h))
                    dwave_v = 2 * np.sin(theta4_v) * l1_v * ( np.sqrt( np.sin(theta1_v) **2 + np.tan(na0_h)**2) - np.sin(theta1_v) )
                    # print('theta4_v',theta4_v)
                    # print('dwave_v',dwave_v)
                    return dwave_v

                # ### analytical waveerror h
                # # print('theta1_h',theta1_h)
                # # print('l1_h * np.sin(theta1_h)',l1_h * np.sin(theta1_h))
                # # print('l4_h * np.sin(theta5_h)',l4_h * np.sin(theta5_h))
                # # print('error',l1_h * np.sin(theta1_h) - l4_h * np.sin(theta5_h))
                # alpha_v = np.arcsin(np.tan(na_v)/np.sin(theta5_h))
                # # print('na0_v',na_v)
                # # print('alpha_v',alpha_v)
                # # dwave_h = 2 * np.sin(theta4_h) * l4_h * np.sin(theta5_h) * (1 - np.cos(alpha_v))
                # dwave_h = 2 * np.sin(theta4_h) * l4_h * ( np.sqrt( np.sin(theta5_h) **2 + np.tan(na_v)**2) - np.sin(theta5_h) )
                # # print('theta4_h',theta4_h)
                # print('dwave_h',dwave_h)

                print('1st mirror wave error center',printAnalyticalWaveerror(na0_h,theta1_v,theta4_v,l1_v))
                print('1st mirror wave error edge1',printAnalyticalWaveerror(na0_h,np.arctan(y1_v / x1_v),theta4_v1,l1_v1))
                print('1st mirror wave error edge2',printAnalyticalWaveerror(na0_h,np.arctan(y2_v / x2_v),theta4_v2,l1_v2))
                print('2nd mirror wave error center',printAnalyticalWaveerror(na_v,theta5_h,theta4_h,l4_h))
                print('2nd mirror wave error edge1',printAnalyticalWaveerror(na_v,theta5_h1,theta4_h1,l4_h1))
                print('2nd mirror wave error edge2',printAnalyticalWaveerror(na_v,theta5_h2,theta4_h2,l4_h2))

                print('===== ===== =====')
            if option_axial:
                bufray[0, :] = 1.
                bufray[1, :] = np.tan(theta1_h)
                bufray[2, :] = np.tan(theta1_v)
            source = np.zeros((3, 2))

    bufray = normalize_vector(bufray)

    center_hyp_v = mirr_ray_intersection(coeffs_hyp_v, bufray, source)
    if not np.isreal(center_hyp_v).all():
        return np.inf
    bufreflect1 = reflect_ray(bufray, norm_vector(coeffs_hyp_v, center_hyp_v))
    bufreflangle1_y = np.arctan(np.mean(bufreflect1[2, 1:]) / np.mean(bufreflect1[0, 1:]))
    if option == 'ray':
        print('coeffs_hyp_v',coeffs_hyp_v)
        print('center_hyp_v',center_hyp_v)
        print('bufray',bufray)
        print('np.mean(bufreflect1[2, 1:])',np.mean(bufreflect1[2, 1:]))
        print('np.mean(bufreflect1[0, 1:])',np.mean(bufreflect1[0, 1:]))
        print('angle_y 1st to 2nd',bufreflangle1_y)
    # print(center_hyp_v)
    # print(bufreflect1)
    # print(np.arctan(bufreflect1[2, 0] / bufreflect1[0, 0]))
    # print('1st')

    # H hyp mirror set (2nd)
    coeffs_hyp_h = np.zeros(10)
    coeffs_hyp_h[0] = 1 / a_hyp_h**2
    coeffs_hyp_h[1] = 1 / b_hyp_h**2
    coeffs_hyp_h[9] = -1.
    # coeffs_hyp_h = shift_x(coeffs_hyp_h, org_hyp_h + astigH)
    org_hyp_h1 = org_hyp_h + astigH
    coeffs_hyp_h = shift_x(coeffs_hyp_h, org_hyp_h1)
    coeffs_hyp_h_org = coeffs_hyp_h.copy()
    axis_x_2nd = np.float64([1.,0.,0.])
    axis_y_2nd = np.float64([0.,1.,0.])
    axis_z_2nd = np.float64([0.,0.,1.])
    axis_x_global = np.float64([1.,0.,0.])
    axis_y_global = np.float64([0.,1.,0.])
    axis_z_global = np.float64([0.,0.,1.])
    if option_axial:
        coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis_z_global, -theta1_h, [0, 0, 0])
        if optionLocalRotation:
            axis_x_2nd, axis_y_2nd, axis_z_2nd = rotatematrix(rotation_matrix, axis_x_2nd, axis_y_2nd, axis_z_2nd)

    mean_bufreflect1 = np.mean(bufreflect1[:, 1:],axis = 1)

    center_hyp_h = mirr_ray_intersection(coeffs_hyp_h, bufreflect1, center_hyp_v)
    if not np.isreal(center_hyp_h).all():
        return np.inf
    if option_alignment:
        if not optin_axialrotation:
            coeffs_hyp_h = rotate_y(coeffs_hyp_h, -bufreflangle1_y, np.mean(center_hyp_v[:, 1:],axis=1))
        if optin_axialrotation:
            # coeffs_hyp_h = rotate_y(coeffs_hyp_h, -roty_local2, np.mean(center_hyp_h[:, 1:],axis=1))
            # center_hyp_h = mirr_ray_intersection(coeffs_hyp_h, bufreflect1, center_hyp_v)
            if optionLocalRotation:
                if optionLocalRotationonlyAll:
                    coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis_y_2nd, omega_V, np.mean(center_hyp_h[:, 1:],axis=1))
                    axis_x_2nd, axis_y_2nd, axis_z_2nd = rotatematrix(rotation_matrix, axis_x_2nd, axis_y_2nd, axis_z_2nd)
                else:
                    coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis_y_global, omega_V, np.mean(center_hyp_h[:, 1:],axis=1))
                    axis_x_2nd, axis_y_2nd, axis_z_2nd = rotatematrix(rotation_matrix, axis_x_2nd, axis_y_2nd, axis_z_2nd)
            else:
                coeffs_hyp_h = rotate_y(coeffs_hyp_h, omega_V, np.mean(center_hyp_h[:, 1:],axis=1))
            center_hyp_h = mirr_ray_intersection(coeffs_hyp_h, bufreflect1, center_hyp_v)

    bufreflect2 = reflect_ray(bufreflect1, norm_vector(coeffs_hyp_h, center_hyp_h))
    bufreflangle2_z = np.arctan(np.mean(bufreflect2[1, 1:]) / np.mean(bufreflect2[0, 1:]))
    bufreflangle2_y = np.arctan(np.mean(bufreflect2[2, 1:]) / np.mean(bufreflect2[0, 1:]))

    if option == 'ray':
        print('angle_y 2nd to 3rd',bufreflangle2_y)
        print('angle_z 2nd to 3rd',bufreflangle2_z)
    # print(bufreflect2)
    # print(bufreflangle2_z)

    # print('2nd')
    # print(center_hyp_h)

    s2f_H = 2 * org_hyp_h
    s2f_V = 2 * org_hyp_v
    # print(s2f_H)
    # print(s2f_V)
    s2f_middle = (s2f_H + s2f_V) / 2
    coeffs_det = np.zeros(10)
    coeffs_det[6] = 1.
    coeffs_det[9] = -(s2f_middle + defocus)

    # if option_axial:
    #     coeffs_det = rotate_y(coeffs_det, theta1_v, [0, 0, 0])
    #     coeffs_det = rotate_y(coeffs_det, bufreflangle1_y, center_hyp_v[:, 0])
    #     coeffs_det = rotate_z(coeffs_det, -theta1_h, [0, 0, 0])
    detcenter = plane_ray_intersection(coeffs_det, bufreflect2, center_hyp_h)
    detcenter = detcenter[:, 0]

    # # if pitch_h != 0:
    # #     coeffs_h = rotate_z(coeffs_h, pitch_h, hcenter)
    # # if roll_h != 0:
    # #     coeffs_h = rotate_x(coeffs_h, roll_h, hcenter)
    # # if yaw_h != 0:
    # #     coeffs_h = rotate_y(coeffs_h, yaw_h, hcenter)
    # if pitch_ell_v != 0:
    #     coeffs_ell_v = rotate_y(coeffs_ell_v, pitch_ell_v, center_ell_v[:, 0])
    # if roll_ell_v != 0:
    #     coeffs_ell_v = rotate_x(coeffs_ell_v, roll_ell_v, center_ell_v[:, 0])
    # if yaw_ell_v != 0:
    #     coeffs_ell_v = rotate_z(coeffs_ell_v, yaw_ell_v, center_ell_v[:, 0])

    # if pitch_ell_h != 0:
    #     coeffs_ell_h = rotate_y(coeffs_ell_h, pitch_ell_h, center_ell_h[:, 0])
    # if roll_ell_h != 0:
    #     coeffs_ell_h = rotate_x(coeffs_ell_h, roll_ell_h, center_ell_h[:, 0])
    # if yaw_ell_h != 0:
    #     coeffs_ell_h = rotate_z(coeffs_ell_h, yaw_ell_h, center_ell_h[:, 0])

    if pitch_hyp_v != 0: # in-plane rotation
        coeffs_hyp_v = rotate_y(coeffs_hyp_v, pitch_hyp_v, center_hyp_v[:, 0])
    if roll_hyp_v != 0:
        coeffs_hyp_v = rotate_x(coeffs_hyp_v, roll_hyp_v, center_hyp_v[:, 0])
    if yaw_hyp_v != 0:
        coeffs_hyp_v = rotate_z(coeffs_hyp_v, yaw_hyp_v, center_hyp_v[:, 0])


    if pitch_hyp_h != 0:
        # coeffs_hyp_h = rotate_y(coeffs_hyp_h, pitch_hyp_h, center_hyp_h[:, 0])
        coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis_y_2nd, pitch_hyp_h, np.mean(center_hyp_h[:, 1:],axis=1))
        # if optionLocalRotation:
        #     axis_x_2nd, axis_y_2nd, axis_z_2nd = rotatematrix(rotation_matrix, axis_x_2nd, axis_y_2nd, axis_z_2nd)
    if yaw_hyp_h != 0: # in-plane rotation
        # coeffs_hyp_h = rotate_z(coeffs_hyp_h, yaw_hyp_h, center_hyp_h[:, 0])
        coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis_z_2nd, yaw_hyp_h, np.mean(center_hyp_h[:, 1:],axis=1))
        # if optionLocalRotation:
        #     axis_x_2nd, axis_y_2nd, axis_z_2nd = rotatematrix(rotation_matrix, axis_x_2nd, axis_y_2nd, axis_z_2nd)
    if roll_hyp_h != 0:
        # coeffs_hyp_h = rotate_x(coeffs_hyp_h, roll_hyp_h, center_hyp_h[:, 0])
        coeffs_hyp_h, rotation_matrix = rotate_general_axis(coeffs_hyp_h, axis_x_2nd, roll_hyp_h, np.mean(center_hyp_h[:, 1:],axis=1))
        # if optionLocalRotation:
        #     axis_x_2nd, axis_y_2nd, axis_z_2nd = rotatematrix(rotation_matrix, axis_x_2nd, axis_y_2nd, axis_z_2nd)

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

    if bool_point_source:
        source = np.zeros((3, ray_num_H * ray_num_V))
        if option_axial:
            rand_p0h = np.linspace(np.arctan(y1_h / x1_h), np.arctan(y2_h / x2_h), ray_num_H)
            rand_p0v = np.linspace(np.arctan(y1_v / x1_v), np.arctan(y2_v / x2_v), ray_num_V)
            rand_p0h = rand_p0h - np.mean(rand_p0h)
            rand_p0v = rand_p0v - np.mean(rand_p0v)
            rand_p0h = rand_p0h*na_ratio_h
            rand_p0v = rand_p0v*na_ratio_v
        if not option_axial:
            rand_p0h = np.linspace(np.arctan(y1_h / x1_h), np.arctan(y2_h / x2_h), ray_num_H)
            rand_p0v = np.linspace(np.arctan(y1_v / x1_v), np.arctan(y2_v / x2_v), ray_num_V)
        if option_2mirror ==False:
            if ray_num_H==1:
                rand_p0h = np.array([0.])
            else:
                rand_p0h = np.linspace(-1e-9, 1e-9, ray_num_H)


        phai0 = np.zeros((3, ray_num_H * ray_num_V))
        for i in range(ray_num_V):
            rand_p0v_here = rand_p0v[i]
            phai0[1, ray_num_H * i:ray_num_H * (i + 1)] = np.tan(rand_p0h)
            phai0[2, ray_num_H * i:ray_num_H * (i + 1)] = np.tan(rand_p0v_here)
            phai0[0, ray_num_H * i:ray_num_H * (i + 1)] = 1.

        phai0 = normalize_vector(phai0)

        vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
        reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))

        dist0to1 = np.linalg.norm(vmirr_hyp - source, axis=0)

        if option_2mirror ==True:
            hmirr_hyp = mirr_ray_intersection(coeffs_hyp_h, reflect1, vmirr_hyp)
            reflect2 = reflect_ray(reflect1, norm_vector(coeffs_hyp_h, hmirr_hyp))
        else:
            hmirr_hyp = vmirr_hyp
            reflect2 = reflect1
            s2f_middle = s2f_V

        dist1to2 = np.linalg.norm(hmirr_hyp - vmirr_hyp, axis=0)

        if option == 'sep_direct':
            defocus = find_defocus(reflect2, hmirr_hyp, s2f_middle,defocus,ray_num)

        coeffs_det = np.zeros(10)
        coeffs_det[6] = 1
        coeffs_det[9] = -(s2f_middle + defocus)
        detcenter = plane_ray_intersection(coeffs_det, reflect2, hmirr_hyp)

        angle = reflect2.copy()

        if option == 'sep' or option == 'wave' or option == 'ray':
            vmirr_hyp_spltavg = vmirr_hyp.copy()
            # 範囲内の値を間引く
            original_array = list(range(ray_num_H*ray_num_V))
            thinned_array_v_y = original_array[round((ray_num_H-1)/2)::ray_num_H]
            # 範囲内の値を間引く
            start = round(ray_num_H*(ray_num_V-1)/2)
            end = round(ray_num_H*(ray_num_V+1)/2)
            thinned_array_h_y = crop(start, end, 1)
            def reset_p0(angle,rand_p0v,rand_p0h,thinned_array_v_y,thinned_array_h_y,ray_num_V,ray_num_H):
                angle_h = np.arctan(angle[1, :]/angle[0, :])
                angle_v = np.arctan(angle[2, :]/angle[0, :])

                angle_v_sep_y = angle_v[thinned_array_v_y]
                angle_h_sep_y = angle_h[thinned_array_h_y]

                output_equal_v = np.linspace(angle_v_sep_y[0],angle_v_sep_y[-1],len(angle_v_sep_y))
                output_equal_h = np.linspace(angle_h_sep_y[0],angle_h_sep_y[-1],len(angle_h_sep_y))

                interp_func_v = interp1d(angle_v_sep_y, rand_p0v, kind='linear')
                interp_func_h = interp1d(angle_h_sep_y, rand_p0h, kind='linear')

                rand_p0v_new = interp_func_v(output_equal_v)
                rand_p0h_new = interp_func_h(output_equal_h)

                phai0 = np.zeros((3, ray_num_H * ray_num_V))
                for i in range(ray_num_V):
                    rand_p0v_here = rand_p0v_new[i]
                    phai0[1, ray_num_H * i:ray_num_H * (i + 1)] = np.tan(rand_p0h_new)
                    phai0[2, ray_num_H * i:ray_num_H * (i + 1)] = np.tan(rand_p0v_here)
                    phai0[0, ray_num_H * i:ray_num_H * (i + 1)] = 1.

                phai0 = normalize_vector(phai0)
                return phai0

            ### 焦点面への入射角を均等に
            phai0 = reset_p0(angle,rand_p0v,rand_p0h,thinned_array_v_y,thinned_array_h_y,ray_num_V,ray_num_H)

            vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
            reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))
            if option_2mirror ==True:
                hmirr_hyp = mirr_ray_intersection(coeffs_hyp_h, reflect1, vmirr_hyp)
                reflect2 = reflect_ray(reflect1, norm_vector(coeffs_hyp_h, hmirr_hyp))
            else:
                hmirr_hyp = vmirr_hyp
                reflect2 = reflect1

            if option == 'sep_direct':
                defocus = find_defocus(reflect2, hmirr_hyp, s2f_middle,defocus,ray_num)

            coeffs_det = np.zeros(10)
            coeffs_det[6] = 1
            coeffs_det[9] = -(s2f_middle + defocus)
            detcenter = plane_ray_intersection(coeffs_det, reflect2, hmirr_hyp)

            if option == 'ray':
                    from scipy.spatial import cKDTree
                    def mindist(A,B):
                        tree = cKDTree(B.T)
                        dist, idx = tree.query(A.T, k=1)  # Aの各点からBへの最近点距離
                        min_dist = np.min(dist)
                        return min_dist
                    print('======================')
                    print('workX srs 1st',np.min(vmirr_hyp[0,:]) - np.max(source[0,:]))
                    print('workX 1st 2nd',np.min(hmirr_hyp[0,:]) - np.max(vmirr_hyp[0,:]))
                    print('workX 2nd fcs',np.min(detcenter[0,:]) - np.max(hmirr_hyp[0,:]))
                    print('======================')
                    print('workAbs srs 1st',mindist(source,vmirr_hyp))
                    print('workAbs 1st 2nd',mindist(vmirr_hyp,hmirr_hyp))
                    print('workAbs 2nd fcs',mindist(hmirr_hyp,detcenter))

                    print('1st W upper',np.linalg.norm(vmirr_hyp[:,0] - vmirr_hyp[:,ray_num-1]))
                    print('1st W lower',np.linalg.norm(vmirr_hyp[:,-1] - vmirr_hyp[:,-ray_num]))
                    print('2nd W lower',np.linalg.norm(hmirr_hyp[:,0] - hmirr_hyp[:,-ray_num]))
                    print('2nd W upper',np.linalg.norm(hmirr_hyp[:,ray_num-1] - hmirr_hyp[:,-1]))

                    conditions_file_path = os.path.join(directory_name, 'kb_configuration.txt')

                    # テキストファイルに変数の値や計算条件を書き込む
                    with open(conditions_file_path, 'w') as file:
                        file.write('=== KB configuration ===\n')
                        file.write('workX srs 1st: {}\n'.format(np.min(vmirr_hyp[0,:]) - np.max(source[0,:])))
                        file.write('workX 1st 2nd: {}\n'.format(np.min(hmirr_hyp[0,:]) - np.max(vmirr_hyp[0,:])))
                        file.write('workX 2nd fcs: {}\n'.format(np.min(detcenter[0,:]) - np.max(hmirr_hyp[0,:])))
                        file.write('workAbs srs 1st: {}\n'.format(mindist(source,vmirr_hyp)))
                        file.write('workAbs 1st 2nd: {}\n'.format(mindist(vmirr_hyp,hmirr_hyp)))
                        file.write('workAbs 2nd fcs: {}\n'.format(mindist(hmirr_hyp,detcenter)))
                        file.write('1st W upper: {}\n'.format(np.linalg.norm(vmirr_hyp[:,0] - vmirr_hyp[:,ray_num-1])))
                        file.write('1st W lower: {}\n'.format(np.linalg.norm(vmirr_hyp[:,-1] - vmirr_hyp[:,-ray_num])))
                        file.write('2nd W lower: {}\n'.format(np.linalg.norm(hmirr_hyp[:,0] - hmirr_hyp[:,-ray_num])))
                        file.write('2nd W upper: {}\n'.format(np.linalg.norm(hmirr_hyp[:,ray_num-1] - hmirr_hyp[:,-1])))

                    fig,axs = plt.subplots(2,1,sharex=True)
                    axs[0].plot(vmirr_hyp[0,:],vmirr_hyp[1,:])
                    axs[0].plot(hmirr_hyp[0,:],hmirr_hyp[1,:])
                    axs[0].plot(detcenter[0,:],detcenter[1,:])
                    axs[0].set_ylabel('Horizontal')

                    axs[1].plot(vmirr_hyp[0,:],vmirr_hyp[2,:])
                    axs[1].plot(hmirr_hyp[0,:],hmirr_hyp[2,:])
                    axs[1].plot(detcenter[0,:],detcenter[2,:])
                    axs[1].set_ylabel('Vertical')
                    axs[0].axis('equal')
                    axs[1].axis('equal')
                    plt.savefig(os.path.join(directory_name,'raytrace_mirror_configuration.png'))
                    # plt.show()


            angle1to2 = reflect1.copy()
            angle = reflect2.copy()
            source_org = source.copy()

        if option == 'wave':
            print('diverg angle H',np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))
            print('diverg angle V',np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))
            # # 全データからランダムに10%だけを選択
            # sample_indices = np.random.choice(detcenter.shape[1], size=int(detcenter.shape[1]*0.001), replace=False)
            theta_y = -np.mean(np.arctan(angle[2, :]/angle[0, :]))
            theta_z = np.mean(np.arctan(angle[1, :]/angle[0, :]))
            source = np.zeros((3,1))
            if option_rotate==True:
                reflect2_rotated_org = rotate_vectors(reflect2, -theta_y, -theta_z)
                focus_apprx = np.mean(detcenter,axis=1)
                hmirr_hyp_points_rotated_org = rotate_points(hmirr_hyp, focus_apprx, -theta_y, -theta_z)
                source_rotated = rotate_points(source, focus_apprx, -theta_y, -theta_z)
                if option_avrgsplt: ### 均等分割
                    vmirr_hyp_points_rotated = rotate_points(vmirr_hyp_spltavg, focus_apprx, -theta_y, -theta_z)

                    phai0 = reset_p0(angle1to2,rand_p0v,rand_p0h,thinned_array_v_y,thinned_array_h_y,ray_num_V,ray_num_H)
                    vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source_org)
                    reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))
                    hmirr_hyp = mirr_ray_intersection(coeffs_hyp_h, reflect1, vmirr_hyp)

                    hmirr_hyp_points_rotated = rotate_points(hmirr_hyp, focus_apprx, -theta_y, -theta_z)
                else:
                    vmirr_hyp_points_rotated = rotate_points(vmirr_hyp, focus_apprx, -theta_y, -theta_z)
                    hmirr_hyp_points_rotated = rotate_points(hmirr_hyp, focus_apprx, -theta_y, -theta_z)
                # coeffs_hyp_h = rotate_z(coeffs_hyp_h,-theta_z,focus_apprx)
                # coeffs_hyp_h = rotate_y(coeffs_hyp_h,-theta_y,focus_apprx)
                # coeffs_hyp_v = rotate_z(coeffs_hyp_v,-theta_z,focus_apprx)
                # coeffs_hyp_v = rotate_y(coeffs_hyp_v,-theta_y,focus_apprx)
                # if False:  ### この方法原因でエラーなるっぽい
                #     points = np.vstack([hmirr_hyp_points_rotated[:,0], hmirr_hyp_points_rotated[:,ray_num_H-1], hmirr_hyp_points_rotated[:,-1], hmirr_hyp_points_rotated[:,-ray_num_H]])
                #     print('hmirr_hyp_points_rotated[:,0]',hmirr_hyp_points_rotated[:,0])
                #     print('points',points.T)
                #     hmirr_hyp_points_rotated_grid = generate_grid_on_mirror_with_normal(coeffs_hyp_h, points.T, ray_num_V,ray_num_H)
                #     points = np.vstack([vmirr_hyp_points_rotated[:,0], vmirr_hyp_points_rotated[:,ray_num_H-1], vmirr_hyp_points_rotated[:,-1], vmirr_hyp_points_rotated[:,-ray_num_H]])
                #     vmirr_hyp_points_rotated_grid = generate_grid_on_mirror_with_normal(coeffs_hyp_v, points.T, ray_num_V,ray_num_H)
                #
                #     print('1st V mirror')
                #     CalcDataPitch(vmirr_hyp_points_rotated_grid, ray_num_V,ray_num_H)
                #     print('2nd H mirror')
                #     CalcDataPitch(hmirr_hyp_points_rotated_grid, ray_num_V,ray_num_H)
                # else:
                hmirr_hyp_points_rotated_grid = hmirr_hyp_points_rotated.copy()
                vmirr_hyp_points_rotated_grid = vmirr_hyp_points_rotated.copy()
                print('1st V mirror')
                CalcDataPitch(vmirr_hyp_points_rotated_grid, ray_num_V,ray_num_H)
                print('2nd H mirror')
                CalcDataPitch(hmirr_hyp_points_rotated_grid, ray_num_V,ray_num_H)

            else:
                reflect2_rotated_org = reflect2
                hmirr_hyp_points_rotated = hmirr_hyp
                vmirr_hyp_points_rotated = vmirr_hyp
                source_rotated = source
            coeffs_det = np.zeros(10)
            coeffs_det[6] = 1
            coeffs_det[9] = -(s2f_middle + defocus)
            detcenter = plane_ray_intersection(coeffs_det, reflect2_rotated_org, hmirr_hyp_points_rotated_org)

            vec0to1 = normalize_vector(vmirr_hyp_points_rotated_grid - source_rotated)
            vec1to2 = normalize_vector(hmirr_hyp_points_rotated_grid - vmirr_hyp_points_rotated_grid)
            vec2to3 = normalize_vector(detcenter - hmirr_hyp_points_rotated_grid)

            vmirr_norm = normalize_vector( (-vec1to2 + vec0to1) / 2 )
            hmirr_norm = normalize_vector( (-vec2to3 + vec1to2) / 2 )
            if np.abs(defocusForWave) > 1e-9:
                coeffs_det2 = np.zeros(10)
                coeffs_det2[6] = 1
                coeffs_det2[9] = -(s2f_middle + defocus+defocusForWave)
                detcenter2 = plane_ray_intersection(coeffs_det2, reflect2_rotated_org, hmirr_hyp_points_rotated_org)
                return source_rotated, vmirr_hyp_points_rotated_grid, hmirr_hyp_points_rotated_grid, detcenter, detcenter2, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vec0to1, vec1to2
            else:
                return source_rotated, vmirr_hyp_points_rotated_grid, hmirr_hyp_points_rotated_grid, detcenter, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vec0to1, vec1to2

        option_tilt = True
        if option_tilt:
            theta_y = -np.mean(np.arctan(angle[2, :]/angle[0, :]))
            theta_z = np.mean(np.arctan(angle[1, :]/angle[0, :]))
            reflect2_rotated = rotate_vectors(reflect2, -theta_y, -theta_z)
            focus_apprx = np.mean(detcenter,axis=1)
            hmirr_hyp_points_rotated = rotate_points(hmirr_hyp, focus_apprx, -theta_y, -theta_z)
            coeffs_det = np.zeros(10)
            coeffs_det[6] = 1
            coeffs_det[9] = -(s2f_middle + defocus)
            detcenter = plane_ray_intersection(coeffs_det, reflect2_rotated, hmirr_hyp_points_rotated)

            hmirr_hyp = hmirr_hyp_points_rotated
            reflect2 = reflect2_rotated
            angle = reflect2

        if option == 'sep':
            focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2 = compare_sep(reflect2_rotated, hmirr_hyp_points_rotated, coeffs_det,ray_num_H,1e-4)
            return focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2
        if option == 'sep_direct':
            focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2 = compare_sep(reflect2_rotated, hmirr_hyp_points_rotated, coeffs_det,ray_num_H,1e-4)
            return focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2
        if option == 'ray_wave':
            if option_HighNA == True:
                defocusWave = 1e-4
                lambda_ = 13.5
            else:
                defocusWave = 1e-5
                lambda_ = 1.35
            coeffs_det2 = np.zeros(10)
            coeffs_det2[6] = 1
            coeffs_det2[9] = -(s2f_middle + defocus + defocusWave)
            detcenter2 = plane_ray_intersection(coeffs_det2, reflect2, hmirr_hyp)

            dist2tofocus = np.linalg.norm(detcenter - hmirr_hyp, axis=0)
            vector2tofocus = (detcenter - hmirr_hyp) / dist2tofocus
            totalDist = dist0to1 + dist1to2 + dist2tofocus
            DistError = (totalDist - np.mean(totalDist))*1e9

            dist2tofocus2 = np.linalg.norm(detcenter2 - hmirr_hyp, axis=0)
            vector2tofocus2 = (detcenter2 - hmirr_hyp) / dist2tofocus
            totalDist2 = dist0to1 + dist1to2 + dist2tofocus2
            DistError2 = (totalDist2 - np.mean(totalDist2))*1e9

            # 補間するグリッドを作成
            grid_H, grid_V = np.meshgrid(
                np.linspace(detcenter2[1, :].min(), detcenter2[1, :].max(), ray_num_H),
                np.linspace(detcenter2[2, :].min(), detcenter2[2, :].max(), ray_num_V)
            )

            CosAngle = angle[0,:]
            # グリッド上にデータを補間 (method: 'linear', 'nearest', 'cubic' から選択)
            if False:
                matrixDistError2 = griddata((detcenter2[1, :], detcenter2[2, :]), DistError2, (grid_H, grid_V), method='cubic')
                meanFocus = np.mean(detcenter,axis=1)
                Sph = np.linalg.norm(detcenter2 - meanFocus[:, np.newaxis], axis=0) * 1e9
                matrixSph2 = griddata((detcenter2[1, :], detcenter2[2, :]), Sph, (grid_H, grid_V), method='cubic')

                # matrixAngle2 = griddata((detcenter2[1, :], detcenter2[2, :]), CosAngle, (grid_H, grid_V), method='cubic')
                # matrixWave2 = matrixDistError2 * matrixAngle2 - matrixSph2

                matrixWave2 = matrixDistError2 - matrixSph2
                matrixWave2 = matrixWave2 - np.nanmean(matrixWave2)
            else:
                # matrixDistError2 = griddata((detcenter2[1, :], detcenter2[2, :]), DistError2, (grid_H, grid_V), method='cubic')
                meanFocus = np.mean(detcenter,axis=1)
                Sph = np.linalg.norm(detcenter2 - meanFocus[:, np.newaxis], axis=0) * 1e9

                Wave2 = DistError2 - Sph
                print('grid_H.shape',grid_H.shape)
                print('Wave2.shape',Wave2.shape)
                print('detcenter2.shape',detcenter2.shape)
                matrixWave2 = griddata((detcenter2[1, :], detcenter2[2, :]), Wave2, (grid_H, grid_V), method='cubic')
                matrixWave2 = matrixWave2 - np.nanmean(matrixWave2)

            np.savetxt('matrixWave2(nm).txt',matrixWave2)
            tifffile.imwrite('matrixWave2(nm).tiff', matrixWave2)


            # plt.figure()
            # plt.pcolormesh(grid_H, grid_V, matrixWave2, cmap='jet', shading='auto',vmin = -1,vmax = 1)
            # # plt.colorbar(label='\u03BB')
            # plt.colorbar(label='wavefront error (nm)')
            # plt.savefig('waveRaytrace.png')
            # plt.show()

            matrixWave2_Corrected = plane_correction_with_nan_and_outlier_filter(matrixWave2)

            print('PV',np.nanmax(matrixWave2_Corrected)-np.nanmin(matrixWave2_Corrected))

            plt.figure()
            plt.pcolormesh(grid_H, grid_V, matrixWave2_Corrected/lambda_, cmap='jet', shading='auto',vmin = -1/4,vmax = 1/4)
            plt.colorbar(label='\u03BB')
            # plt.colorbar(label='wavefront error (nm)')
            plt.title(f'PV 6σ={np.nanstd(matrixWave2_Corrected/lambda_)*6}')
            plt.savefig(os.path.join(directory_name,'waveRaytrace_Corrected.png'), transparent=True, dpi=300)

            plt.figure()
            plt.pcolormesh(grid_H, grid_V, matrixWave2_Corrected/lambda_, cmap='jet', shading='auto',vmin = -1/128,vmax = 1/128)
            plt.colorbar(label='\u03BB')
            # plt.colorbar(label='wavefront error (nm)')
            plt.title(f'PV 6σ={np.nanstd(matrixWave2_Corrected/lambda_)*6}')
            plt.savefig(os.path.join(directory_name,'waveRaytrace_Corrected_2.png'), transparent=True, dpi=300)

            np.savetxt(os.path.join(directory_name, 'matrixWave2_Corrected(lambda).txt'), matrixWave2_Corrected/lambda_)
            # plt.show()
            pv = np.nanstd(matrixWave2_Corrected/lambda_)*6

            # plt.figure()
            # sample_detcenter = detcenter.copy()
            # sample_DistError = DistError.copy()
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

            return pv

        if option == 'ray':
            # print('WD_design',s2f_middle + defocus)
            print('ID',np.min(hmirr_hyp[0,:]) - np.max(vmirr_hyp[0,:]))
            print('WD',np.min(detcenter[0,:]) - np.max(hmirr_hyp[0,:]))
            print('diverg angle H',np.arctan(y1_h / x1_h) - np.arctan(y2_h / x2_h))
            print('diverg angle V',np.arctan(y1_v / x1_v) - np.arctan(y2_v / x2_v))
            print(theta1_v)
            print(np.cos(theta1_v)*s2f_middle)
            print(theta1_h)
            print(np.mean(detcenter[0,:]))
            print(np.mean(detcenter[1,:]))
            print(np.mean(detcenter[2,:]))
            print(coeffs_det)
            print('s2f_H',s2f_H)
            print('s2f_V',s2f_V)
            mabiki = round(np.sqrt(ray_num_H*ray_num_V)/50)
            defocussize = 4e-6

            # defocussize = 2e-6
            # print(np.std(detcenter0[0,:]))
            # print(np.std(detcenter[0,:]))
            print('s2f_middle + defocus',s2f_middle + defocus)
            # f_stand = s2f_middle + defocus
            # f_stand = 146.6959224409841
            coeffs_det = np.zeros(10)
            coeffs_det[6] = 1
            coeffs_det[9] = -(s2f_middle + defocus) + defocussize
            detcenter1 = plane_ray_intersection(coeffs_det, reflect2, hmirr_hyp)

            coeffs_det = np.zeros(10)
            coeffs_det[6] = 1
            coeffs_det[9] = -(s2f_middle + defocus) - defocussize
            detcenter2 = plane_ray_intersection(coeffs_det, reflect2, hmirr_hyp)

            # plot_ray_sideview(18,10,5,reflect1,vmirr_hyp,ray_num)
            # plot_ray_sideview(0.2,0.2,5,reflect1,vmirr_hyp,ray_num)

            phai0 = normalize_vector(phai0)

            plt.figure()
            plt.scatter(phai0[1, :], phai0[2, :])
            plt.scatter(phai0[1, ::ray_num_H], phai0[2, ::ray_num_H],color='r')
            plt.scatter(phai0[1, round((ray_num_H-1)/2)::ray_num_H], phai0[2, round((ray_num_H-1)/2)::ray_num_H],color='y')
            plt.scatter(phai0[1, ray_num_H-1::ray_num_H], phai0[2, ray_num_H-1::ray_num_H],color='g')
            plt.title('angle')
            plt.xlabel('Horizontal Angle (rad)')
            plt.ylabel('Vertical Angle (rad)')
            plt.axis('equal')
            # plt.show()
            plt.close()

            plt.figure()
            plt.scatter(phai0[1, :], phai0[2, :])
            plt.scatter((phai0[1, :ray_num_H]), (phai0[2, :ray_num_H]),color='r')
            plt.scatter(phai0[1, round(ray_num_H*(ray_num_V-1)/2) : round(ray_num_H*(ray_num_V+1)/2)], phai0[2, round(ray_num_H*(ray_num_V-1)/2) : round(ray_num_H*(ray_num_V+1)/2)],color='y')
            plt.scatter((phai0[1, -ray_num_H:]), (phai0[2, -ray_num_H:]),color='g')
            plt.title('angle')
            plt.xlabel('Horizontal Angle (rad)')
            plt.ylabel('Vertical Angle (rad)')
            plt.axis('equal')
            # plt.show()
            plt.close()

            vmirr_hyp = mirr_ray_intersection(coeffs_hyp_v, phai0, source)
            reflect1 = reflect_ray(phai0, norm_vector(coeffs_hyp_v, vmirr_hyp))

            angle_1st, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect1,norm_vector(coeffs_hyp_v, vmirr_hyp))

            if option_2mirror ==True:
                hmirr_hyp = mirr_ray_intersection(coeffs_hyp_h, reflect1, vmirr_hyp)
                reflect2 = reflect_ray(reflect1, norm_vector(coeffs_hyp_h, hmirr_hyp))
            else:
                hmirr_hyp = vmirr_hyp
                reflect2 = reflect1

            angle_2nd, angles_yx_rad, angles_zx_rad  = angle_between_2vector(reflect2,norm_vector(coeffs_hyp_h, hmirr_hyp))
            coeffs_det = np.zeros(10)
            coeffs_det[6] = 1
            coeffs_det[9] = -(s2f_middle + defocus)

            detcenter0 = plane_ray_intersection(coeffs_det, reflect2, hmirr_hyp)

            angles_between_rad, angles_yx_rad, angles_zx_rad = angle_between_2vector(reflect_ray(reflect2, norm_vector(coeffs_det, detcenter0)),norm_vector(coeffs_det, detcenter0))

            print('NA_set',NA_h)
            print(np.sin((np.max(angles_yx_rad) - np.min(angles_yx_rad))/2))
            print('NA_set',NA_v)
            print(np.sin((np.max(angles_zx_rad) - np.min(angles_zx_rad))/2))

            if option_tilt:
                hmirr_hyp0 = hmirr_hyp
                reflect2_rotated = rotate_vectors(reflect2, -theta_y, -theta_z)
                focus_apprx = np.mean(detcenter,axis=1)
                hmirr_hyp_points_rotated = rotate_points(hmirr_hyp, focus_apprx, -theta_y, -theta_z)
                coeffs_det = np.zeros(10)
                coeffs_det[6] = 1
                coeffs_det[9] = -(s2f_middle + defocus)
                detcenter = plane_ray_intersection(coeffs_det, reflect2_rotated, hmirr_hyp_points_rotated)
                hmirr_hyp = hmirr_hyp_points_rotated
                reflect2 = reflect2_rotated
                angle = reflect2

            # 範囲内の値を間引く
            original_array = list(range(len(detcenter1[0,:])))
            first_thinned_array = original_array[::ray_num_H]
            thinned_array_v_r = first_thinned_array[::mabiki]
            # 範囲内の値を間引く
            original_array = list(range(len(detcenter1[0,:])))
            first_thinned_array = original_array[round((ray_num_H-1)/2)::ray_num_H]
            thinned_array_v_y = first_thinned_array[::mabiki]
            # 範囲内の値を間引く
            original_array = list(range(len(detcenter1[0,:])))
            first_thinned_array = original_array[ray_num_H-1::ray_num_H]
            thinned_array_v_g = first_thinned_array[::mabiki]
            # 範囲内の値を間引く
            start = 0
            end = ray_num_H
            thinned_array_h_r = crop(start, end, mabiki)
            # 範囲内の値を間引く
            start = round(ray_num_H*(ray_num_V-1)/2)
            end = round(ray_num_H*(ray_num_V+1)/2)
            thinned_array_h_y = crop(start, end, mabiki)
            # 範囲内の値を間引く
            start = ray_num_H*ray_num_V - ray_num_H
            end = ray_num_H*ray_num_V
            thinned_array_h_g = crop(start, end, mabiki)

            # プロットの準備
            fig, axs = plt.subplots(2, 2, figsize=(10, 15))  # 2つのプロットを並べる
            # fig, axs = plt.subplots(2, 4, figsize=(10, 45))  # 2つのプロットを並べる
            input_val  = -coeffs_det[9]

            # 初期のプロット
            third_r = len(thinned_array_h_r) *2 // 3
            third_y = len(thinned_array_h_y) *2 // 3
            third_g = len(thinned_array_h_g) *2 // 3

            # 前1/3のプロット（赤）
            axs[0, 0].plot([detcenter1[0, thinned_array_h_r[:third_r]], detcenter2[0, thinned_array_h_r[:third_r]]],
                           [detcenter1[1, thinned_array_h_r[:third_r]], detcenter2[1, thinned_array_h_r[:third_r]]], c='r')

            # 後ろ1/3のプロット（ピンク）
            axs[0, 0].plot([detcenter1[0, thinned_array_h_r[-third_r:]], detcenter2[0, thinned_array_h_r[-third_r:]]],
                           [detcenter1[1, thinned_array_h_r[-third_r:]], detcenter2[1, thinned_array_h_r[-third_r:]]], c='purple')

            # thinned_array_h_y の前1/3のプロット（darkyellow）
            axs[0, 0].plot([detcenter1[0, thinned_array_h_y[:third_y]], detcenter2[0, thinned_array_h_y[:third_y]]],
                           [detcenter1[1, thinned_array_h_y[:third_y]], detcenter2[1, thinned_array_h_y[:third_y]]], c='y')

            # thinned_array_h_y の後ろ1/3のプロット（purple）
            axs[0, 0].plot([detcenter1[0, thinned_array_h_y[-third_y:]], detcenter2[0, thinned_array_h_y[-third_y:]]],
                           [detcenter1[1, thinned_array_h_y[-third_y:]], detcenter2[1, thinned_array_h_y[-third_y:]]], c='#B8860B')

            # thinned_array_h_g の前1/3のプロット（緑）
            axs[0, 0].plot([detcenter1[0, thinned_array_h_g[:third_g]], detcenter2[0, thinned_array_h_g[:third_g]]],
                           [detcenter1[1, thinned_array_h_g[:third_g]], detcenter2[1, thinned_array_h_g[:third_g]]], c='g')

            # thinned_array_h_g の後ろ1/3のプロット（薄緑）
            axs[0, 0].plot([detcenter1[0, thinned_array_h_g[-third_g:]], detcenter2[0, thinned_array_h_g[-third_g:]]],
                           [detcenter1[1, thinned_array_h_g[-third_g:]], detcenter2[1, thinned_array_h_g[-third_g:]]], c='lightgreen')
            axs[0,0].plot([input_val, input_val],
                        [np.min(detcenter2[1, :]), np.max(detcenter1[1, :])], color='k')
            axs[0,0].set_title('V aperture 0')
            axs[0,0].set_xlabel('Axial (m)')
            axs[0,0].set_ylabel('Horizontal Position (m)')

            axs[1,1].scatter(detcenter[1, :], detcenter[2, :])
            axs[1,1].scatter(detcenter[1, ::ray_num_H], detcenter[2, ::ray_num_H],color='r')
            axs[1,1].scatter(detcenter[1, round((ray_num_H-1)/2)::ray_num_H], detcenter[2, round((ray_num_H-1)/2)::ray_num_H],color='y')
            axs[1,1].scatter(detcenter[1, ray_num_H-1::ray_num_H], detcenter[2, ray_num_H-1::ray_num_H],color='g')
            axs[1,1].scatter(detcenter[1, ray_num_H-1::ray_num_H-1][:-1], detcenter[2, ray_num_H-1::ray_num_H-1][:-1],color='k')
            axs[1,1].scatter(detcenter[1, ::ray_num_H+1], detcenter[2, ::ray_num_H+1],color='gray')
            axs[1,1].scatter(np.mean(detcenter[1, ::ray_num_H]), np.mean(detcenter[2, ::ray_num_H]),color='r',marker='x',s=100)
            axs[1,1].scatter(np.mean(detcenter[1, round((ray_num_H-1)/2)::ray_num_H]), np.mean(detcenter[2, round((ray_num_H-1)/2)::ray_num_H]),color='y',marker='x',s=100)
            axs[1,1].scatter(np.mean(detcenter[1, ray_num_H-1::ray_num_H]), np.mean(detcenter[2, ray_num_H-1::ray_num_H]),color='g',marker='x',s=100)
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
            axs[1, 0].plot([detcenter1[0, thinned_array_v_r[:third_r]], detcenter2[0, thinned_array_v_r[:third_r]]],
                           [detcenter1[2, thinned_array_v_r[:third_r]], detcenter2[2, thinned_array_v_r[:third_r]]], c='r')

            # 後ろ1/3のプロット（ピンク）
            axs[1, 0].plot([detcenter1[0, thinned_array_v_r[-third_r:]], detcenter2[0, thinned_array_v_r[-third_r:]]],
                           [detcenter1[2, thinned_array_v_r[-third_r:]], detcenter2[2, thinned_array_v_r[-third_r:]]], c='purple')

            # thinned_array_v_y の前1/3のプロット（darkyellow）
            axs[1, 0].plot([detcenter1[0, thinned_array_v_y[:third_y]], detcenter2[0, thinned_array_v_y[:third_y]]],
                           [detcenter1[2, thinned_array_v_y[:third_y]], detcenter2[2, thinned_array_v_y[:third_y]]], c='y')

            # thinned_array_v_y の後ろ1/3のプロット（purple）
            axs[1, 0].plot([detcenter1[0, thinned_array_v_y[-third_y:]], detcenter2[0, thinned_array_v_y[-third_y:]]],
                           [detcenter1[2, thinned_array_v_y[-third_y:]], detcenter2[2, thinned_array_v_y[-third_y:]]], c='#B8860B')

            # thinned_array_v_g の前1/3のプロット（緑）
            axs[1, 0].plot([detcenter1[0, thinned_array_v_g[:third_g]], detcenter2[0, thinned_array_v_g[:third_g]]],
                           [detcenter1[2, thinned_array_v_g[:third_g]], detcenter2[2, thinned_array_v_g[:third_g]]], c='g')

            # thinned_array_v_g の後ろ1/3のプロット（薄緑）
            axs[1, 0].plot([detcenter1[0, thinned_array_v_g[-third_g:]], detcenter2[0, thinned_array_v_g[-third_g:]]],
                           [detcenter1[2, thinned_array_v_g[-third_g:]], detcenter2[2, thinned_array_v_g[-third_g:]]], c='lightgreen')

            axs[1,0].plot([input_val, input_val],
                        [np.min(detcenter2[2, :]), np.max(detcenter1[2, :])], color='k')
            axs[1,0].set_title('H aperture 0')
            axs[1,0].set_xlabel('Axial (m)')
            axs[1,0].set_ylabel('Vertical Position (m)')

            axs[0,1].scatter(detcenter[1, :], detcenter[2, :])
            axs[0,1].scatter(detcenter[1, :ray_num_H], detcenter[2, :ray_num_H],color='r')
            axs[0,1].scatter(detcenter[1, round(ray_num_H*(ray_num_V-1)/2) : round(ray_num_H*(ray_num_V+1)/2)], detcenter[2, round(ray_num_H*(ray_num_V-1)/2) : round(ray_num_H*(ray_num_V+1)/2)],color='y')
            axs[0,1].scatter(detcenter[1, -ray_num_H:], detcenter[2, -ray_num_H:],color='g')
            axs[0,1].scatter(detcenter[1, ray_num_H-1::ray_num_H-1][:-1], detcenter[2, ray_num_H-1::ray_num_H-1][:-1],color='k')
            axs[0,1].scatter(detcenter[1, ::ray_num_H+1], detcenter[2, ::ray_num_H+1],color='gray')
            axs[0,1].scatter(np.mean(detcenter[1, :ray_num_H]), np.mean(detcenter[2, :ray_num_H]),color='r',marker='x',s=100)
            axs[0,1].scatter(np.mean(detcenter[1, round(ray_num_H*(ray_num_V-1)/2) : round(ray_num_H*(ray_num_V+1)/2)]), np.mean(detcenter[2, round(ray_num_H*(ray_num_V-1)/2) : round(ray_num_H*(ray_num_V+1)/2)]),color='y',marker='x',s=100)
            axs[0,1].scatter(np.mean(detcenter[1, -ray_num_H:]), np.mean(detcenter[2, -ray_num_H:]),color='g',marker='x',s=100)
            axs[0,1].set_title('focus @V aperture 0')
            axs[0,1].set_xlabel('Horizontal (m)')
            axs[0,1].set_ylabel('Vertical (m)')
            axs[0,1].axis('equal')

            # タイトル用の新しいサイズ計算
            size_v = np.max(detcenter[2,:]) - np.min(detcenter[2,:])
            size_h = np.max(detcenter[1,:]) - np.min(detcenter[1,:])

            # タイトルの更新
            title1 = f'Params 0-1: {params[0:2]}'
            title2 = f'Params 2-7: {params[2:8]}'
            title3 = f'Params 8-13: {params[8:14]}'
            title4 = f'Size V: {size_v}'
            title5 = f'Size H: {size_h}'

            fig.suptitle(f'{title1}\n{title2}\n{title3}\n{title4}\n{title5}', fontsize=12)
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])
            fig.tight_layout(pad=4.0)  # パディングを調整
            # マウスイベントでクリックした位置のx座標を取得してプロットを更新
            def on_click(event):
                if event.inaxes == axs[0,0] or event.inaxes == axs[1,0]:  # クリックが左のプロット内で行われたか確認
                    input_val = event.xdata  # x座標を取得
                    coeffs_det[9] = -input_val
                    detcenter = plane_ray_intersection(coeffs_det, reflect2, hmirr_hyp)
                    # 既存の範囲を保持するためにxlimとylimを記録
                    xlim_0_0 = axs[0,0].get_xlim()
                    ylim_0_0 = axs[0,0].get_ylim()
                    xlim_1_0 = axs[1,0].get_xlim()
                    ylim_1_0 = axs[1,0].get_ylim()

                    # input_valを使って再計算（例として新しいプロットを追加）
                    axs[1,1].cla()  # 右側プロットをクリア
                    axs[1,1].scatter(detcenter[1, :], detcenter[2, :])
                    axs[1,1].scatter(detcenter[1, ::ray_num_H], detcenter[2, ::ray_num_H],color='r')
                    axs[1,1].scatter(detcenter[1, round((ray_num_H-1)/2)::ray_num_H], detcenter[2, round((ray_num_H-1)/2)::ray_num_H],color='y')
                    axs[1,1].scatter(detcenter[1, ray_num_H-1::ray_num_H], detcenter[2, ray_num_H-1::ray_num_H],color='g')
                    axs[1,1].scatter(detcenter[1, ray_num_H-1::ray_num_H-1][:-1], detcenter[2, ray_num_H-1::ray_num_H-1][:-1],color='k')
                    axs[1,1].scatter(detcenter[1, ::ray_num_H+1], detcenter[2, ::ray_num_H+1],color='gray')
                    axs[1,1].scatter(np.mean(detcenter[1, ::ray_num_H]), np.mean(detcenter[2, ::ray_num_H]),color='r',marker='x',s=100)
                    axs[1,1].scatter(np.mean(detcenter[1, round((ray_num_H-1)/2)::ray_num_H]), np.mean(detcenter[2, round((ray_num_H-1)/2)::ray_num_H]),color='y',marker='x',s=100)
                    axs[1,1].scatter(np.mean(detcenter[1, ray_num_H-1::ray_num_H]), np.mean(detcenter[2, ray_num_H-1::ray_num_H]),color='g',marker='x',s=100)
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
                    axs[0, 0].plot([detcenter1[0, thinned_array_h_r[:third_r]], detcenter2[0, thinned_array_h_r[:third_r]]],
                                   [detcenter1[1, thinned_array_h_r[:third_r]], detcenter2[1, thinned_array_h_r[:third_r]]], c='r')

                    # 後ろ1/3のプロット（ピンク）
                    axs[0, 0].plot([detcenter1[0, thinned_array_h_r[-third_r:]], detcenter2[0, thinned_array_h_r[-third_r:]]],
                                   [detcenter1[1, thinned_array_h_r[-third_r:]], detcenter2[1, thinned_array_h_r[-third_r:]]], c='purple')

                    # thinned_array_h_y の前1/3のプロット（darkyellow）
                    axs[0, 0].plot([detcenter1[0, thinned_array_h_y[:third_y]], detcenter2[0, thinned_array_h_y[:third_y]]],
                                   [detcenter1[1, thinned_array_h_y[:third_y]], detcenter2[1, thinned_array_h_y[:third_y]]], c='y')

                    # thinned_array_h_y の後ろ1/3のプロット（purple）
                    axs[0, 0].plot([detcenter1[0, thinned_array_h_y[-third_y:]], detcenter2[0, thinned_array_h_y[-third_y:]]],
                                   [detcenter1[1, thinned_array_h_y[-third_y:]], detcenter2[1, thinned_array_h_y[-third_y:]]], c='#B8860B')

                    # thinned_array_h_g の前1/3のプロット（緑）
                    axs[0, 0].plot([detcenter1[0, thinned_array_h_g[:third_g]], detcenter2[0, thinned_array_h_g[:third_g]]],
                                   [detcenter1[1, thinned_array_h_g[:third_g]], detcenter2[1, thinned_array_h_g[:third_g]]], c='g')

                    # thinned_array_h_g の後ろ1/3のプロット（薄緑）
                    axs[0, 0].plot([detcenter1[0, thinned_array_h_g[-third_g:]], detcenter2[0, thinned_array_h_g[-third_g:]]],
                                   [detcenter1[1, thinned_array_h_g[-third_g:]], detcenter2[1, thinned_array_h_g[-third_g:]]], c='lightgreen')
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
                    axs[1, 0].plot([detcenter1[0, thinned_array_v_r[:third_r]], detcenter2[0, thinned_array_v_r[:third_r]]],
                                   [detcenter1[2, thinned_array_v_r[:third_r]], detcenter2[2, thinned_array_v_r[:third_r]]], c='r')

                    # 後ろ1/3のプロット（ピンク）
                    axs[1, 0].plot([detcenter1[0, thinned_array_v_r[-third_r:]], detcenter2[0, thinned_array_v_r[-third_r:]]],
                                   [detcenter1[2, thinned_array_v_r[-third_r:]], detcenter2[2, thinned_array_v_r[-third_r:]]], c='purple')

                    # thinned_array_v_y の前1/3のプロット（darkyellow）
                    axs[1, 0].plot([detcenter1[0, thinned_array_v_y[:third_y]], detcenter2[0, thinned_array_v_y[:third_y]]],
                                   [detcenter1[2, thinned_array_v_y[:third_y]], detcenter2[2, thinned_array_v_y[:third_y]]], c='y')

                    # thinned_array_v_y の後ろ1/3のプロット（purple）
                    axs[1, 0].plot([detcenter1[0, thinned_array_v_y[-third_y:]], detcenter2[0, thinned_array_v_y[-third_y:]]],
                                   [detcenter1[2, thinned_array_v_y[-third_y:]], detcenter2[2, thinned_array_v_y[-third_y:]]], c='#B8860B')

                    # thinned_array_v_g の前1/3のプロット（緑）
                    axs[1, 0].plot([detcenter1[0, thinned_array_v_g[:third_g]], detcenter2[0, thinned_array_v_g[:third_g]]],
                                   [detcenter1[2, thinned_array_v_g[:third_g]], detcenter2[2, thinned_array_v_g[:third_g]]], c='g')

                    # thinned_array_v_g の後ろ1/3のプロット（薄緑）
                    axs[1, 0].plot([detcenter1[0, thinned_array_v_g[-third_g:]], detcenter2[0, thinned_array_v_g[-third_g:]]],
                                   [detcenter1[2, thinned_array_v_g[-third_g:]], detcenter2[2, thinned_array_v_g[-third_g:]]], c='lightgreen')

                    axs[1,0].plot([input_val, input_val],
                                [np.min(detcenter2[2, :]), np.max(detcenter1[2, :])], color='k')
                    axs[1,0].set_title('H aperture 0')
                    axs[1,0].set_xlabel('Axial (m)')
                    axs[1,0].set_ylabel('Vertical Position (m)')

                    axs[0,1].cla()  # 右側プロットをクリア
                    axs[0,1].scatter(detcenter[1, :], detcenter[2, :])
                    axs[0,1].scatter(detcenter[1, :ray_num_H], detcenter[2, :ray_num_H],color='r')
                    axs[0,1].scatter(detcenter[1, round(ray_num_H*(ray_num_V-1)/2) : round(ray_num_H*(ray_num_V+1)/2)], detcenter[2, round(ray_num_H*(ray_num_V-1)/2) : round(ray_num_H*(ray_num_V+1)/2)],color='y')
                    axs[0,1].scatter(detcenter[1, -ray_num_H:], detcenter[2, -ray_num_H:],color='g')
                    axs[0,1].scatter(detcenter[1, ray_num_H-1::ray_num_H-1][:-1], detcenter[2, ray_num_H-1::ray_num_H-1][:-1],color='k')
                    axs[0,1].scatter(detcenter[1, ::ray_num_H+1], detcenter[2, ::ray_num_H+1],color='gray')
                    axs[0,1].scatter(np.mean(detcenter[1, :ray_num_H]), np.mean(detcenter[2, :ray_num_H]),color='r',marker='x',s=100)
                    axs[0,1].scatter(np.mean(detcenter[1, round(ray_num_H*(ray_num_V-1)/2) : round(ray_num_H*(ray_num_V+1)/2)]), np.mean(detcenter[2, round(ray_num_H*(ray_num_V-1)/2) : round(ray_num_H*(ray_num_V+1)/2)]),color='y',marker='x',s=100)
                    axs[0,1].scatter(np.mean(detcenter[1, -ray_num_H:]), np.mean(detcenter[2, -ray_num_H:]),color='g',marker='x',s=100)
                    axs[0,1].set_title('focus @V aperture 0')
                    axs[0,1].set_xlabel('Horizontal (m)')
                    axs[0,1].set_ylabel('Vertical (m)')
                    axs[0,1].axis('equal')

                    # axs[0,2].scatter(input_val,np.mean(detcenter[1, :ray_num]),color='r')
                    # axs[0,2].scatter(input_val,np.mean(detcenter[1, round((ray_num**2)/2) : round((ray_num**2 + ray_num*2)/2)]),color='y')
                    # axs[0,2].scatter(input_val,np.mean(detcenter[1, -ray_num:-1]),color='g')

                    # axs[1,2].scatter(input_val,np.mean(detcenter[2, ::ray_num]),color='r')
                    # axs[1,2].scatter(input_val,np.mean(detcenter[2, round(ray_num/2)-1::ray_num]),color='y')
                    # axs[1,2].scatter(input_val,np.mean(detcenter[2, ray_num-1::ray_num]),color='g')
                    # axs[1,3].scatter(input_val,np.mean(detcenter[1, ::ray_num]),color='r')
                    # axs[1,3].scatter(input_val,np.mean(detcenter[1, round(ray_num/2)-1::ray_num]),color='y')
                    # axs[1,3].scatter(input_val,np.mean(detcenter[1, ray_num-1::ray_num]),color='g')

                    axs[0,0].set_xlim(xlim_0_0)  # クリア後に元の範囲を再設定
                    axs[0,0].set_ylim(ylim_0_0)

                    axs[1,0].set_xlim(xlim_1_0)
                    axs[1,0].set_ylim(ylim_1_0)

                    # タイトル用の新しいサイズ計算
                    size_v = np.max(detcenter[2,:]) - np.min(detcenter[2,:])
                    size_h = np.max(detcenter[1,:]) - np.min(detcenter[1,:])

                    std_obl1 = np.sqrt(np.std(detcenter[1, ray_num_H-1::ray_num_H-1][:-1])**2 + np.std(detcenter[2, ray_num_H-1::ray_num_H-1][:-1])**2)
                    std_obl2 = np.sqrt(np.std(detcenter[1, ::ray_num_H+1])**2 + np.std(detcenter[2, ::ray_num_H+1])**2)
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('std_obl1',std_obl1)
                    print('std_obl2',std_obl2)
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # タイトルの更新
                    title1 = f'Params 0-1: {params[0:2]}'
                    title2 = f'Params 2-7: {params[2:8]}'
                    title3 = f'Params 8-13: {params[8:14]}'
                    title4 = f'Size V: {size_v}'
                    title5 = f'Size H: {size_h}'

                    fig.suptitle(f'{title1}\n{title2}\n{title3}\n{title4}\n{title5}', fontsize=12)

                    fig.canvas.draw_idle()  # 再描画
            # イベントリスナーを設定
            fig.canvas.mpl_connect('button_press_event', on_click)

            plt.savefig(os.path.join(directory_name,'multipleAroundFocus.png'), transparent=True, dpi=300)

            plt.figure(figsize=(10, 10))
            plt.scatter(detcenter[1, :], detcenter[2, :])
            plt.axis('equal')
            plt.savefig(os.path.join(directory_name,'spotdiagram.png'), transparent=True, dpi=300)
            # plt.show()
            # plt.close()

            # Crv_1 = mirr_ray_RoC(coeffs_hyp_v, phai0, vmirr_hyp)
            # plt.figure()
            # original_array = list(range(len(detcenter1[0,:])))
            # first_thinned_array = original_array[::ray_num_H]
            # thinned_array_v_r = first_thinned_array
            # # 範囲内の値を間引く
            # original_array = list(range(len(detcenter1[0,:])))
            # first_thinned_array = original_array[round((ray_num_H-1)/2)::ray_num_H]
            # thinned_array_v_y = first_thinned_array
            # # 範囲内の値を間引く
            # original_array = list(range(len(detcenter1[0,:])))
            # first_thinned_array = original_array[ray_num_H-1::ray_num_H]
            # thinned_array_v_g = first_thinned_array
            #
            # plt.plot(vmirr_hyp[0, thinned_array_v_r],Crv_1[thinned_array_v_r],c = 'r')
            # plt.plot(vmirr_hyp[0, thinned_array_v_y],Crv_1[thinned_array_v_y],c = 'y')
            # plt.plot(vmirr_hyp[0, thinned_array_v_g],Crv_1[thinned_array_v_g],c = 'g')
            #
            # # PNGファイルとして保存
            # plt.savefig('Curvature1.png', dpi=300)
            # # plt.show()
            # plt.close()
            #
            # plt.figure()
            # # 範囲内の値を間引く
            # start = 0
            # end = ray_num_H
            # thinned_array_h_r = crop(start, end, mabiki)
            # # 範囲内の値を間引く
            # start = round(ray_num_H*(ray_num_V-1)/2)
            # end = round(ray_num_H*(ray_num_V+1)/2)
            # thinned_array_h_y = crop(start, end, mabiki)
            # # 範囲内の値を間引く
            # start = ray_num_H*ray_num_V - ray_num_H
            # end = ray_num_H*ray_num_V
            # thinned_array_h_g = crop(start, end, mabiki)
            #
            # plt.plot(vmirr_hyp[1, thinned_array_h_r],Crv_1[thinned_array_h_r],c = 'r')
            # plt.plot(vmirr_hyp[1, thinned_array_h_y],Crv_1[thinned_array_h_y],c = 'y')
            # plt.plot(vmirr_hyp[1, thinned_array_h_g],Crv_1[thinned_array_h_g],c = 'g')
            #
            # # PNGファイルとして保存
            # plt.savefig('Curvature2.png', dpi=300)
            # # plt.show()
            # plt.close()
            #
            # Crv_2 = mirr_ray_RoC(coeffs_hyp_h, reflect1, hmirr_hyp0)
            # plt.figure()
            # original_array = list(range(len(detcenter1[0,:])))
            # first_thinned_array = original_array[::ray_num_H]
            # thinned_array_v_r = first_thinned_array
            # # 範囲内の値を間引く
            # original_array = list(range(len(detcenter1[0,:])))
            # first_thinned_array = original_array[round((ray_num_H-1)/2)::ray_num_H]
            # thinned_array_v_y = first_thinned_array
            # # 範囲内の値を間引く
            # original_array = list(range(len(detcenter1[0,:])))
            # first_thinned_array = original_array[ray_num_H-1::ray_num_H]
            # thinned_array_v_g = first_thinned_array
            #
            # plt.plot(hmirr_hyp0[2, thinned_array_v_r],Crv_2[thinned_array_v_r],c = 'r')
            # plt.plot(hmirr_hyp0[2, thinned_array_v_y],Crv_2[thinned_array_v_y],c = 'y')
            # plt.plot(hmirr_hyp0[2, thinned_array_v_g],Crv_2[thinned_array_v_g],c = 'g')
            # # PNGファイルとして保存
            # plt.savefig('Curvature3.png', dpi=300)
            # # plt.show()
            # plt.close()
            #
            # plt.figure()
            # # 範囲内の値を間引く
            # start = 0
            # end = ray_num_H
            # thinned_array_h_r = crop(start, end, mabiki)
            # # 範囲内の値を間引く
            # start = round(ray_num_H*(ray_num_V-1)/2)
            # end = round(ray_num_H*(ray_num_V+1)/2)
            # thinned_array_h_y = crop(start, end, mabiki)
            # # 範囲内の値を間引く
            # start = ray_num_H*ray_num_V - ray_num_H
            # end = ray_num_H*ray_num_V
            # thinned_array_h_g = crop(start, end, mabiki)
            #
            # plt.plot(hmirr_hyp0[0, thinned_array_h_r],Crv_2[thinned_array_h_r],c = 'r')
            # plt.plot(hmirr_hyp0[0, thinned_array_h_y],Crv_2[thinned_array_h_y],c = 'y')
            # plt.plot(hmirr_hyp0[0, thinned_array_h_g],Crv_2[thinned_array_h_g],c = 'g')
            # # PNGファイルとして保存
            # plt.savefig('Curvature4.png', dpi=300)
            # # plt.show()
            # plt.close()
            #
            # plt.figure()
            # scatter = plt.scatter(hmirr_hyp0[0, :], hmirr_hyp0[1, :],c=Crv_2, cmap='viridis')
            # # カラーバーを追加
            # plt.colorbar(scatter, label='Curvature (1/m)')
            # plt.title('Curvature @ 2nd Mirror')
            # plt.xlabel('Axial (m)')
            # plt.ylabel('Horizontal (m)')
            # plt.axis('equal')
            # # PNGファイルとして保存
            # plt.savefig('Curvature6.png', dpi=300)
            # # plt.show()
            # plt.close()
            #
            # # hmirr_hyp[1, :] の最小値を取得
            # min_value = np.min(hmirr_hyp0[1, :])
            #
            # # 最小値に近いデータを抽出するための範囲（例えば、最小値から ±epsilon 範囲でフィルタリング）
            # epsilon = 0.0008  # ここで範囲を適宜設定する
            # close_indices = np.abs(hmirr_hyp0[1, :] - min_value) <= epsilon
            #
            # # フィルタリングされたデータを用いてプロット
            # # filtered_hmirr_hyp = hmirr_hyp[:, close_indices]
            # # filtered_Crv_2 = Crv_2[close_indices]
            # filtered_hmirr_hyp = hmirr_hyp0[:, ray_num_H-1::ray_num_H]
            # filtered_Crv_2 = Crv_2[ray_num_H-1::ray_num_H]
            # # プロット
            # plt.figure()
            # scatter = plt.scatter(filtered_hmirr_hyp[0, :], filtered_hmirr_hyp[2, :], c=filtered_Crv_2, cmap='viridis', vmin=0.1213, vmax=0.1226)
            # # scatter = plt.scatter(filtered_hmirr_hyp[0, :], filtered_hmirr_hyp[2, :], c=filtered_Crv_2, cmap='viridis')
            # plt.colorbar(scatter, label='Curvature (1/m)')
            # # plt.title('Curvature @ 2nd Mirror (Near Minimum Horizontal Values)')
            # # plt.xlabel('Axial (m)')
            # # plt.ylabel('Vertical (m)')
            # # plt.axis('equal')
            # # # PNGファイルとして保存
            # # plt.savefig('Curvature7.png', dpi=300)
            # # plt.show()
            #
            #
            # # hmirr_hyp[1, :] の最小値を取得
            # max_value = np.max(hmirr_hyp[1, :])
            #
            # # 最小値に近いデータを抽出するための範囲（例えば、最小値から ±epsilon 範囲でフィルタリング）
            # epsilon = 0.001  # ここで範囲を適宜設定する
            # close_indices = np.abs(hmirr_hyp0[1, :] - max_value) <= epsilon
            #
            # # フィルタリングされたデータを用いてプロット
            # filtered_hmirr_hyp = hmirr_hyp0[:, ::ray_num_H]
            # filtered_Crv_2 = Crv_2[::ray_num_H]
            #
            # # プロット
            # # plt.figure()
            # scatter = plt.scatter(filtered_hmirr_hyp[0, :], filtered_hmirr_hyp[2, :], c=filtered_Crv_2, cmap='viridis', vmin=0.03980, vmax=0.04014)
            # # scatter = plt.scatter(filtered_hmirr_hyp[0, :], filtered_hmirr_hyp[2, :], c=filtered_Crv_2, cmap='viridis')
            # plt.colorbar(scatter, label='Curvature (1/m)')
            # # plt.title('Curvature @ 2nd Mirror (Near Maximum Horizontal Values)')
            # plt.xlabel('Axial (m)')
            # plt.ylabel('Vertical (m)')
            # plt.axis('equal')
            # # PNGファイルとして保存
            # plt.savefig('Curvature8.png', dpi=300)
            # # plt.show()
            # plt.close()

# 焦点面での標準偏差を計算
    return vmirr_hyp, hmirr_hyp, detcenter, angle
# 二次曲面の関数の定義
def quadratic_surface(xy, a, b, c, d, e, f):
    x, y = xy
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
def objective_function(var_params, fixed_params, var_indices, num_params,option_param):
    params = np.zeros(num_params)

    # 可変パラメータを適切な位置に挿入
    for i, var_idx in enumerate(var_indices):
        params[var_idx] = var_params[i]

    # 固定パラメータを適切な位置に挿入
    for i, fixed_idx in enumerate(fixed_params.keys()):
        params[fixed_idx] = fixed_params[fixed_idx]

    # area_min = auto_focus_forSimu(params,option_param)
    area_min = auto_focus_NA(50, params,1,1, False,'D')
    return area_min
#####################################
def auto_focus_NA(num_adj_astg,initial_params,na_ratio_h,na_ratio_v,option,option_param,option_disp='ray',option_mode=False,source_shift0=[0.,0.,0.],option_legendre=False):
    initial_a_min = -0.3 + initial_params[0].copy()  # Initial minimum value for 'a'
    initial_a_max = 0.3 + initial_params[0].copy() # Initial maximum value for 'a'
    shrink_factor = 0.1    # Range reduction factor per loop
    num_adj_astg = 100      # Number of steps for each range
    max_attempts = 16     # Maximum attempts to avoid infinite loop

    # Set the initial range
    a_min = initial_a_min
    a_max = initial_a_max
    if option_param == 'D':
        distance_ = np.linspace(-0.01, 0.01, num_adj_astg)

    attempt = 0  # Initialize attempt counter
    while attempt < max_attempts:
        # Generate list of 'a' values
        a = np.linspace(a_min, a_max, num_adj_astg)

        # Initialize size_v_ and size_h_
        size_v_ = np.zeros(num_adj_astg)
        size_h_ = np.zeros(num_adj_astg)

        # Compute for each 'a' value
        for i in range(len(a)):
            initial_params[0] = np.float64(a[i])
            if option_AKB:
                if option_mode == 'FoC':
                    vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle = plot_result_debug(initial_params, 'test',source_shift=source_shift0,option_tilt = False)
                else:
                    vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle = plot_result_debug(initial_params, 'test')
            else:
                vmirr_hyp, hmirr_hyp, detcenter, angle = KB_debug(initial_params, na_ratio_h, na_ratio_v, 'test')
            size_v_[i] = np.std(detcenter[2, :])
            size_h_[i] = np.std(detcenter[1, :])

        # Calculate astig_shift
        if not option_mode == 'FoC':
            astig_shift = a[np.argmin(size_h_)] - a[np.argmin(size_v_)]
            initial_params[1] = initial_params[1] - astig_shift

        # Get the minimum values
        size_v_param = np.min(size_v_)
        size_h_param = np.min(size_h_)

        # # plt
        # plt.plot(a, size_v_, label='size_v_')
        # plt.plot(a, size_h_, label='size_h_')
        # plt.xlabel('a')
        # plt.ylabel('Size')
        # plt.title(f'initial_params[1]={initial_params[1]}')
        # plt.legend()
        # # plt.grid()
        # plt.show()

        # Second check if not converged
        if not np.argmin(size_h_) == np.argmin(size_v_) and not option_mode == 'FoC':
            for i in range(len(a)):
                initial_params[0] = np.float64(a[i])
                if option_AKB:
                    if option_mode == 'FoC':
                        vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle = plot_result_debug(initial_params, 'test',source_shift=source_shift0,option_tilt = False)
                    else:
                        vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle = plot_result_debug(initial_params, 'test')
                else:
                    vmirr_hyp, hmirr_hyp, detcenter, angle = KB_debug(initial_params, na_ratio_h, na_ratio_v, 'test')
                size_v_[i] = np.std(detcenter[2, :])
                size_h_[i] = np.std(detcenter[1, :])
                if option_param == 'D':
                    distance_[i] = np.mean(np.sqrt((detcenter[1, :] - np.mean(detcenter[1, :]))**2 + (detcenter[2, :] - np.mean(detcenter[2, :]))**2))

            astig_shift = a[np.argmin(size_h_)] - a[np.argmin(size_v_)]
            # if option_wolter_3_1:
            #     astig_shift = -astig_shift
            initial_params[1] = initial_params[1] - astig_shift
            size_v_param = np.min(size_v_)
            size_h_param = np.min(size_h_)

        # Set the optimal 'a'
        initial_params[0] = a[np.argmin(size_h_)]

        # Update the range for 'a'
        best_a = initial_params[0]
        delta_a = (a_max - a_min) * shrink_factor
        a_min = best_a - delta_a / 2
        a_max = best_a + delta_a / 2

        # Calculate the distance
        distance_ = np.sqrt(size_v_**2 + size_h_**2)
        min_distance = np.min(distance_)

        axial_distance = a[np.argmin(size_h_)] - a[np.argmin(size_v_)]
        # Check if the distance condition is met
        if axial_distance <= 1e-11 and axial_distance > 1e-15:
            print(f" attempt :{attempt}")
            break

        attempt += 1  # Increment attempt counter

    # Display results
    print(f"  Optimal 'a': {best_a}, distance: {axial_distance}")
    print(f"  size_v_param: {size_v_param}, size_h_param: {size_h_param}\n")
    print(f"       v_shift: {a[np.argmin(size_v_)]},      h_shift: {a[np.argmin(size_h_)]}\n")
    print(f"  astigmatism: {initial_params[1]}")

    if attempt == max_attempts:
        print("Warning: Maximum attempts reached. Returning current best result.")
    if option_param == 'FoC':
        print('params = ',initial_params[0:1])
        vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle = plot_result_debug(initial_params, 'test',source_shift=source_shift0,option_tilt = False)
        return detcenter
    if option_legendre:
        innerproducts, orders, pvs = plot_result_debug(initial_params, 'ray_wave',option_legendre=True)
        return innerproducts, orders, pvs
    if option:
        print('params = ',initial_params[0:26])
        if option_AKB:
            if option_disp == 'ray_wave':
                b = plot_result_debug(initial_params,option_disp)
                return b
            # vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle = plot_result_debug(initial_params,option_disp)
            if option_mode == 'FoC':
                vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle = plot_result_debug(initial_params, option_disp,source_shift=source_shift0,option_tilt = False)
            else:
                vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle = plot_result_debug(initial_params, option_disp)

        else:
            if option_disp == 'ray_wave':
                pv = KB_debug(initial_params, na_ratio_h, na_ratio_v, option_disp)
                return pv
            vmirr_hyp, hmirr_hyp, detcenter, angle = KB_debug(initial_params,na_ratio_h,na_ratio_v, option_disp)
        
    if option_param == 'V':
        # return np.min(size_v_)
        return size_v_param, size_h_param,initial_params
    elif option_param == 'H':
        # return np.min(size_h_)
        return size_v_param, size_h_param,initial_params
    elif option_param == 'D':
        return np.min(distance_)
    return size_v_param, size_h_param,initial_params

def auto_focus_sep(initial_params0,adj_param1,adj_param2,la,ua,option='none',option_eval=None):
    if option =='abrr':
        size_v_param, size_h_param, initial_params = auto_focus_NA(50, initial_params0,1,1, False,'')
        if option_AKB:
            c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12 = plot_result_debug(initial_params,'sep')
        else:
            c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12 = KB_debug(initial_params,1,1, 'sep')
        # c1,c2,c3,c4,c5,c6 = KB_debug(initial_params,na_ratio_h,na_ratio_v, 'sep_direct')
        focus_v0, focus_h0, pos_v0, pos_h0, std_v0, std_h0, focus_v0_l, focus_h0_l, focus_v0_u, focus_h0_u, focus_std_obl1, focus_std_obl2 = c1,c2,c3[:,0,:],c4[:,0,:],c5,c6,c7,c8,c9,c10,c11,c12

        focus_v0_edge = (focus_v0[0] - np.mean(focus_v0) + focus_v0[2] - np.mean(focus_v0))/2.
        focus_h0_edge = (focus_h0[0] - np.mean(focus_h0) + focus_h0[2] - np.mean(focus_h0))/2.
        coma_v0 = focus_v0_l - focus_v0_u
        coma_h0 = focus_h0_l - focus_h0_u
        coma_v0_edge = (coma_v0[0] - np.mean(coma_v0) + coma_v0[2] - np.mean(coma_v0))/2.
        coma_h0_edge = (coma_h0[0] - np.mean(coma_h0) + coma_h0[2] - np.mean(coma_h0))/2.

        focus_len_v0 = focus_v0[0] - np.mean(focus_v0) - (focus_v0[2] - np.mean(focus_v0))
        coma_h_v0 = ((focus_v0_l[1] - focus_v0_u[1]) + (focus_v0_l[0] - focus_v0_u[0] + focus_v0_l[2] - focus_v0_u[2])/2)/2
        oblique_ast = focus_std_obl1 - focus_std_obl2

        focus_len_h0 = focus_h0[0] - np.mean(focus_h0) - (focus_h0[2] - np.mean(focus_h0))
        coma_v_h0 = ((focus_h0_l[1] - focus_h0_u[1]) + (focus_h0_l[0] - focus_h0_u[0] + focus_h0_l[2] - focus_h0_u[2])/2)/2

        focus_valance_cnt_edg_v0 = focus_v0[1] - np.mean(focus_v0) - focus_v0_edge
        focus_valance_cnt_edg_h0 = focus_h0[1] - np.mean(focus_h0) - focus_h0_edge



        coma_valance_v0 = coma_v0[0] - np.mean(coma_v0) - (coma_v0[2] - np.mean(coma_v0))
        coma_valance_h0 = coma_h0[0] - np.mean(coma_h0) - (coma_h0[2] - np.mean(coma_h0))

        coma_valance_cnt_edg_v0 = coma_v0[1] - np.mean(coma_v0) - coma_v0_edge
        coma_valance_cnt_edg_h0 = coma_h0[1] - np.mean(coma_h0) - coma_h0_edge
        # a0 = oblique_ast
        # a1 = coma_v_h0
        # a2 = coma_h_v0
        # a3 = focus_len_h0
        # a4 = focus_len_v0
        # a5 = (focus_valance_cnt_edg_v0 + focus_valance_cnt_edg_h0)/2
        # abrr = np.array([a0,a1,a2,a3,a4,a5])
        a0 = oblique_ast
        a1 = coma_v_h0
        a2 = coma_h_v0
        a3 = focus_len_h0
        a4 = focus_len_v0
        # a6 = focus_valance_cnt_edg_h0
        # a7 = focus_valance_cnt_edg_v0
        a6 = coma_valance_cnt_edg_h0
        a7 = coma_valance_cnt_edg_v0
        a8 = coma_valance_h0
        a9 = coma_valance_v0
        # a5 = (focus_valance_cnt_edg_v0 + focus_valance_cnt_edg_h0)/2
        # abrr = np.array([a0,a1,a2,a3,a4,a5])
        if option_eval == '9':
            abrr = np.array([a0,a1,a2,a3,a4,a6,a7,a8,a9])
            return abrr
        if option_eval == '7':
            abrr = np.array([a0,a3,a4,a6,a7,a8,a9])
            return abrr
        if option_eval == '7coma':
            abrr = np.array([a0,a1,a2,a3,a4,a8,a9])
            return abrr
        if option_eval == '5':
            abrr = np.array([a0,a3,a4,a8,a9])
            return abrr
        if option_eval == '5coma':
            abrr = np.array([a0,a1,a2,a3,a4])
            return abrr
        if option_eval == '2':
            abrr = np.array([a1,a2])
            return abrr
        if option_eval == '3':
            abrr = np.array([a0,a3,a4])
            return abrr
        if option_eval == 'KB':
            abrr = np.array([a0,a2,a4])
            return abrr
        if option_AKB:
            abrr = np.array([a0,a1,a2,a3,a4,a6,a7,a8,a9])
        else:
            abrr = np.array([a0,a2,a4])
        return abrr

    # 初期のパラメータ設定
    # initial_a_min = -0.3   # a の最初の最小値
    # initial_a_max = 0.3   # a の最初の最大値
    # shrink_factor = 0.1      # 各ループで範囲を縮小する割合
    # num_adj_astg = 50        # 各範囲でのステップ数
    # max_loops = 12            # ループ回数 (収束条件により調整)
    na_ratio_h = 1
    na_ratio_v = 1
    # 初期の範囲を設定
    # a_min = initial_a_min
    # a_max = initial_a_max

    initial_params = initial_params0.copy()
    # option_param = 'V'
    num_adj_param = 5
    # adj_param1 = 9
    # adj_param2 = 9
    a_param = np.linspace(la, ua, num_adj_param) + (initial_params[adj_param1] + initial_params[adj_param2])/2;
    size_v_param = np.linspace(0.0005, 0.0015, num_adj_param);
    size_h_param = np.linspace(0.0005, 0.0015, num_adj_param);
    astig = np.linspace(0.0005, 0.0015, num_adj_param);
    focus_v0 = np.zeros((num_adj_param,3))
    focus_h0 = np.zeros((num_adj_param,3))
    focus_v0_l = np.zeros((num_adj_param,3))
    focus_h0_l = np.zeros((num_adj_param,3))
    focus_v0_u = np.zeros((num_adj_param,3))
    focus_h0_u = np.zeros((num_adj_param,3))
    std_v0 = np.zeros((num_adj_param,3))
    std_h0 = np.zeros((num_adj_param,3))
    pos_v0 = np.zeros((num_adj_param,3,3))
    pos_h0 = np.zeros((num_adj_param,3,3))
    focus_std_obl1 = np.zeros((num_adj_param))
    focus_std_obl2 = np.zeros((num_adj_param))

    for j in range(len(a_param)):
        initial_params[adj_param1] = a_param[j]
        initial_params[adj_param2] = a_param[j]
        # initial_params[1] = -0.0012
        size_v_param[j], size_h_param[j], initial_params = auto_focus_NA(50, initial_params,1,1, False,'')# auto_focus_NA(num_adj_astg,initial_params,na_ratio_h,na_ratio_v,option):

        if option_AKB:
            c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12 = plot_result_debug(initial_params,'sep')
        else:
            c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12 = KB_debug(initial_params,na_ratio_h,na_ratio_v, 'sep')
        # c1,c2,c3,c4,c5,c6 = KB_debug(initial_params,na_ratio_h,na_ratio_v, 'sep_direct')
        focus_v0[j,:], focus_h0[j,:], pos_v0[j,:,:], pos_h0[j,:,:], std_v0[j,:], std_h0[j,:], focus_v0_l[j,:], focus_h0_l[j,:], focus_v0_u[j,:], focus_h0_u[j,:], focus_std_obl1[j], focus_std_obl2[j] = c1,c2,c3[:,0,:],c4[:,0,:],c5,c6,c7,c8,c9,c10,c11,c12
        astig[j] = initial_params[1]

    focus_v0_edge = (focus_v0[:,0] - np.mean(focus_v0,axis=1) + focus_v0[:,2] - np.mean(focus_v0,axis=1))/2.
    focus_h0_edge = (focus_h0[:,0] - np.mean(focus_h0,axis=1) + focus_h0[:,2] - np.mean(focus_h0,axis=1))/2.
    coma_v0 = focus_v0_l - focus_v0_u
    coma_h0 = focus_h0_l - focus_h0_u
    coma_v0_edge = (coma_v0[:,0] - np.mean(coma_v0,axis=1) + coma_v0[:,2] - np.mean(coma_v0,axis=1))/2.
    coma_h0_edge = (coma_h0[:,0] - np.mean(coma_h0,axis=1) + coma_h0[:,2] - np.mean(coma_h0,axis=1))/2.


    print('minimize coma r',a_param[np.argmin(abs(coma_v0[:,0]))])
    print('minimize coma y',a_param[np.argmin(abs(coma_v0[:,1]))])
    print('minimize coma g',a_param[np.argmin(abs(coma_v0[:,2]))])

    print('axial focus distance @V aperture 0',a_param[np.argmin(abs(focus_v0[:,0] - focus_v0[:,2]))])
    print('axial focus distance @H aperture 0',a_param[np.argmin(abs(focus_h0[:,0] - focus_h0[:,2]))])
    print('focus size std @V aperture 0',a_param[np.argmin(abs(std_v0[:,0] - std_v0[:,2]))])
    print('focus size std @H aperture 0',a_param[np.argmin(abs(std_h0[:,0] - std_h0[:,2]))])


    focus_len_v0 = focus_v0[:,0] - np.mean(focus_v0,axis=1) - (focus_v0[:,2] - np.mean(focus_v0,axis=1))
    coma_h_v0 = ((coma_v0[:,1]) + (coma_v0[:,0] + coma_v0[:,2])/2)/2
    oblique_ast = focus_std_obl1 - focus_std_obl2

    focus_len_h0 = focus_h0[:,0] - np.mean(focus_h0,axis=1) - (focus_h0[:,2] - np.mean(focus_h0,axis=1))
    coma_v_h0 = ((coma_h0[:,1]) + (coma_h0[:,0] + coma_h0[:,2])/2)/2

    focus_valance_cnt_edg_v0 = focus_v0[:,1] - np.mean(focus_v0,axis=1) - focus_v0_edge
    focus_valance_cnt_edg_h0 = focus_h0[:,1] - np.mean(focus_h0,axis=1) - focus_h0_edge

    coma_valance_v0 = coma_v0[:,0] - np.mean(coma_v0,axis=1) - (coma_v0[:,2] - np.mean(coma_v0,axis=1))
    coma_valance_h0 = coma_h0[:,0] - np.mean(coma_h0,axis=1) - (coma_h0[:,2] - np.mean(coma_h0,axis=1))

    coma_valance_cnt_edg_v0 = coma_v0[:,1] - np.mean(coma_v0,axis=1) - coma_v0_edge
    coma_valance_cnt_edg_h0 = coma_h0[:,1] - np.mean(coma_h0,axis=1) - coma_h0_edge
    # if adj_param1 == 8:
    #     opt_arg = np.argmin(abs(focus_len_v0))
    #     opt_param = a_param[opt_arg]
    #     print(f'param {adj_param1} =',opt_param)
    #
    # if adj_param1 == 10:
    #     opt_arg = np.argmin(abs(coma_h_v0 ))
    #     opt_param = a_param[opt_arg]
    #     print(f'param {adj_param1} =',opt_param)
    #
    # if adj_param1 == 9:
    #     opt_arg = np.argmin(abs(oblique_ast))
    #     opt_param = a_param[opt_arg]
    #     print(f'param {adj_param1} =',opt_param)

    if adj_param1 == 8 or adj_param1 == 20:
        opt_arg = np.argmin(abs(focus_len_v0))
        opt_param = a_param[opt_arg]
        print('adjust focus_len_v0')
        print(f'param {adj_param1} =',opt_param)

    if adj_param1 == 10 or adj_param1 == 22:
        opt_arg = np.argmin(abs(coma_h_v0 ))
        opt_param = a_param[opt_arg]
        print('adjust coma_h_v0')
        print(f'param {adj_param1} =',opt_param)

    if adj_param1 == 16:
        opt_arg = np.argmin(abs(focus_len_h0))
        opt_param = a_param[opt_arg]
        print('adjust focus_len_h0')
        print(f'param {adj_param1} =',opt_param)

    if adj_param1 == 14:
        opt_arg = np.argmin(abs(coma_v_h0 ))
        opt_param = a_param[opt_arg]
        print('adjust coma_v_h0')
        print(f'param {adj_param1} =',opt_param)

    if adj_param1 == 15 or adj_param1 == 21:
        opt_arg = np.argmin(abs(oblique_ast))
        opt_param = a_param[opt_arg]
        print('adjust oblique_ast')
        print(f'param {adj_param1} =',opt_param)
    if adj_param1 == 9:
        opt_arg = np.argmin(abs(focus_valance_cnt_edg_v0 + focus_valance_cnt_edg_h0))
        opt_param = a_param[opt_arg]
        print('adjust focus_len_cnt_edg')
        print(f'param {adj_param1} =',opt_param)

    print('dif focus length v0 (r-g) =',focus_len_v0[opt_arg])
    print('coma h v0 (y+(r+g)/2)/2 =',coma_h_v0[opt_arg])
    print('dif focus length h0 (r-g) =',focus_len_h0[opt_arg])
    print('coma v h0 (y+(r+g)/2)/2 =',coma_v_h0[opt_arg])
    print('oblique_ast = ', oblique_ast[opt_arg])
    print('focus_valance_cnt_edg_v0 = ', focus_valance_cnt_edg_v0[opt_arg])
    print('focus_valance_cnt_edg_h0 = ', focus_valance_cnt_edg_h0[opt_arg])
    initial_params0[adj_param1] = opt_param

    def linearfit(a, b):
        model = LinearRegression()
        model.fit(a, b)

        # Get regression coefficients
        slope = model.coef_[0]   # Slope
        intercept = model.intercept_  # Intercept

        # Calculate R^2 score
        r2 = r2_score(b, model.predict(a))

        # Output the result
        if r2 < 0.9:
            print(f"R^2 score: {r2:.4f}")
            print(f"No correlation")
            return 0., np.mean(b), r2
        print(f"Regression equation: b = {slope:.9f} * a + {intercept:.9f}")
        print(f"R^2 score: {r2:.4f}")
        return slope, intercept, r2

    if option == 'matrix':
        a = a_param.reshape(-1, 1)
        m0, intercept0, r20 = linearfit(a, oblique_ast)
        m1, intercept1, r21 = linearfit(a, coma_v_h0)
        m2, intercept2, r22 = linearfit(a, coma_h_v0)
        m3, intercept3, r23 = linearfit(a, focus_len_h0)
        m4, intercept4, r24 = linearfit(a, focus_len_v0)
        # m6, intercept6, r26 = linearfit(a, focus_valance_cnt_edg_h0)
        # m7, intercept7, r27 = linearfit(a, focus_valance_cnt_edg_v0)
        m6, intercept6, r26 = linearfit(a, coma_valance_cnt_edg_h0)
        m7, intercept7, r27 = linearfit(a, coma_valance_cnt_edg_v0)
        m8, intercept8, r28 = linearfit(a, coma_valance_h0)
        m9, intercept9, r29 = linearfit(a, coma_valance_v0)

        if option_AKB:
            M = np.array([m0, m1, m2, m3, m4, m6, m7, m8, m9])
        else:
            M = np.array([m0, m2, m4])
        intercepts = np.array([intercept0, intercept1, intercept2, intercept3, intercept4, intercept6, intercept7, intercept8, intercept9])
        r2_scores = np.array([r20, r21, r22, r23, r24, r26, r27, r28, r29])

        fig, axs = plt.subplots(3, 3, figsize=(20, 20))

        # Plot oblique_ast with linear fit line, equation, and R^2
        axs[0, 0].plot(a_param, oblique_ast, c='g', label='Data')
        axs[0, 0].plot(a_param, m0 * a_param + intercept0, c='r',
                       label=f'Fit: b = {m0:.9f} * a + {intercept0:.9f}\nR^2 = {r20:.4f}')
        axs[0, 0].set_title('oblique_ast')
        axs[0, 0].legend()

        axs[0, 1].plot(a_param, coma_v_h0, c='g', label='Data')
        axs[0, 1].plot(a_param, m1 * a_param + intercept1, c='r',
                       label=f'Fit: b = {m1:.9f} * a + {intercept1:.9f}\nR^2 = {r21:.4f}')
        axs[0, 1].set_title('coma_v_h0')
        axs[0, 1].legend()

        axs[0, 2].plot(a_param, coma_h_v0, c='g', label='Data')
        axs[0, 2].plot(a_param, m2 * a_param + intercept2, c='r',
                       label=f'Fit: b = {m2:.9f} * a + {intercept2:.9f}\nR^2 = {r22:.4f}')
        axs[0, 2].set_title('coma_h_v0')
        axs[0, 2].legend()

        axs[1, 0].plot(a_param, focus_len_h0, c='g', label='Data')
        axs[1, 0].plot(a_param, m3 * a_param + intercept3, c='r',
                       label=f'Fit: b = {m3:.9f} * a + {intercept3:.9f}\nR^2 = {r23:.4f}')
        axs[1, 0].set_title('focus_len_h0')
        axs[1, 0].legend()

        axs[1, 1].plot(a_param, focus_len_v0, c='g', label='Data')
        axs[1, 1].plot(a_param, m4 * a_param + intercept4, c='r',
                       label=f'Fit: b = {m4:.9f} * a + {intercept4:.9f}\nR^2 = {r24:.4f}')
        axs[1, 1].set_title('focus_len_v0')
        axs[1, 1].legend()

        # axs[1, 2].plot(a_param, focus_valance_cnt_edg_h0, c='g', label='Data')
        # axs[1, 2].plot(a_param, m6 * a_param + intercept6, c='r',
        #                label=f'Fit: b = {m6:.9f} * a + {intercept6:.9f}\nR^2 = {r26:.4f}')
        # axs[1, 2].set_title('focus_valance_cnt_edg_h0')
        # axs[1, 2].legend()
        #
        # axs[2, 0].plot(a_param, focus_valance_cnt_edg_v0, c='g', label='Data')
        # axs[2, 0].plot(a_param, m7 * a_param + intercept7, c='r',
        #                label=f'Fit: b = {m7:.9f} * a + {intercept7:.9f}\nR^2 = {r27:.4f}')
        # axs[2, 0].set_title('focus_valance_cnt_edg_v0')
        # axs[2, 0].legend()

        axs[1, 2].plot(a_param, coma_valance_cnt_edg_h0, c='g', label='Data')
        axs[1, 2].plot(a_param, m6 * a_param + intercept6, c='r',
                       label=f'Fit: b = {m6:.9f} * a + {intercept6:.9f}\nR^2 = {r26:.4f}')
        axs[1, 2].set_title('coma_valance_cnt_edg_h0')
        axs[1, 2].legend()

        axs[2, 0].plot(a_param, coma_valance_cnt_edg_v0, c='g', label='Data')
        axs[2, 0].plot(a_param, m7 * a_param + intercept7, c='r',
                       label=f'Fit: b = {m7:.9f} * a + {intercept7:.9f}\nR^2 = {r27:.4f}')
        axs[2, 0].set_title('coma_valance_cnt_edg_v0')
        axs[2, 0].legend()

        axs[2, 1].plot(a_param, coma_valance_h0, c='g', label='Data')
        axs[2, 1].plot(a_param, m8 * a_param + intercept8, c='r',
                       label=f'Fit: b = {m8:.9f} * a + {intercept8:.9f}\nR^2 = {r28:.4f}')
        axs[2, 1].set_title('coma_valance_h0')
        axs[2, 1].legend()

        axs[2, 2].plot(a_param, coma_valance_v0, c='g', label='Data')
        axs[2, 2].plot(a_param, m9 * a_param + intercept9, c='r',
                       label=f'Fit: b = {m9:.9f} * a + {intercept9:.9f}\nR^2 = {r29:.4f}')
        axs[2, 2].set_title('coma_valance_v0')
        axs[2, 2].legend()
        # 各データの平均値と範囲を計算
        means = np.array([np.mean(oblique_ast), np.mean(coma_v_h0), np.mean(coma_h_v0),
                          np.mean(focus_len_h0), np.mean(focus_len_v0),
                          np.mean(coma_valance_cnt_edg_h0), np.mean(coma_valance_cnt_edg_v0),
                          np.mean(coma_valance_h0), np.mean(coma_valance_v0)])

        ranges = np.array([np.max(oblique_ast) - np.min(oblique_ast),
                           np.max(coma_v_h0) - np.min(coma_v_h0),
                           np.max(coma_h_v0) - np.min(coma_h_v0),
                           np.max(focus_len_h0) - np.min(focus_len_h0),
                           np.max(focus_len_v0) - np.min(focus_len_v0),
                           np.max(coma_valance_cnt_edg_h0) - np.min(coma_valance_cnt_edg_h0),
                           np.max(coma_valance_cnt_edg_v0) - np.min(coma_valance_cnt_edg_v0),
                           np.max(coma_valance_h0) - np.min(coma_valance_h0),
                           np.max(coma_valance_v0) - np.min(coma_valance_v0)])

        # 最大差を計算
        max_range = np.max(ranges)

        # 各サブプロットで ylim を平均値 ± 最大差/2 に設定
        for ax, mean in zip(axs.flat, means):
            ax.set_ylim(mean - max_range / 2, mean + max_range / 2)
        fig.suptitle(f'param_{adj_param1}')
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.tight_layout(pad=8.0)  # Adjust padding
        # plt.savefig(f'contribution_plots_prm{adj_param1}.png', dpi=300)
        # フォントサイズを指定
        title_fontsize = 18
        label_fontsize = 14
        legend_fontsize = 12
        suptitle_fontsize = 24

        # 各サブプロットに対してフォントサイズを指定
        for ax, title in zip(axs.flat, ['oblique_ast', 'coma_v_h0', 'coma_h_v0',
                                        'focus_len_h0', 'focus_len_v0',
                                        'coma_valance_cnt_edg_h0', 'coma_valance_cnt_edg_v0',
                                        'coma_valance_h0', 'coma_valance_v0']):
            ax.set_title(title, fontsize=title_fontsize)
            ax.legend(fontsize=legend_fontsize)
            ax.set_xlabel('a_param', fontsize=label_fontsize)
            ax.set_ylabel('Value', fontsize=label_fontsize)

        # suptitle のフォントサイズを指定
        fig.suptitle(f'param_{adj_param1}', fontsize=suptitle_fontsize)
        plt.savefig(f'contribution_plots_prm{adj_param1}_2.png', dpi=600)
        plt.close()

        fig2, axes2 = plt.subplots(2, 1)
        axes2[0].plot(a_param, size_v_param)
        axes2[1].plot(a_param, size_h_param)
        axes2[0].set_title('size_v')
        axes2[1].set_title('size_h')
        fig.suptitle(f'param_{adj_param1}', fontsize=suptitle_fontsize)
        plt.savefig(f'size_plots_prm{adj_param1}_2.png', dpi=600)
        plt.close()

        if option_eval == '7':
            M = np.array([m0,m3,m4,m6,m7,m8,m9])
            return M
        if option_eval == '9':
            M = np.array([m0,m1,m2,m3,m4,m6,m7,m8,m9])
            return M
        if option_eval == '5':
            M = np.array([m0,m3,m4,m8,m9])
            return M
        if option_eval == '2':
            M = np.array([[m1,m2],[intercept1,intercept2]])
            return M
        if option_eval == '3':
            M = np.array([[m0,m3,m4],[intercept0,intercept3,intercept4]])
            return M
        if option_eval == 'KB':
            M = np.array([m0,m2,m4])
            return M
        if option_eval == '3_intercept':
            M = np.array([[m0,m2,m4],[intercept0,intercept2,intercept4]])
            return M
        if option_eval == '5coma':
            M = np.array([m0,m1,m2,m3,m4])
            return M
        if option_eval == '7coma':
            M = np.array([m0,m1,m2,m3,m4,m8,m9])
            return M
        if option_eval == 'MinimizeH':
            minarg = np.argmin(size_h_param)
            return a_param[minarg]
        return M
    # vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle = auto_focus_NA(50, initial_params0,1,1, True,'')
    return

def load_npz_data(filename):
    """
    指定したファイル名の npz ファイルを読み込み、データを返す関数。

    Parameters:
    filename (str): 読み込みたい npz ファイルの名前（パスを含む）。

    Returns:
    numpy.ndarray: 読み込んだデータの配列。ファイルが存在しない場合は None を返す。
    """
    if os.path.exists(filename):
        with np.load(filename) as data:  # コンテキストマネージャで開くと自動的に閉じる
            return data['data']  # 保存時のキーワード 'data' でデータを取得
    else:
        print(f"{filename} が存在しません。")
        return None
def downsample_array_3_n(array_3_n, ray_num_V, ray_num_H, downsample_h, downsample_v):
    x = np.reshape(array_3_n[0,:],(ray_num_V, ray_num_H))
    y = np.reshape(array_3_n[1,:],(ray_num_V, ray_num_H))
    z = np.reshape(array_3_n[2,:],(ray_num_V, ray_num_H))
    print('x_origin.shape',x.shape)
    count = downsample_h // 2  # # 0→0回、2→1回、4→2回
    for i in range(count):
        x = x[:,::2]
        y = y[:,::2]
        z = z[:,::2]

    count = downsample_v // 2  # # 0→0回、2→1回、4→2回
    for i in range(count):
        x = x[::2,:]
        y = y[::2,:]
        z = z[::2,:]
    size_v = x.shape[0]
    size_h = x.shape[1]
    print('x_downsampled.shape',x.shape)
    array_3_n_downsampled = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    return array_3_n_downsampled, size_v, size_h

def downsample_array_any_n(array_m_n, ray_num_V, ray_num_H, downsample_h, downsample_v):
    """
    m行n列のデータ（点群など）を、任意の行数に対応してダウンサンプリング

    Parameters
    ----------
    array_m_n : np.ndarray
        m行n列の点群データ（mは3次元に限らない）
    ray_num_V : int
        縦方向の元の点数
    ray_num_H : int
        横方向の元の点数
    downsample_h : int
        横方向の間引き量（2なら半分、4なら1/4など）
    downsample_v : int
        縦方向の間引き量（2なら半分、4なら1/4など）

    Returns
    -------
    array_m_n_downsampled : np.ndarray
        間引き後の点群データ（m行, ダウンサンプリング後の点数列）
    size_v : int
        間引き後の縦方向の点数
    size_h : int
        間引き後の横方向の点数
    """
    if array_m_n.ndim == 1:
        array_m_n = array_m_n.reshape(1, -1)  # 1次元なら強制1行n列
    m, n = array_m_n.shape

    # 各成分を正方行列にreshape
    component_list = []
    for i in range(m):
        component = np.reshape(array_m_n[i, :], (ray_num_V, ray_num_H))
        component_list.append(component)

    print('Original shape:', component_list[0].shape)

    # 横方向のダウンサンプリング
    count = downsample_h // 2
    for _ in range(count):
        for j in range(m):
            component_list[j] = component_list[j][:, ::2]

    # 縦方向のダウンサンプリング
    count = downsample_v // 2
    for _ in range(count):
        for j in range(m):
            component_list[j] = component_list[j][::2, :]

    size_v = component_list[0].shape[0]
    size_h = component_list[0].shape[1]

    print('Downsampled shape:', component_list[0].shape)

    # 各成分をフラットに戻して結合
    array_m_n_downsampled = np.vstack([c.flatten() for c in component_list])

    return array_m_n_downsampled, size_v, size_h

def calc_dS(points,ray_num_V, ray_num_H):
    grid_points = points.reshape(3, ray_num_V, ray_num_H)
    """
    grid_points : (3, l, k) の3次元座標
    """
    l, k = grid_points.shape[1:]

    # 面積格納
    dS = np.zeros((l, k))

    # 内部点の面積計算
    for i in range(1, l-1):
        for j in range(1, k-1):
            p = grid_points[:, i, j]

            p_right = grid_points[:, i, j+1]
            p_left = grid_points[:, i, j-1]
            p_up = grid_points[:, i-1, j]
            p_down = grid_points[:, i+1, j]

            triangles = [
                (p, p_right, p_up),
                (p, p_up, p_left),
                (p, p_left, p_down),
                (p, p_down, p_right)
            ]

            area_sum = 0.0

            for v0, v1, v2 in triangles:
                e1 = v1 - v0
                e2 = v2 - v0
                area = np.linalg.norm(np.cross(e1, e2)) / 2
                area_sum += area

            dS[i, j] = area_sum

    # 縁の補間処理（縁以外の最近傍をコピー）
    for i in range(l):
        for j in range(k):
            if i == 0:           # 上端
                dS[i, j] = dS[1, j]
            elif i == l-1:       # 下端
                dS[i, j] = dS[l-2, j]
            elif j == 0:         # 左端
                dS[i, j] = dS[i, 1]
            elif j == k-1:       # 右端
                dS[i, j] = dS[i, k-2]

    # 4隅の補間処理（これは個別指定でOK）
    dS[0, 0]     = dS[1, 1]      # 左上
    dS[0, k-1]   = dS[1, k-2]    # 右上
    dS[l-1, 0]   = dS[l-2, 1]    # 左下
    dS[l-1, k-1] = dS[l-2, k-2]  # 右下

    return dS

def saveWaveData(initial_params, ysize = 1e-6, zsize = 1e-6):
    if option_AKB:
        # plot_result_debug(initial_params,'ray_wave')
        if np.abs(defocusForWave) > 1e-9:
            source, vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, detcenter2, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vmirr2_norm, hmirr2_norm, vec0to1, vec1to2, vec2to3, vec3to4 = plot_result_debug(initial_params,'wave')
        else:
            source, vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vmirr2_norm, hmirr2_norm, vec0to1, vec1to2, vec2to3, vec3to4 = plot_result_debug(initial_params,'wave')
    else:
        # KB_debug(initial_params,1,1,'ray_wave')
        if np.abs(defocusForWave) > 1e-9:
            source, vmirr_hyp, hmirr_hyp, detcenter, detcenter2, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vec0to1, vec1to2 = KB_debug(initial_params,1,1,'wave')
        else:
            source, vmirr_hyp, hmirr_hyp, detcenter, ray_num_H, ray_num_V, vmirr_norm, hmirr_norm, vec0to1, vec1to2 = KB_debug(initial_params,1,1,'wave')

    if ray_num_H%2 == 1:
        vmirr_hyp, size_v1, size_h1 = downsample_array_3_n(vmirr_hyp, ray_num_V, ray_num_H, downsample_h1, downsample_v1)
        hmirr_hyp, size_v2, size_h2 = downsample_array_3_n(hmirr_hyp, ray_num_V, ray_num_H, downsample_h2, downsample_v2)
        detcenter, size_v_f, size_h_f = downsample_array_3_n(detcenter, ray_num_V, ray_num_H, downsample_h_f, downsample_v_f)
        if option_AKB:
            vmirr_ell, size_v1, size_h1 = downsample_array_3_n(vmirr_ell, ray_num_V, ray_num_H, downsample_h1, downsample_v1)
            hmirr_ell, size_v2, size_h2 = downsample_array_3_n(hmirr_ell, ray_num_V, ray_num_H, downsample_h2, downsample_v2)
        if np.abs(defocusForWave) > 1e-9:
            detcenter2, size_v_f, size_h_f = downsample_array_3_n(detcenter2, ray_num_V, ray_num_H, downsample_h_f, downsample_v_f)
        # vmirr_norm, _, _ = downsample_array_3_n(vmirr_norm, ray_num_V, ray_num_H, downsample_h1, downsample_v1)
        # hmirr_norm, _, _ = downsample_array_3_n(hmirr_norm, ray_num_V, ray_num_H, downsample_h2, downsample_v2)
        # vec0to1, _, _ = downsample_array_3_n(vec0to1, ray_num_V, ray_num_H, downsample_h1, downsample_v1)
        # vec1to2, _, _ = downsample_array_3_n(vec1to2, ray_num_V, ray_num_H, downsample_h2, downsample_v2)
        # del vmirr_norm, hmirr_norm, vec0to1, vec1to2
    else:
        print('without downsampling')

    print('source',source)
    # .npy ファイルに保存
    # フォルダ名として使用する現在時刻の文字列を取得
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory_name = f"output_{timestamp}"

    # 新しいフォルダを作成
    os.makedirs(directory_name, exist_ok=True)

    np.save(os.path.join(directory_name, 'points_source.npy'), source[:,0])
    # データを新規フォルダ内に保存
    print('source[:,0]',source[:,0].dtype)
    print('source[:,0]',source[:,0].shape)

    dS1 = calc_dS(vmirr_hyp,size_v1, size_h1)
    vmirr_hyp = np.vstack((vmirr_hyp, dS1.flatten()))

    np.save(os.path.join(directory_name, 'points_M1.npy'), vmirr_hyp)

    print('vmirr_hyp',vmirr_hyp.dtype)
    print('vmirr_hyp',vmirr_hyp.shape)
    print('xFOV vmirr_hyp',[np.min(vmirr_hyp[0,:]),np.max(vmirr_hyp[0,:]),np.max(vmirr_hyp[0,:])-np.min(vmirr_hyp[0,:])])
    print('yFOV vmirr_hyp',[np.min(vmirr_hyp[1,:]),np.max(vmirr_hyp[1,:]),np.max(vmirr_hyp[1,:])-np.min(vmirr_hyp[1,:])])
    print('zFOV vmirr_hyp',[np.min(vmirr_hyp[2,:]),np.max(vmirr_hyp[2,:]),np.max(vmirr_hyp[2,:])-np.min(vmirr_hyp[2,:])])

    dS2 = calc_dS(hmirr_hyp,size_v2, size_h2)
    hmirr_hyp = np.vstack((hmirr_hyp, dS2.flatten()))

    np.save(os.path.join(directory_name, 'points_M2.npy'), hmirr_hyp)

    print('hmirr_hyp',hmirr_hyp.dtype)
    print('hmirr_hyp',hmirr_hyp.shape)
    print('xFOV hmirr_hyp',[np.min(hmirr_hyp[0,:]),np.max(hmirr_hyp[0,:]),np.max(hmirr_hyp[0,:])-np.min(hmirr_hyp[0,:])])
    print('yFOV hmirr_hyp',[np.min(hmirr_hyp[1,:]),np.max(hmirr_hyp[1,:]),np.max(hmirr_hyp[1,:])-np.min(hmirr_hyp[1,:])])
    print('zFOV hmirr_hyp',[np.min(hmirr_hyp[2,:]),np.max(hmirr_hyp[2,:]),np.max(hmirr_hyp[2,:])-np.min(hmirr_hyp[2,:])])

    if option_AKB:
        dS3 = calc_dS(vmirr_ell,size_v1, size_h1)
        vmirr_ell = np.vstack((vmirr_ell, dS3.flatten()))

        np.save(os.path.join(directory_name, 'points_M3.npy'), vmirr_ell)

        print('vmirr_ell',vmirr_ell.dtype)
        print('vmirr_ell',vmirr_ell.shape)

        dS4 = calc_dS(hmirr_ell,size_v2, size_h2)
        hmirr_ell = np.vstack((hmirr_ell, dS4.flatten()))

        np.save(os.path.join(directory_name, 'points_M4.npy'), hmirr_ell)

        print('hmirr_ell',hmirr_ell.dtype)
        print('hmirr_ell',hmirr_ell.shape)

    y = detcenter[1,:]
    z = detcenter[2,:]
    print('ysize',np.max(y) - np.min(y))
    print('zsize',np.max(z) - np.min(z))
    print('pixy',(np.max(y) - np.min(y))/(size_h_f-1))
    print('pixz',(np.max(z) - np.min(z))/(size_v_f-1))

    # y_grid = np.linspace(np.min(y)-ysize*5, np.max(y)+ysize*5, ray_num_H)
    # z_grid = np.linspace(np.min(z)-zsize*5, np.max(z)+zsize*5, ray_num_V)
    # y_grid = np.linspace(np.min(y)-ysize*25, np.max(y)+ysize*25, ray_num_H)
    # z_grid = np.linspace(np.min(z)-zsize*25, np.max(z)+zsize*25, ray_num_V)
    y_grid = np.linspace((np.min(y) + np.max(y))/2-ysize, (np.min(y) + np.max(y))/2+ysize, size_h_f)
    z_grid = np.linspace((np.min(z) + np.max(z))/2-zsize, (np.min(z) + np.max(z))/2+zsize, size_v_f)
    # y_grid = np.linspace(-0.005459918052256706, -0.005459016833730451, width)
    # z_grid = np.linspace(-0.03998099986156593, -0.039979093398728624, width)
    yy, zz = np.meshgrid(y_grid,z_grid)
    # plt.figure()
    # plt.scatter(vmirr_hyp[1,:],vmirr_hyp[2,:])
    # plt.show()
    print('pixy', y_grid[1] - y_grid[0])
    print('pixz', z_grid[1] - z_grid[0])

    y_flattened = yy.flatten()  # 元の y の形状に戻す
    z_flattened = zz.flatten()  # 元の z の形状に戻す
    x_flattened = np.full_like(y_flattened,fill_value=np.mean(detcenter[0, :]))
    print('yFOV',[np.min(y_flattened),np.max(y_flattened),np.max(y_flattened)-np.min(y_flattened)])
    print('zFOV',[np.min(z_flattened),np.max(z_flattened),np.max(z_flattened)-np.min(z_flattened)])
    print('np.std(detcenter[0, :])',np.std(detcenter[0, :]))
    # 新しい detcenter として、元の形 (3, N) にデータを再構成
    new_detcenter = np.vstack([x_flattened, y_flattened, z_flattened])
    print(new_detcenter.shape)
    print('new_detcenter',new_detcenter.dtype)
    np.save(os.path.join(directory_name, 'points_gridImage.npy'), new_detcenter)

    if np.abs(defocusForWave) > 1e-9:
        if option_HighNA:
            ysize = 2e-7 + defocusForWave*0.082*2
            zsize = 2e-7 + defocusForWave*0.082*2
        else:
            ysize = 2e-7 + defocusForWave*0.01*2
            zsize = 2e-7 + defocusForWave*0.01*2
        y2 = detcenter2[1,:]
        z2 = detcenter2[2,:]
        print('ysize',np.max(y2) - np.min(y2))
        print('zsize',np.max(z2) - np.min(z2))
        print('pixy',(np.max(y2) - np.min(y2))/(size_h_f-1))
        print('pixz',(np.max(z2) - np.min(z2))/(size_v_f-1))

        y_grid2 = np.linspace((np.min(y2) + np.max(y2))/2-ysize, (np.min(y2) + np.max(y2))/2+ysize, size_h_f)
        z_grid2 = np.linspace((np.min(z2) + np.max(z2))/2-zsize, (np.min(z2) + np.max(z2))/2+zsize, size_v_f)
        yy2, zz2 = np.meshgrid(y_grid2,z_grid2)
        print('pixy', y_grid2[1] - y_grid2[0])
        print('pixz', z_grid2[1] - z_grid2[0])

        y_flattened2 = yy2.flatten()  # 元の y の形状に戻す
        z_flattened2 = zz2.flatten()  # 元の z の形状に戻す
        x_flattened2 = np.full_like(y_flattened2,fill_value=np.mean(detcenter2[0, :]))
        print('yFOV2',[np.min(y_flattened2),np.max(y_flattened2),np.max(y_flattened2)-np.min(y_flattened2)])
        print('zFOV2',[np.min(z_flattened2),np.max(z_flattened2),np.max(z_flattened2)-np.min(z_flattened2)])
        print('np.std(detcenter2[0, :])',np.std(detcenter2[0, :]))
        # 新しい detcenter として、元の形 (3, N) にデータを再構成
        new_detcenter2 = np.vstack([x_flattened2, y_flattened2, z_flattened2])
        print(new_detcenter2.shape)
        print('new_detcenter2',new_detcenter2.dtype)
        np.save(os.path.join(directory_name, 'points_gridDefocus.npy'), new_detcenter2)
        print('setDefocus',defocusForWave)
        print('Defocus',np.mean(new_detcenter2[0,:]-new_detcenter[0,:]))
    # 計算条件を保存するテキストファイルのパスを設定
    conditions_file_path = os.path.join(directory_name, 'calculation_conditions.txt')

    # テキストファイルに変数の値や計算条件を書き込む
    with open(conditions_file_path, 'w') as file:
        file.write("Conditions\n")
        file.write("====================\n")
        file.write(f"time: {timestamp}\n")
        file.write(f"params 0-1: {initial_params[0:2]}\n")
        file.write(f"params 2-7: {initial_params[2:8]}\n")
        file.write(f"params 8-13: {initial_params[8:14]}\n")
        file.write(f"params 14-19: {initial_params[14:20]}\n")
        file.write(f"params 20-26: {initial_params[20:26]}\n")
        file.write(f"grid pitch_y: {y_grid[1] - y_grid[0]}\n")
        file.write(f"grid pitch_z: {z_grid[1] - z_grid[0]}\n")
        file.write(f"grid size_y: {np.max(y_grid) - np.min(y_grid)}\n")
        file.write(f"grid size_z: {np.max(z_grid) - np.min(z_grid)}\n")
        file.write(f"grid pix_y: {size_h_f}\n")
        file.write(f"grid pix_z: {size_v_f}\n")
        file.write(f"grid pix_H1: {size_h1}\n")
        file.write(f"grid pix_V1: {size_v1}\n")
        file.write(f"grid pix_H2: {size_h2}\n")
        file.write(f"grid pix_V2: {size_v2}\n")
        file.write(f"option_AKB: {option_AKB}\n")
        file.write(f"option_HighNA: {option_HighNA}\n")
        file.write(f"defocusForWave: {defocusForWave}\n")
        file.write(f"calc both mirrors?: {option_2mirror}\n")
        file.write(f"option_avrgsplt: {option_avrgsplt}\n")
        file.write("====================\n")

    # index = 0
    # cosi1 = np.sum(vec0to1 * vmirr_norm, axis=0)[index]
    # vmirr_hyp_here = vmirr_hyp[0:3,index]
    # vec1to2r = -hmirr_hyp[0:3,:] + vmirr_hyp_here[:,None]
    # vec1to2r = normalize_vector(vec1to2r)
    # cosr1 = np.sum(vec1to2r * vmirr_norm, axis=0)
    #
    # plt.figure()
    # scatter = plt.scatter(hmirr_hyp[1, :], hmirr_hyp[2, :],c=cosi1 + cosr1, cmap='jet')
    # plt.colorbar(scatter, label='cos')
    # # plt.axis('equal')
    # plt.show()
    #
    # index = -1
    # cosi1 = np.sum(vec0to1 * vmirr_norm, axis=0)[index]
    # vmirr_hyp_here = vmirr_hyp[0:3,index]
    # vec1to2r = -hmirr_hyp[0:3,:] + vmirr_hyp_here[:,None]
    # vec1to2r = normalize_vector(vec1to2r)
    # cosr1 = np.sum(vec1to2r * vmirr_norm, axis=0)
    #
    # plt.figure()
    # scatter = plt.scatter(hmirr_hyp[1, :], hmirr_hyp[2, :],c=cosi1 + cosr1, cmap='jet')
    # plt.colorbar(scatter, label='cos')
    # # plt.axis('equal')
    # plt.show()
    # plt.figure()
    # scatter = plt.scatter(vmirr_hyp[1, :], vmirr_hyp[2, :],c=dS1.flatten(), cmap='jet')
    # plt.colorbar(scatter, label='dS1')
    # # plt.axis('equal')
    # plt.show()
    #
    # plt.figure()
    # scatter = plt.scatter(vmirr_hyp[1, :], vmirr_hyp[2, :],c=cos1*dS1.flatten(), cmap='jet')
    # plt.colorbar(scatter, label='cos1 * dS1')
    # # plt.axis('equal')
    # plt.show()
    #
    # cos2 = np.sum(vec1to2 * hmirr_norm, axis=0)
    #
    # plt.figure()
    # scatter = plt.scatter(hmirr_hyp[1, :], hmirr_hyp[2, :],c=cos2, cmap='jet')
    # plt.colorbar(scatter, label='cos')
    # # plt.axis('equal')
    # plt.show()
    #
    # plt.figure()
    # scatter = plt.scatter(hmirr_hyp[1, :], hmirr_hyp[2, :],c=dS2.flatten(), cmap='jet')
    # plt.colorbar(scatter, label='dS2')
    # # plt.axis('equal')
    # plt.show()
    #
    # plt.figure()
    # scatter = plt.scatter(hmirr_hyp[1, :], hmirr_hyp[2, :],c=cos2*dS2.flatten(), cmap='jet')
    # plt.colorbar(scatter, label='cos2 * dS2')
    # # plt.axis('equal')
    # plt.show()

    # # # dS = calc_dS(vmirr_hyp,size_v1, size_h1)
    # # # dS_flatten = dS.flatten()
    # plt.figure()
    # scatter = plt.scatter(vmirr_hyp[1, :], vmirr_hyp[2, :],c=vmirr_hyp[3, :], cmap='jet')
    # plt.colorbar(scatter, label='ds')
    # plt.axis('equal')
    # plt.show()
    # plt.figure()
    # scatter = plt.scatter(hmirr_hyp[1, :], hmirr_hyp[2, :],c=np.arange(hmirr_hyp.shape[1]), cmap='jet')
    # plt.colorbar(scatter, label='ds')
    # plt.axis('equal')
    # plt.show()
    #
    # plt.figure()
    # scatter = plt.scatter(hmirr_hyp[1, :], hmirr_hyp[2, :],c=hmirr_hyp[3, :], cmap='jet')
    # plt.colorbar(scatter, label='ds')
    # plt.axis('equal')
    # plt.show()
    # #
    #
    #
    # dS = calc_dS(vmirr_hyp,size_v1, size_h1)
    # dS_flatten = dS.flatten()
    # plt.figure()
    # scatter = plt.scatter(vmirr_hyp[0, :], vmirr_hyp[1, :],c=dS_flatten, cmap='jet')
    # plt.colorbar(scatter, label='ds')
    # # plt.axis('equal')
    # plt.show()
    #
    # dS = calc_dS(hmirr_hyp,size_v2, size_h2)
    # dS_flatten = dS.flatten()
    # plt.figure()
    # scatter = plt.scatter(hmirr_hyp[0, :], hmirr_hyp[1, :],c=dS_flatten, cmap='jet')
    # plt.colorbar(scatter, label='ds')
    # # plt.axis('equal')
    # plt.show()

    # plt.figure()
    # sample_detcenter = detcenter.copy()
    # sample_DistError = DistError.copy()
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
    sys.exit()
    return

def calc_FoC(initial_params,range_h=[-5e-3,5e-3,15],range_v=[-5e-3,5e-3,15]):
    
    range_h = np.linspace(range_h[0], range_h[1], range_h[2])
    range_v = np.linspace(range_v[0], range_v[1], range_v[2])

    

    # objectpointX = np.zeros((len(range_h),len(range_v)))
    focuspointX = np.zeros((len(range_h),len(range_v)))
    focuspointY = np.zeros((len(range_h),len(range_v)))
    focuspointZ = np.zeros((len(range_h),len(range_v)))
    focussizeV = np.zeros((len(range_h),len(range_v)))
    focussizeH = np.zeros((len(range_h),len(range_v)))

    for i in range(len(range_v)):
        for j in range(len(range_h)):
            S0 = [0., range_h[j], range_v[i]]
            print('source_shift0',S0)
            detcenter = auto_focus_NA(50, initial_params,1,1, True,'FoC', option_mode='FoC',source_shift0=S0)

            # vmirr_hyp, hmirr_hyp, vmirr_ell, hmirr_ell, detcenter, angle = plot_result_debug(initial_params, 'test',source_shift=S0,option_tilt = False)

            focuspoint = np.mean(detcenter, axis=1)
            focuspointX[i,j] = focuspoint[0]
            focuspointY[i,j] = focuspoint[1]
            focuspointZ[i,j] = focuspoint[2]
            focussizeH[i,j] = np.max(detcenter[1,:]) - np.min(detcenter[1,:])
            focussizeV[i,j] = np.max(detcenter[2,:]) - np.min(detcenter[2,:])
    
    fig,axs = plt.subplots(2, 2)
    im00 = axs[0, 0].scatter(focuspointY,focuspointX,c=focussizeH, cmap='jet')
    axs[0, 0].set_title('focussizeH')
    axs[0, 0].set_ylabel('focuspointX')
    axs[0, 0].set_xlabel('focuspointY')
    fig.colorbar(im00, ax=axs[0, 0])
    im01 = axs[0, 1].scatter(focuspointZ,focuspointX,c=focussizeH, cmap='jet')
    axs[0, 1].set_title('focussizeH')
    axs[0, 1].set_ylabel('focuspointX')
    axs[0, 1].set_xlabel('focuspointZ')
    fig.colorbar(im01, ax=axs[0, 1])
    im10 = axs[1, 0].scatter(focuspointY,focuspointX,c=focussizeV, cmap='jet')
    axs[1, 0].set_title('focussizeV')
    axs[1, 0].set_ylabel('focuspointX')
    axs[1, 0].set_xlabel('focuspointY')
    fig.colorbar(im10, ax=axs[1, 0])
    im11 = axs[1, 1].scatter(focuspointZ,focuspointX,c=focussizeV, cmap='jet')
    axs[1, 1].set_title('focussizeV')
    axs[1, 1].set_ylabel('focuspointX')
    axs[1, 1].set_xlabel('focuspointZ')
    fig.colorbar(im11, ax=axs[1, 1])
    # plt.axis('equal')

    fig1, axs1 = plt.subplots(2,2)
    axs1[0,0].plot(range_h, focussizeH[int(len(range_v)/2),:], label='focussizeH')
    axs1[0,0].set_xlabel('range_h')
    axs1[0,0].set_ylabel('focussizeH')
    axs1[0,1].plot(range_v, focussizeV[:,int(len(range_h)/2)], label='focussizeV')
    axs1[0,1].set_xlabel('range_v')
    axs1[0,1].set_ylabel('focussizeV')
    axs1[1,0].plot(range_h, focussizeV[int(len(range_v)/2),:], label='focussizeV')
    axs1[1,0].set_xlabel('range_h')
    axs1[1,0].set_ylabel('focussizeV')
    axs1[1,1].plot(range_v, focussizeH[:,int(len(range_h)/2)], label='focussizeH')
    axs1[1,1].set_xlabel('range_v')
    axs1[1,1].set_ylabel('focussizeH')

    fig2, axs2 = plt.subplots(2,2)
    axs2[0,0].plot(focuspointY[int(len(range_v)/2),:],focuspointX[int(len(range_v)/2),:])
    axs2[0, 0].set_ylabel('focuspointY')
    axs2[0, 0].set_xlabel('focuspointX')
    
    axs2[0,1].plot(focuspointZ[int(len(range_v)/2),:],focuspointX[int(len(range_v)/2),:])
    axs2[0, 1].set_ylabel('focuspointZ')
    axs2[0, 1].set_xlabel('focuspointX')
    
    axs2[1,0].plot(focuspointY[:,int(len(range_h)/2)],focuspointX[:,int(len(range_h)/2)])
    axs2[1, 0].set_ylabel('focuspointY')
    axs2[1, 0].set_xlabel('focuspointX')
    
    axs2[1,1].plot(focuspointZ[:,int(len(range_h)/2)],focuspointX[:,int(len(range_h)/2)])
    axs2[1, 1].set_ylabel('focuspointZ')
    axs2[1, 1].set_xlabel('focuspointX')
    plt.show()        
    
    
    return

def Legendrealignment(initial_params, num_param, range_param,tuning=True):
    global option_mpmath
    # Legendre alignment
    innerproducts = []
    pvs = []
    for i in range(int(len(range_param))):
        initial_params1 = initial_params.copy()
        initial_params1[num_param] += range_param[i]
        option_mpmath = False
        if tuning:
            inner, order, pv = auto_focus_NA(50, initial_params1,1,1, True,'',option_disp='ray_wave',option_legendre=True)
        option_mpmath = True
        inner, order, pv = plot_result_debug(initial_params1, 'ray_wave',option_legendre=True)
        option_mpmath = False
        innerproducts.append(inner)
        pvs.append(pv)

    innerproducts = np.array(innerproducts)
    pvs = np.array(pvs)
    row = innerproducts.shape[0]
    col = innerproducts.shape[1]

    ### plot
    
    ### default color map
    colors = plt.cm.viridis(np.linspace(0, 1, col))
    for i in range(col):
        plt.figure()
        if False:
            plt.plot(range_param, innerproducts[:,i], label=f'legendre {order[i]}', color=colors[i])
            ### linear fit
            coeffs = np.polyfit(range_param, innerproducts[:,i], 1)
        else:
            plt.plot(range_param, pvs[:,i], label=f'legendre {order[i]}', color=colors[i])
            plt.ylim(-0.25, 0.25)
            ### linear fit
            coeffs = np.polyfit(range_param, pvs[:,i], 1)
        fit_line = np.polyval(coeffs, range_param)
        plt.plot(range_param, fit_line, '--', label=f'y={coeffs[0]:.2e}x + {coeffs[1]:.2e}', color=colors[i])
        plt.xlabel(f'Alignment{ num_param}')
        plt.ylabel('Inner Product')
        plt.title(f'Legendre{order[i]} Alignment')
        plt.legend()
        plt.savefig(os.path.join(directory_name, f'Legendre_alignment_{num_param}_{order[i]}.png'))
        plt.close()

    plt.figure()
    plt.plot(range_param, pvs[:,-1], label=f'legendre {order[i]}', color='k')
    plt.ylim(-0.25, 0.25)
    ### linear fit
    coeffs = np.polyfit(range_param, pvs[:,-1], 1)
    fit_line = np.polyval(coeffs, range_param)
    plt.plot(range_param, fit_line, '--', label=f'y={coeffs[0]:.6e}x + {coeffs[1]:.6e}', color='k')
    plt.xlabel(f'Alignment{ num_param}')
    plt.ylabel('Inner Product')
    plt.title(f'All Alignment')
    plt.legend()
    plt.savefig(os.path.join(directory_name, f'Legendre_alignment_{num_param}_all.png'))
    plt.close()


def Finetuning(initial_params):
    global option_mpmath
    # Legendre alignment
    num_param = 1 
    range_param = np.linspace(-1e-7, 1e-7, 5)  # 調整範囲を設定
    innerproducts = []
    pvs = []
    for i in range(int(len(range_param))):
        initial_params1 = initial_params.copy()
        initial_params1[num_param] += range_param[i]
        option_mpmath = True
        inner, order, pv = plot_result_debug(initial_params1, 'ray_wave',option_legendre=True)
        option_mpmath = False
        innerproducts.append(inner)
        pvs.append(pv)

    innerproducts = np.array(innerproducts)
    pvs = np.array(pvs)

    pv_ast = pvs[:,5] - pvs[:,3]
    a_ast, b_ast = np.polyfit(range_param, pv_ast, 1)
    initial_params[num_param] += -b_ast/a_ast
    print('Finetuning astigmatism',initial_params[num_param])

    plt.figure()
    plt.plot(range_param, pv_ast, label=f'legendre {order[i]}', color='k')
    ### linear fit
    coeffs = np.polyfit(range_param, pv_ast, 1)
    fit_line = np.polyval(coeffs, range_param)
    plt.plot(range_param, fit_line, '--', label=f'y={coeffs[0]:.6e}x + {coeffs[1]:.6e}', color='k')
    plt.xlabel(f'Alignment{ num_param}')
    plt.ylabel('Ast')
    plt.title(f'Ast Alignment')
    plt.legend()
    plt.savefig(os.path.join(directory_name, f'Legendre_alignment_{num_param}_ast.png'))
    plt.close()

    num_param = 0
    range_param = np.linspace(-1e-7, 1e-7, 5)  # 調整範囲を設定
    innerproducts = []
    pvs = []
    for i in range(int(len(range_param))):
        initial_params1 = initial_params.copy()
        initial_params1[num_param] += range_param[i]
        option_mpmath = True
        inner, order, pv = plot_result_debug(initial_params1, 'ray_wave',option_legendre=True)
        option_mpmath = False
        innerproducts.append(inner)
        pvs.append(pv)

    innerproducts = np.array(innerproducts)
    pvs = np.array(pvs)

    pv_defocus = (pvs[:,5] + pvs[:,3])
    a_defocus, b_defocus = np.polyfit(range_param, pv_defocus, 1)
    initial_params[num_param] += -b_defocus/a_defocus
    print('Finetuning defocus',initial_params[num_param])

    plt.figure()
    plt.plot(range_param, pv_defocus, label=f'legendre {order[i]}', color='k')
    ### linear fit
    coeffs = np.polyfit(range_param, pv_defocus, 1)
    fit_line = np.polyval(coeffs, range_param)
    plt.plot(range_param, fit_line, '--', label=f'y={coeffs[0]:.6e}x + {coeffs[1]:.6e}', color='k')
    plt.xlabel(f'Alignment{ num_param}')
    plt.ylabel('Defocus')
    plt.title(f'Defocus Alignment')
    plt.legend()
    plt.savefig(os.path.join(directory_name, f'Legendre_alignment_{num_param}_defocus.png'))
    plt.close()
    return initial_params


#####################################
# defocus, astigH, \
#    0        1
# pitch_hyp_v, roll_hyp_v, yaw_hyp_v, decenterX_hyp_v, decenterY_hyp_v, decenterZ_hyp_v,\
#      2           3           4           5                6                7
# pitch_hyp_h, roll_hyp_h, yaw_hyp_h, decenterX_hyp_h, decenterY_hyp_h, decenterZ_hyp_h,\
#      8           9           10          11               12               13
# pitch_ell_v, roll_ell_v, yaw_ell_v, decenterX_ell_v, decenterY_ell_v, decenterZ_ell_v,\
#      14          15          16          17               18               19
# pitch_ell_h, roll_ell_h, yaw_ell_h, decenterX_ell_h, decenterY_ell_h, decenterZ_ell_h  = params
#      20          21          22          23               24               25

# l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v = [np.float64(146.), np.float64(0.8),  np.float64(0.25), np.float64(0.5), np.float64(0.1), np.float64(0.13), np.float64(0.22)]
# a_h, b_h, a_v, b_v, l1v, l2v, [xh_s, xh_e, yh_s, yh_e, sita1h, sita3h, accept_h, NA_h, xv_s, xv_e, yv_s, yv_e, sita1v, sita3v, accept_v, NA_v, s2f_h, diff, gap] = KB_define(l1h, l2h, inc_h, mlen_h, wd_v, inc_v, mlen_v)
# sys.exit()

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# 初期値
num_params = 26  # 総パラメータ数

# initial_params = [0., 0.,
#                   0., 0., 0., 0., 0., 0.,
#                   0., 0., 0., 0., 0., 0.,
#                   0., 0., 0., 0., 0., 0.,
#                   0., 0., 0., 0., 0., 0.]
initial_params = np.array([0., 0.,
                  0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0.],dtype=np.longdouble)

# initial_params = np.array([0., -3.12012012e-5,
#                   0., 0., 0., 0., 0., 0.,
#                   -8.877777778e-3, -1.22212440e-4, -1.17e-5, 0., 0., 0.,
#                   2.75757575e-7, 0., 2.22222222e-5, 0., 0., 0.,
#                   2.5e-4, -1.19963131e-4, -1e-6, 0., 0., 0.],dtype=np.longdouble)

if option_AKB == True:
    if option_HighNA == False:
        ### AKB small
        ### Best alignment
        initial_params[9] =  -9.33896492e-07 +1.85e-6
        initial_params[14] = 5.84119937e-08
        initial_params[15] = -1.24027287e-05
        # initial_params[16] = -8.32392236e-04
        # initial_params[20] = 1.59650980e-03
        initial_params[21] = -1.37494535e-04 +1.85e-6
        initial_params[22] = -2.27244686e-06

        # ### FiX inplane
        initial_params[9] =  3.32484327e-5
        initial_params[10] = 0.
        initial_params[14] = 5.28700262e-9
        initial_params[15] = 8.24930354e-6
        initial_params[21] = -9.83732562e-5
        initial_params[22] = -2.07390025e-6


    else:
        ### AKB Large
        # # initial_params[9] =  -0.01264161
        # # initial_params[15] = 0.00455386
        # # initial_params[21] = -0.0027282
        # #
        # # initial_params[16] = 0.02445858
        # # initial_params[20] = -0.01597553
        # #
        # ### 1 Oblique
        # initial_params[9] =  -0.0085
        # initial_params[21] =  -0.0085
        # ### 2 Coma
        # initial_params[14] = 0.00023628
        # initial_params[22] = 0.00044446
        # # ### 3 Plane rotation
        # # initial_params[9] = -0.09595625
        # # initial_params[15] = -0.00410658
        # # initial_params[21] = -0.01190125
        # # ### 4 Coma
        # # initial_params[14] = -0.015
        # # initial_params[22] = 0.02
        #
        # # 3 Manual adjust
        # # initial_params[15] = -1e-4
        # # initial_params[21] = initial_params[21] - 3e-4
        # # initial_params[22] = initial_params[22] - 5e-5
        # # initial_params[20] = initial_params[20] + 0.022470357969894193
        # # initial_params[15] = initial_params[15] - 0.00023172392349930375
        # # initial_params[21] = initial_params[21] - 0.0013973110019272002
        # # initial_params[22] = initial_params[22] - 0.00010106315071820974
        # # initial_params[14] = initial_params[14] - 1.6e-06
        # ###  Best alignment
        # initial_params[14] = 6.25887113e-05
        # initial_params[15] = -7.11817756e-05
        # initial_params[16] = -1.92915809e-03
        # initial_params[20] = 1.84049615e-02
        # initial_params[21] = -8.63695329e-03
        # initial_params[22] = -9.47159502e-04
        # initial_params[9] = -8.78294573e-04

        # ### Fix inplane
        # initial_params[9] = 0.00188414
        # initial_params[14] = 1.53960991e-5
        # initial_params[15] = 5.69757870e-4
        # initial_params[21] = -0.00675583
        # initial_params[22] = -0.0011852
        print('')
        if option_wolter_3_1:
            print('set astigmatism')
            initial_params[0] = -5.74878134e-03
            initial_params[1] =  -2.87598218e-03
            # # # initial_params[0] = 5.
            # # # initial_params[1] = 10.09
            # # # ## Plane rotation
            # # # initial_params[8] = 5e-3
            # # # initial_params[20] = 5e-3
            
            # ### setting1
            # initial_params[9] = -27.8843108e-6
            # initial_params[21] = -27.8843108e-6
            # initial_params[10] = 8e-05
            # initial_params[22] = 8e-05

            # ### setting6
            # initial_params[9] = -2.51723411957558e-05
            # initial_params[21] = -2.51723411957558e-05
            # initial_params[10] = 7e-05
            # initial_params[22] = 7e-05

            ### setting12
            initial_params[9] = -3.44496117e-05
            initial_params[21] = -3.44496117e-05

            option_set = True
            initial_params[10] = 1.3e-04
            initial_params[22] = 1.3e-04

            # initial_params[12] += -2e-3
            # initial_params[24] += -2e-3
            # initial_params[10] += -6.029604417835369e-05
            # initial_params[22] += -6.029604417835369e-05
            
            # ### independent
            # option_set = False
            # initial_params[10] = 1.231404958677686e-05 + 3.0806451612903227e-06
            # initial_params[22] = 1.231404958677686e-05 + 3.0806451612903227e-06

            # initial_params[16] = 1e-3

            # initial_params[9] += -493e-6-1.173202614379085e-06-4.281045751633987e-06
            # initial_params[21] += -493e-6-1.173202614379085e-06-4.281045751633987e-06

            # initial_params[15] += -106e-6

            # factor = -10.0
            # initial_params[17] = 1e-5 * factor
            # initial_params[14] = (42e-6+8.814814814814815e-07 +2.5e-6/10) * factor +4.2e-6
            # initial_params[19] += (-1.68E-04)*1/4 * factor
            # initial_params[14] += (1.33E-03 - 3.74E-06)*1/4 * factor

            # factor = 10.0
            # initial_params[17] = 1e-5 * factor
            # # initial_params[14] = 4.5e-4-0.000025
            # initial_params[14] = (42e-6+8.814814814814815e-07 +2.5e-6/10) * factor
            # initial_params[19] += (-1.68E-04)*1/4 * factor
            # initial_params[14] += (1.33E-03 - 3.74E-06)*1/4 * factor


            ##-1.68E-04 1.33E-03

            # initial_params[10] +=  1e-4
            # initial_params[22] +=  1e-4
            # initial_params[20] +=  1e-4
            # initial_params[21] +=  4e-5
            # initial_params[9] +=  1e-5
            # initial_params[10] +=  599.5049554e-6
            # initial_params[22] +=  599.5049554e-6
            # # # initial_params[10] =  -6e-5

            # initial_params[2] +=  5e-6
            # initial_params[14] +=  5e-6
            # initial_params[15] +=  2e-5
            # initial_params[16] +=  1e-5
            # initial_params[10] += 164.1852798e-6*15
            # initial_params[10] += 623.752495e-6*15 -4.6e-3
            # initial_params[22] += 623.752495e-6 -4.6e-3
            # initial_params[9] += +35.2e-6

            # initial_params[9] += 7.94e-6
            # initial_params[21] += 7.94e-6
else:
    if option_HighNA == False:
        # # KB Small omega 0.06236049099088688
        # initial_params[8] = 4.87124219e-03
        # initial_params[9] = 1.89294655e-12
        # initial_params[10] = 4.2566550509431976e-10
        # # initial_params[8] = 0.06236049099088688
        # # initial_params[8] = 0.056678445743113724 + 4.87124219e-03
        # # omegav1 0.009011012477840984
        # # omegah1 0.004287211443082652
        # # omegav2 0.028688758361589926
        # initial_params[1] = -0.00011514
        print('KB s')
    else:
        # ## KB Large
        # initial_params[8] = 3.715133e-2 +0.4566017057125132
        # initial_params[8] = 0.5038395891681975
        # initial_params[9] = 4.65100097e-8
        # initial_params[10] = 1.21012885e-7
        #
        # initial_params[8] = 4.93753036e-01
        # initial_params[9] = 4.65100097e-08
        # initial_params[10] = 5e-05
        print('KB L')
        # initial_params[0] = -0.12
        # initial_params[1] = -0.09
        # l_i1 = np.float64(145.7500024376426)
        
        # # l_o1 = np.float64(0.085)
        # # theta_g1 = np.float64(0.2)
        # # l_o1: 0.149786, theta_g1: 0.150881, target_l_o2: 0.021402, aperture_Ell1: 0.015456, aperture_Ell2: 0.005000        

        # var_l_o1 = np.float64(0.15)
        # theta_g1 = np.float64(0.16)
        # target_l_o2 = np.float64(0.02125) ### WD
        # l_o1 = var_l_o1.copy()  # 初期値の設定
        # na_o_sin = np.float64(0.082)
        # target_gap = np.float64(0.013)
        # vals1 = 146 + np.linspace(-100, 100, 3)
        vals2 = np.linspace(0.005, 0.02, 5)
        vals3 = np.linspace(0.01, 0.03, 5)
        vals4 = np.linspace(0.1, 0.22, 5)
        vals5 = np.linspace(0.18, 0.22, 5)
        
        pvs = []
        div_angle1 = []
        div_angle2 = []
        mirr_length1 = []
        mirr_length2 = []
        Aperture1 = []
        Aperture2 = []
        Mag1 = []
        Mag2 = []

        l_i1 = np.float64(140.) 
        target_gap = np.float64(0.005)
        target_l_o2 = np.float64(0.02) ### WD
        theta_g1 = np.float64(0.18)
        l_o1 = np.float64(0.2)
        
        
        
        na_o_sin = np.float64(0.082)
        ast = np.float64(0.)
        # for val1 in vals1:
        for val2 in vals2:
            for val3 in vals3:
                for val4 in vals4:
                    for val5 in vals5:
                        # l_i1 = np.float64(val1)
                        target_gap = np.float64(val2)
                        target_l_o2 = np.float64(val3)  # WD
                        theta_g1 = np.float64(val4)
                        l_o1 = np.float64(val5)
                        # 各変数の値を設定
                        directory_name = f"output_{timestamp}_KB/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        # 新しいフォルダを作成
                        os.makedirs(directory_name, exist_ok=True)

                        conditions_file_path = os.path.join(directory_name, 'kb_params.txt')

                        # テキストファイルに変数の値や計算条件を書き込む
                        with open(conditions_file_path, 'w') as file:
                            file.write("input\n")
                            file.write("====================\n")
                            file.write(f"l_i1: {l_i1}\n")
                            file.write(f"l_o1: {l_o1}\n")
                            file.write(f"theta_g1: {theta_g1}\n")
                            file.write(f"na_o_sin: {na_o_sin}\n")
                            file.write(f"target_l_o2: {target_l_o2}\n")
                            file.write(f"target_gap: {target_gap}\n")
                            file.write(f"ast: {ast}\n")
                        kb_manager.set_design(l_i1, l_o1, theta_g1, na_o_sin, target_l_o2, target_gap, ast)

                        option_mpmath = False
                        # auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
                        pvs.append(auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray_wave'))
                        div_angle1.append(kb_manager.Ell1.theta_i1-kb_manager.Ell1.theta_i2)
                        div_angle2.append(kb_manager.Ell2.theta_i1-kb_manager.Ell2.theta_i2)
                        mirr_length1.append(kb_manager.Ell1.mirr_length)
                        mirr_length2.append(kb_manager.Ell2.mirr_length)
                        Aperture1.append(kb_manager.Ell1.mirr_length*kb_manager.Ell1.theta_centre)
                        Aperture2.append(kb_manager.Ell2.mirr_length*kb_manager.Ell2.theta_centre)
                        Mag1.append((kb_manager.Ell1.m1 + kb_manager.Ell1.m2) / 2)
                        Mag2.append((kb_manager.Ell2.m1 + kb_manager.Ell2.m2) / 2)

                        plt.close('all')
                        
                        kb_manager = KBDesignManager()

        directory_name = f"output_{timestamp}_KB"
        pvs = np.array(pvs)
        div_angle1 = np.array(div_angle1)
        div_angle2 = np.array(div_angle2)
        mirr_length1 = np.array(mirr_length1)
        mirr_length2 = np.array(mirr_length2)
        Aperture1 = np.array(Aperture1)
        Aperture2 = np.array(Aperture2)
        Mag1 = np.array(Mag1)
        Mag2 = np.array(Mag2)
        names = ['Div Angle1', 'Div Angle2', 'Mirror Length1', 'Mirror Length2', 'Aperture1', 'Aperture2', 'Magnification1', 'Magnification2']
        param_stack = np.vstack((div_angle1, div_angle2, mirr_length1, mirr_length2, Aperture1, Aperture2, Mag1, Mag2))

        savenames = ['PV','DivAngle1', 'DivAngle2', 'MirrorLength1', 'MirrorLength2', 'Aperture1', 'Aperture2', 'Magnification1', 'Magnification2']
        savedata = np.vstack((pvs, div_angle1, div_angle2, mirr_length1, mirr_length2, Aperture1, Aperture2, Mag1, Mag2))
        # pandas.DataFrame にする
        df = pd.DataFrame(savedata.T, columns=savenames)

        # CSV に保存
        df.to_csv(os.path.join(directory_name,'parameters.csv'), index=False)

        num_parameter = 8
        fig, ax = plt.subplots(3,3, figsize=(15, 15))
        ax = ax.ravel()
        fig.suptitle('KB Design PV Dependence on Divergence Angles')

        for i in range(num_parameter):
            ax[i].scatter(param_stack[i], pvs, color='k')
            coeff = np.polyfit(param_stack[i], pvs, 1)
            fit_line = np.polyval(coeff, param_stack[i])
            r2 = np.corrcoef(param_stack[i], pvs)[0, 1]**2
            ax[i].plot(param_stack[i], fit_line, '--', label=f'y={coeff[0]:.2e}x + {coeff[1]:.2e} (R²={r2:.2f})', color='k')
            ax[i].set_xlabel(names[i])
            ax[i].set_ylabel('PV')
            ax[i].set_title('KB Design PV')
            ax[i].legend()
        fig.subplots_adjust(hspace=0.5)  # デフォルトは 0.2 くらい
        plt.savefig(os.path.join(directory_name, 'PV_dependence.png'))
        plt.show()
        sys.exit()

# option_mpmath = True
# plot_result_debug(initial_params,'ray_wave')

# calc_FoC(initial_params)
# initial_params[2] += 1e-5

# Legendrealignment(initial_params, [8,20], np.linspace(-2e-3, 2e-3, 5), tuning=True)

option_mpmath = False
auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray_wave')
initial_params = Finetuning(initial_params)
option_mpmath = True
plot_result_debug(initial_params,'ray_wave')

# auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
# option_mpmath = True
# plot_result_debug(initial_params,'ray')
# plot_result_debug(initial_params,'ray_wave')
# option_mpmath = False
print('initial_params', initial_params)
Legendrealignment(initial_params, [10,22], np.linspace(-2e-5, 2e-5, 5), tuning=True)
# Legendrealignment(initial_params, [12,24], np.linspace(-2e-3, 2e-3, 5), tuning=True)
plt.show()
# Legendrealignment(initial_params, [9,21], np.linspace(-1e-5, 1e-5, 5), tuning=False)
# Legendrealignment(initial_params, [10,22], np.linspace(-1e-5, 1e-5, 5), tuning=False)
# Legendrealignment(initial_params, [10], np.linspace(-1e-5, 1e-5, 5), tuning=False)
# Legendrealignment(initial_params, [22], np.linspace(-1e-5, 1e-5, 5), tuning=False)
# Legendrealignment(initial_params, [2], np.linspace(-1e-5, 1e-5, 5), tuning=False)
# Legendrealignment(initial_params, [16], np.linspace(-1e-5, 1e-5, 5), tuning=False)

### V Ell
# Legendrealignment(initial_params, [14], np.linspace(-1e-5, 1e-5, 5), tuning=True)
# Legendrealignment(initial_params, [15], np.linspace(-1e-5, 1e-5, 5), tuning=True)
# Legendrealignment(initial_params, [16], np.linspace(-1e-5, 1e-5, 5), tuning=True)
# Legendrealignment(initial_params, [17], np.linspace(-1e-5, 1e-5, 5), tuning=True)
# Legendrealignment(initial_params, [19], np.linspace(-1e-6, 1e-6, 5), tuning=True)

# Legendrealignment(initial_params, [9,21], np.linspace(-1e-5, 1e-5, 5), tuning=True)
# Legendrealignment(initial_params, [10,22], np.linspace(-1e-3, 1e-3, 5), tuning=True)
# Legendrealignment(initial_params, [10,22], np.linspace(-5e-4, 5e-4, 5), tuning=False)
# Legendrealignment(initial_params, [8,20], np.linspace(-1e-3, 1e-3, 5), tuning=False)

# Legendrealignment(initial_params, [23], np.linspace(-1e-4, 1e-4, 5), tuning=True)
# Legendrealignment(initial_params, [24], np.linspace(-1e-5, 1e-5, 5), tuning=True)
# Legendrealignment(initial_params, [20], np.linspace(-1e-4, 1e-4, 5), tuning=True)
# Legendrealignment(initial_params, [21], np.linspace(-1e-4, 1e-4, 5), tuning=True)
# Legendrealignment(initial_params, [22], np.linspace(-1e-4, 1e-4, 5), tuning=True)

# Legendrealignment(initial_params, [8], np.linspace(-1e-4, 1e-4, 5), tuning=True)
# Legendrealignment(initial_params, [9], np.linspace(-1e-4, 1e-4, 5), tuning=True)
# Legendrealignment(initial_params, [10], np.linspace(-5e-4, 5e-4, 5), tuning=True)
# Legendrealignment(initial_params, [11], np.linspace(-1e-4, 1e-4, 5), tuning=True)
# Legendrealignment(initial_params, [12], np.linspace(-1e-5, 1e-5, 5), tuning=True)

# Legendrealignment(initial_params, [2,14], np.linspace(-2e-5, 2e-5, 5), tuning=False)
# Legendrealignment(initial_params, [2], np.linspace(-1e-4, 1e-4, 5), tuning=True)
# Legendrealignment(initial_params, [3], np.linspace(-1e-5, 1e-5, 5), tuning=True)
# Legendrealignment(initial_params, [4], np.linspace(-1e-4, 1e-4, 5), tuning=True)
# Legendrealignment(initial_params, [5], np.linspace(-1e-6, 1e-6, 5), tuning=True)
# Legendrealignment(initial_params, [7], np.linspace(-1e-6, 1e-6, 5), tuning=True)



if option_AKB == False:
    initial_params1 = initial_params.copy()
    abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = 'KB')
    print('abrr',abrr)
    M9 = auto_focus_sep(initial_params1.copy(),9,9,-1e-6,1e-6,option = 'matrix', option_eval = 'KB')
    print('initial_params1',initial_params1.copy())
    M10 = auto_focus_sep(initial_params1.copy(),10,10,-1e-6,1e-6,option = 'matrix', option_eval = 'KB')
    print('initial_params1',initial_params1.copy())
    M8 = auto_focus_sep(initial_params1.copy(),8,8,-1e-4,1e-4,option = 'matrix', option_eval = 'KB')
    M = np.array(np.vstack([M8, M9, M10]),dtype=np.float64)
    M = M.astype(np.float64)  # 明示的に float64 に変換
    print(M)
    inverse_M = np.linalg.inv(M)
    print('inverse_M',inverse_M)

    params = -np.dot(abrr,inverse_M)
    print('params',params)
    print('initial_params1',initial_params1)
    print('initial_params',initial_params)
    initial_params[8] = initial_params[8] + params[0]
    initial_params[9] = initial_params[9] + params[1]
    initial_params[10] = initial_params[10] + params[2]

    auto_focus_NA(50, initial_params.copy(),1,1, True,'',option_disp='ray')
    auto_focus_NA(50, initial_params.copy(),1,1, True,'',option_disp='ray_wave')

auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray_wave')

initial_params1 = initial_params.copy()
# M22 = auto_focus_sep(initial_params1.copy(),22,22,-1e-6,1e-6,option = 'matrix', option_eval = '3')
# M8 = auto_focus_sep(initial_params1.copy(),8,20,-1e-2,1e-2,option = 'matrix', option_eval = '3')
if option_wolter_3_1 or option_wolter_3_3_tandem:
    if False:
        M9 = auto_focus_sep(initial_params1.copy(),9,21,-5e-5,5e-5,option = 'matrix', option_eval = '3_intercept')
        # M9 = auto_focus_sep(initial_params1.copy(),9,21,-5e-4,5e-4,option = 'matrix', option_eval = '3_intercept')
        initial_params[9] = -M9[1,0]/M9[0,0]
        initial_params[21] = -M9[1,0]/M9[0,0]
        initial_params1 = initial_params.copy()
        print('M9',M9)
        M10 = auto_focus_sep(initial_params1.copy(),10,22,-5e-5,5e-5,option = 'matrix', option_eval = '3_intercept')
        # M10 = auto_focus_sep(initial_params1.copy(),10,22,-5e-4,5e-4,option = 'matrix', option_eval = '3_intercept')
        initial_params[10] = -M10[1,1]/M10[0,1]
        initial_params[22] = -M10[1,1]/M10[0,1]
        print('M10',M10)
        initial_params1 = initial_params.copy()
        param_MinH = auto_focus_sep(initial_params1.copy(),10,22,-1e-5,1e-5,option = 'matrix', option_eval = 'MinimizeH')
        print('param_MinH',param_MinH)
        initial_params[10] = param_MinH
        initial_params[22] = param_MinH

        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray_wave')

    if False:
        base_directory = f"output_{timestamp}"
        ###### Torerance WolterH
        ### incidence
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/WolterH_incidence'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[10] += 5e-4
        initial_params1[22] += 5e-4
        auto_focus_NA(50, initial_params1,1,1, True,'',option_disp='ray_wave')

        ### Perpendicularity
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/WolterH_perpendicularity'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[9] += 1e-5
        initial_params1[21] += 1e-5
        auto_focus_NA(50, initial_params1,1,1, True,'',option_disp='ray_wave')

        ###### Torerance HypH relative angle
        ### incidence
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/HypH_incidence'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[10] += 1e-4
        auto_focus_NA(50, initial_params1,1,1, True,'',option_disp='ray_wave')

        ### Perpendicularity
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/HypH_perpendicularity'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[9] += 4e-5
        auto_focus_NA(50, initial_params1,1,1, True,'',option_disp='ray_wave')

        ### rotation
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/HypH_rotation'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[8] += 1e-4
        auto_focus_NA(50, initial_params1,1,1, True,'',option_disp='ray_wave')

        ###### Torerance EllV relative angle
        ### incidence
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/EllV_incidence'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[14] += 5e-6
        auto_focus_NA(50, initial_params1,1,1, True,'',option_disp='ray_wave')

        ### Perpendicularity
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/EllV_perpendicularity'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[15] += 2e-5
        auto_focus_NA(50, initial_params1,1,1, True,'',option_disp='ray_wave')

        ### rotation
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/EllV_rotation'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[16] += 1e-5
        auto_focus_NA(50, initial_params1,1,1, True,'',option_disp='ray_wave')

    if True:
        base_directory = f"output_{timestamp}"
        ###### Torerance WolterH
        ### incidence
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/WolterH_incidence'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[10] += 5e-5
        initial_params1[22] += 5e-5
        plot_result_debug(initial_params1,'ray_wave')

        ### Perpendicularity
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/WolterH_perpendicularity'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[9] += 1e-6
        initial_params1[21] += 1e-6
        plot_result_debug(initial_params1,'ray_wave')

        ###### Torerance HypH relative angle
        ### incidence
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/HypH_incidence'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[10] += 1e-5
        plot_result_debug(initial_params1,'ray_wave')

        ### Perpendicularity
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/HypH_perpendicularity'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[9] += 4e-6
        plot_result_debug(initial_params1,'ray_wave')

        ### rotation
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/HypH_rotation'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[8] += 1e-5
        plot_result_debug(initial_params1,'ray_wave')

        ###### Torerance EllV relative angle
        ### incidence
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/EllV_incidence'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[14] += 5e-7
        plot_result_debug(initial_params1,'ray_wave')

        ### Perpendicularity
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/EllV_perpendicularity'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[15] += 2e-6
        plot_result_debug(initial_params1,'ray_wave')

        ### rotation
        initial_params1 = initial_params.copy()
        directory_name = base_directory + '/EllV_rotation'
        os.makedirs(directory_name, exist_ok=True)
        initial_params1[16] += 1e-6
        plot_result_debug(initial_params1,'ray_wave')

    sys.exit()

    M14 = auto_focus_sep(initial_params1.copy(),14,14,-1e-5,1e-5,option = 'matrix', option_eval = '3')
    M15 = auto_focus_sep(initial_params1.copy(),15,15,-5e-5,5e-5,option = 'matrix', option_eval = '3')
    M16 = auto_focus_sep(initial_params1.copy(),16,16,-5e-5,5e-5,option = 'matrix', option_eval = '3')
    print('M14',M14)
    print('M15',M15)
    print('M16',M16)

else:
    option_fix2ndcoma = False
    # option_abrr = 'inplaneadjust'
    option_abrr = 'inplanefix'
    # option_abrr = 'inplaneadjustwithcoma'
    option_adjust_oblique = False
    


    option_adjust_coma = False
    ### 面内回転固定
    if option_abrr == 'KB_coma':
        initial_params1 = initial_params.copy()
        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '2')
        print('abrr',abrr)
        M10 = auto_focus_sep(initial_params1,10,10,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        initial_params[10] = - M10[1,1]/M10[0,1]
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray_wave')
        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '9')
        print('initial_params',initial_params)
        print('abrr',abrr)
    if option_adjust_coma:
        initial_params1 = initial_params.copy()
        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '2')
        print('abrr',abrr)
        M14 = auto_focus_sep(initial_params1,14,14,-1e-4,1e-4,option = 'matrix', option_eval = '2')
        M22 = auto_focus_sep(initial_params1,22,22,-1e-4,1e-4,option = 'matrix', option_eval = '2')

        initial_params[14] = - M14[1,0]/M14[0,0]
        initial_params[22] = - M22[1,1]/M22[0,1]
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '9')
        print('initial_params',initial_params)
        print('abrr',abrr)
        saveWaveData(initial_params)

    if option_abrr == 'KB':
        initial_params1 = initial_params.copy()
        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = 'KB')
        print('abrr',abrr)
        M9 = auto_focus_sep(initial_params1.copy(),9,9,-1e-6,1e-6,option = 'matrix', option_eval = 'KB')
        print('initial_params1',initial_params1.copy())
        M10 = auto_focus_sep(initial_params1.copy(),10,10,-1e-6,1e-6,option = 'matrix', option_eval = 'KB')
        print('initial_params1',initial_params1.copy())
        M8 = auto_focus_sep(initial_params1.copy(),8,8,-1e-4,1e-4,option = 'matrix', option_eval = 'KB')
        M = np.array(np.vstack([M8, M9, M10]),dtype=np.float64)
        M = M.astype(np.float64)  # 明示的に float64 に変換
        print(M)
        inverse_M = np.linalg.inv(M)
        print('inverse_M',inverse_M)

        params = -np.dot(abrr,inverse_M)
        print('params',params)
        print('initial_params1',initial_params1)
        print('initial_params',initial_params)
        initial_params[8] = initial_params[8] + params[0]
        initial_params[9] = initial_params[9] + params[1]
        initial_params[10] = initial_params[10] + params[2]

        auto_focus_NA(50, initial_params.copy(),1,1, True,'',option_disp='ray')
        abrr = auto_focus_sep(initial_params.copy(),0,0,0,0,option = 'abrr', option_eval = 'KB')
        print('abrr',abrr)
        auto_focus_NA(50, initial_params.copy(),1,1, True,'',option_disp='ray_wave')

        saveWaveData(initial_params)

    if option_abrr == 'inplanefix':
        initial_params1 = initial_params
        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '3')
        print('abrr',abrr)
        M9 = auto_focus_sep(initial_params1,9,9,-1e-6,1e-6,option = 'matrix', option_eval = '3')
        M15 = auto_focus_sep(initial_params1,15,15,-1e-6,1e-6,option = 'matrix', option_eval = '3')
        M21 = auto_focus_sep(initial_params1,21,21,-1e-6,1e-6,option = 'matrix', option_eval = '3')

        M = np.array(np.vstack([M9[0,:], M15[0,:], M21[0,:]]),dtype=np.float64)
        M = M.astype(np.float64)  # 明示的に float64 に変換
        print(M)
        inverse_M = np.linalg.inv(M)
        print('inverse_M',inverse_M)

        params_woComa = -np.dot(abrr,inverse_M)
        print('params_woComa',params_woComa)

        initial_params[9] = initial_params[9] + params_woComa[0]
        initial_params[15] = initial_params[15] + params_woComa[1]
        initial_params[21] = initial_params[21] + params_woComa[2]

        initial_params1 = initial_params
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')

        saveWaveData(initial_params)

        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '2')
        print('abrr',abrr)
        M14 = auto_focus_sep(initial_params1,14,14,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        M22 = auto_focus_sep(initial_params1,22,22,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        initial_params[14] = - M14[1,0]/M14[0,0]
        initial_params[22] = - M22[1,1]/M22[0,1]
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
    elif option_abrr == 'inplanefixwithcoma':
        initial_params1 = initial_params.copy()
        abrr = auto_focus_sep(initial_params.copy(),0,0,0,0,option = 'abrr', option_eval = '5coma')
        print('abrr',abrr)
        M9 = auto_focus_sep(initial_params1,9,9,-1e-6,1e-6,option = 'matrix', option_eval = '5coma')
        initial_params1 = initial_params.copy()
        M14 = auto_focus_sep(initial_params1,14,14,-1e-6,1e-6,option = 'matrix', option_eval = '5coma')
        initial_params1 = initial_params.copy()
        M15 = auto_focus_sep(initial_params1,15,15,-1e-6,1e-6,option = 'matrix', option_eval = '5coma')
        initial_params1 = initial_params.copy()
        M21 = auto_focus_sep(initial_params1,21,21,-1e-6,1e-6,option = 'matrix', option_eval = '5coma')
        initial_params1 = initial_params.copy()
        if option_fix2ndcoma:
            M22 = auto_focus_sep(initial_params1,22,22,-1e-6,1e-6,option = 'matrix', option_eval = '5coma')
            M = np.array(np.vstack([M9, M14, M15, M21, M22]),dtype=np.float64)
        else:
            M10 = auto_focus_sep(initial_params1,10,10,-1e-6,1e-6,option = 'matrix', option_eval = '5coma')
            M = np.array(np.vstack([M9, M14, M15, M21, M10]),dtype=np.float64)
        M = M.astype(np.float64)  # 明示的に float64 に変換
        print(M)
        print(M.dtype)
        print(np.linalg.det(M))
        print(abrr.dtype, abrr.shape)
        inverse_M = np.linalg.inv(M)
        print('inverse_M',inverse_M)
        params = -np.dot(abrr,inverse_M)
        print('params_woComa',params)
        initial_params[9] = initial_params[9] + params[0]
        initial_params[14] = initial_params[14] + params[1]
        initial_params[15] = initial_params[15] + params[2]
        initial_params[21] = initial_params[21] + params[3]
        if option_fix2ndcoma:
            initial_params[22] = initial_params[22] + params[4]
        else:
            initial_params[10] = initial_params[10] + params[4]

        initial_params1 = initial_params.copy()
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '2')
        print('abrr',abrr)
        M14 = auto_focus_sep(initial_params1,14,14,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        M22 = auto_focus_sep(initial_params1,22,22,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        initial_params[14] = - M14[1,0]/M14[0,0]
        initial_params[22] = - M22[1,1]/M22[0,1]
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
    elif option_abrr == 'inplaneadjust':
        initial_params1 = initial_params
        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '5')
        print('abrr',abrr)
        M9 = auto_focus_sep(initial_params1,9,9,-1e-6,1e-6,option = 'matrix', option_eval = '5')
        M15 = auto_focus_sep(initial_params1,15,15,-1e-6,1e-6,option = 'matrix', option_eval = '5')
        M16 = auto_focus_sep(initial_params1,16,16,-1e-3,1e-3,option = 'matrix', option_eval = '5')
        M20 = auto_focus_sep(initial_params1,20,20,-1e-3,1e-3,option = 'matrix', option_eval = '5')
        M21 = auto_focus_sep(initial_params1,21,21,-1e-6,1e-6,option = 'matrix', option_eval = '5')


        M = np.array(np.vstack([M9, M15, M16, M20, M21]),dtype=np.float64)
        M = M.astype(np.float64)  # 明示的に float64 に変換
        print(M)
        inverse_M = np.linalg.inv(M)
        print('inverse_M',inverse_M)

        params_woComa = -np.dot(abrr,inverse_M)
        print('params_woComa',params_woComa)
        # params = -np.dot(abrr,inverse_M)
        # print('params',params)

        initial_params[9] = initial_params[9] + params_woComa[0]
        initial_params[15] = initial_params[15] + params_woComa[1]
        initial_params[16] = initial_params[16] + params_woComa[2]
        initial_params[20] = initial_params[20] + params_woComa[3]
        initial_params[21] = initial_params[21] + params_woComa[4]
        # initial_params[22] = initial_params[22] - abrr[2]/M22[1]

        initial_params1 = initial_params
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')

        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '2')
        print('abrr',abrr)
        M14 = auto_focus_sep(initial_params1,14,14,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        M22 = auto_focus_sep(initial_params1,22,22,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        initial_params[14] = - M14[1,0]/M14[0,0]
        initial_params[22] = - M22[1,1]/M22[0,1]
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
    elif option_abrr == 'inplaneadjustwithcoma':
        initial_params1 = initial_params.copy()
        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '7coma')
        print('abrr',abrr)
        M9 = auto_focus_sep(initial_params1,9,9,-1e-6,1e-6,option = 'matrix', option_eval = '7coma')
        initial_params1 = initial_params.copy()
        M14 = auto_focus_sep(initial_params1,14,14,-1e-6,1e-6,option = 'matrix', option_eval = '7coma')
        initial_params1 = initial_params.copy()
        M15 = auto_focus_sep(initial_params1,15,15,-1e-6,1e-6,option = 'matrix', option_eval = '7coma')
        initial_params1 = initial_params.copy()
        M16 = auto_focus_sep(initial_params1,16,16,-1e-3,1e-3,option = 'matrix', option_eval = '7coma')
        initial_params1 = initial_params.copy()
        M20 = auto_focus_sep(initial_params1,20,20,-1e-3,1e-3,option = 'matrix', option_eval = '7coma')
        initial_params1 = initial_params.copy()
        M21 = auto_focus_sep(initial_params1,21,21,-1e-6,1e-6,option = 'matrix', option_eval = '7coma')
        initial_params1 = initial_params.copy()
        M22 = auto_focus_sep(initial_params1,22,22,-1e-6,1e-6,option = 'matrix', option_eval = '7coma')

        M = np.array(np.vstack([M9, M14, M15, M16, M20, M21, M22]),dtype=np.float64)
        M = M.astype(np.float64)  # 明示的に float64 に変換
        print(M)
        inverse_M = np.linalg.inv(M)
        print('inverse_M',inverse_M)

        params_woComa = -np.dot(abrr,inverse_M)
        print('params_woComa',params_woComa)
        # params = -np.dot(abrr,inverse_M)
        # print('params',params)

        initial_params[9] = initial_params[9] + params_woComa[0]
        initial_params[14] = initial_params[14] + params_woComa[1]
        initial_params[15] = initial_params[15] + params_woComa[2]
        initial_params[16] = initial_params[16] + params_woComa[3]
        initial_params[20] = initial_params[20] + params_woComa[4]
        initial_params[21] = initial_params[21] + params_woComa[5]
        initial_params[22] = initial_params[22] + params_woComa[6]

        initial_params1 = initial_params.copy()
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')

        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '2')
        print('abrr',abrr)
        M14 = auto_focus_sep(initial_params1,14,14,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        M22 = auto_focus_sep(initial_params1,22,22,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        initial_params[14] = - M14[1,0]/M14[0,0]
        initial_params[22] = - M22[1,1]/M22[0,1]
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
    elif option_abrr == 'alladjust':
        initial_params1 = initial_params.copy()
        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '9')
        print('abrr',abrr)
        M8 = auto_focus_sep(initial_params1,8,8,-1e-4,1e-4,option = 'matrix', option_eval = '9')
        M9 = auto_focus_sep(initial_params1,9,9,-1e-6,1e-6,option = 'matrix', option_eval = '9')
        M10 = auto_focus_sep(initial_params1,10,10,-1e-6,1e-6,option = 'matrix', option_eval = '9')
        M14 = auto_focus_sep(initial_params1,14,14,-1e-6,1e-6,option = 'matrix', option_eval = '9')
        M15 = auto_focus_sep(initial_params1,15,15,-1e-6,1e-6,option = 'matrix', option_eval = '9')
        M16 = auto_focus_sep(initial_params1,16,16,-1e-3,1e-3,option = 'matrix', option_eval = '9')
        M20 = auto_focus_sep(initial_params1,20,20,-1e-3,1e-3,option = 'matrix', option_eval = '9')
        M21 = auto_focus_sep(initial_params1,21,21,-1e-6,1e-6,option = 'matrix', option_eval = '9')
        M22 = auto_focus_sep(initial_params1,22,22,-1e-6,1e-6,option = 'matrix', option_eval = '9')

        M = np.array(np.vstack([M8, M9, M10, M14, M15, M16, M20, M21, M22]),dtype=np.float64)
        M = M.astype(np.float64)  # 明示的に float64 に変換
        print(M)
        print(M.dtype)
        print(np.linalg.det(M))
        print(abrr.dtype, abrr.shape)
        inverse_M = np.linalg.inv(M)
        print('inverse_M',inverse_M)

        params_woComa = -np.dot(abrr,inverse_M)
        print('params_woComa',params_woComa)


        if False:
            def solve_linear_system(A, x0, y0):
                """
                線形方程式 Ax + b = 0 を解く。

                A: nxn の係数行列
                x0: 初期点 (x1_0, ..., xn_0) の numpy 配列
                y0: 初期点における y の値 (y1_0, ..., yn_0) の numpy 配列

                戻り値: y1 = ... = yn = 0 となる (x1, ..., xn)
                """
                # 定数ベクトル b の計算
                b = y0 - A @ x0
                print(b)
                print(b.shape)
                print(np.isnan(b).any())
                print(np.isinf(b).any())

                # 係数行列 A のランクを確認して特異行列かどうかを判断
                if np.linalg.matrix_rank(A) < A.shape[0]:
                    print("警告: 係数行列 A は特異行列です。擬似逆行列を使用して解を求めます。")

                # 擬似逆行列を用いて解を求める
                x = np.linalg.pinv(A) @ (-b)
                return x

            # 例: A, x0, y0 の設定
            n = 9  # n×n のサイズを指定
            x0 = np.array([initial_params[9],initial_params[10],initial_params[11],initial_params[14],initial_params[15],initial_params[16],initial_params[20],initial_params[21],initial_params[22]])  # 初期点
            y0 = abrr.copy()  # 初期 y の値

            # 解を求める
            x_solution = solve_linear_system(M, x0, y0)
            print("解 (x1, ..., xn):", x_solution)
            params_woComa = x_solution.copy()
        # params = -np.dot(abrr,inverse_M)
        # print('params',params)
        initial_params[8] = initial_params[8] + params_woComa[0]
        initial_params[9] = initial_params[9] + params_woComa[1]
        initial_params[10] = initial_params[10] + params_woComa[2]
        initial_params[14] = initial_params[14] + params_woComa[3]
        initial_params[15] = initial_params[15] + params_woComa[4]
        initial_params[16] = initial_params[16] + params_woComa[5]
        initial_params[20] = initial_params[20] + params_woComa[6]
        initial_params[21] = initial_params[21] + params_woComa[7]
        initial_params[22] = initial_params[22] + params_woComa[8]

        initial_params1 = initial_params
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')

        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '2')
        print('abrr',abrr)
        M14 = auto_focus_sep(initial_params1,14,14,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        M22 = auto_focus_sep(initial_params1,22,22,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        initial_params[14] = - M14[1,0]/M14[0,0]
        initial_params[22] = - M22[1,1]/M22[0,1]
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
    elif option_abrr == 'alladjust34':
        initial_params1 = initial_params.copy()
        abrr = auto_focus_sep(initial_params.copy(),0,0,0,0,option = 'abrr', option_eval = '5coma')
        print('abrr',abrr)
        M14 = auto_focus_sep(initial_params1,14,14,-1e-6,1e-6,option = 'matrix', option_eval = '5coma')
        M16 = auto_focus_sep(initial_params1,16,16,-1e-3,1e-3,option = 'matrix', option_eval = '5coma')
        M20 = auto_focus_sep(initial_params1,20,20,-1e-3,1e-3,option = 'matrix', option_eval = '5coma')
        M21 = auto_focus_sep(initial_params1,21,21,-1e-6,1e-6,option = 'matrix', option_eval = '5coma')
        M22 = auto_focus_sep(initial_params1,22,22,-1e-6,1e-6,option = 'matrix', option_eval = '5coma')

        M = np.array(np.vstack([M14, M16, M20, M21, M22]),dtype=np.float64)
        M = M.astype(np.float64)  # 明示的に float64 に変換
        print(M)
        print(M.dtype)
        print(np.linalg.det(M))
        print(abrr.dtype, abrr.shape)
        inverse_M = np.linalg.inv(M)
        print('inverse_M',inverse_M)
        params = -np.dot(abrr,inverse_M)
        print('params_woComa',params)
        initial_params[14] = initial_params[14] + params[0]
        initial_params[16] = initial_params[16] + params[1]
        initial_params[20] = initial_params[20] + params[2]
        initial_params[21] = initial_params[21] + params[3]
        initial_params[22] = initial_params[22] + params[4]

        initial_params1 = initial_params.copy()
        abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '2')
        print('abrr',abrr)
        M14 = auto_focus_sep(initial_params1,14,14,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        M22 = auto_focus_sep(initial_params1,22,22,-1e-6,1e-6,option = 'matrix', option_eval = '2')
        initial_params[14] = - M14[1,0]/M14[0,0]
        initial_params[22] = - M22[1,1]/M22[0,1]
        auto_focus_NA(50, initial_params,1,1, True,'',option_disp='ray')
    abrr = auto_focus_sep(initial_params,0,0,0,0,option = 'abrr', option_eval = '9')
    print('initial_params',initial_params)
    print('abrr',abrr)
    saveWaveData(initial_params)
