import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AKB_calc_rotate import rotation_2D 
from AKB_raytrace_20250312 import point_rotate_x, point_rotate_y, point_rotate_z
def normal_from_quad(p0, p1, p2, p3):
    """
    四辺形（p0,p1,p2,p3）の平均法線ベクトルを単位ベクトルで返す。
    頂点は反時計回り（または時計回り）に並べてください。

    p0...p3: 各々 iterable of length 3 (x,y,z)
    return: ndarray of shape (3,) の単位法線ベクトル
    """
    pts = [np.asarray(p, dtype=float) for p in (p0, p1, p2, p3)]
    # Newell's method による面法線の積算
    n = np.zeros(3, dtype=float)
    for i in range(4):
        j = (i + 1) % 4
        xi, yi, zi = pts[i]
        xj, yj, zj = pts[j]
        n[0] += (yi - yj) * (zi + zj)
        n[1] += (zi - zj) * (xi + xj)
        n[2] += (xi - xj) * (yi + yj)
    norm = np.linalg.norm(n)
    if norm == 0:
        # 退避：全頂点同一平面だが凹状，もしくは一直線上など
        return np.array([np.nan, np.nan, np.nan])
    return n / norm

def skew(v):
    """ベクトル v のスキュー対称行列 [v]_× を返す"""
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]], dtype=float)

def rotation_matrix_from_vectors(ref, tgt, eps=1e-8):
    """
    ref を tgt へ回す 3x3 回転行列 R を返す。
    ref, tgt は長さ 3 のベクトル
    """
    # 正規化
    a = np.asarray(ref, float)
    b = np.asarray(tgt, float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        raise ValueError("zero-length vector")
    a /= na
    b /= nb

    v = np.cross(a, b)
    c = np.dot(a, b)
    s2 = np.dot(v, v)         # s^2 = ||v||^2 = 1 - c^2

    if s2 < eps:
        # ref と tgt が平行か逆向き
        if c > 0:
            return np.eye(3)  # ほぼ同じ方向
        # 逆向きのときは任意の直交軸で 180° 回転
        # ref と直交する適当な軸を作る
        axis = np.cross(a, [1,0,0])
        if np.linalg.norm(axis) < eps:
            axis = np.cross(a, [0,1,0])
        axis /= np.linalg.norm(axis)
        K = skew(axis)
        return np.eye(3) + 2 * K @ K  # sin(pi)=0, 1−cos(pi)=2

    # 通常ケース：Rodrigues の公式を v = axis * sinθ の形で使う
    K = skew(v)
    R = np.eye(3) + K + K @ K * ((1 - c) / s2)
    return R

def rotate_point_cloud(data, ref_vec, tgt_vec):
    """
    data: shape (3, N) の点群
    ref_vec: 基準法線ベクトル (長さ 3)
    tgt_vec: 目標法線ベクトル (長さ 3)
    return: shape (3, N) の回転後点群
    """
    R = rotation_matrix_from_vectors(ref_vec, tgt_vec)
    return R @ data

if __name__ == "__main__":
    hyp_v = np.load(r"C:\Users\K.Hanada\Desktop\AKBRaytracing\output_20260116_103710_wolter_3_1_correct\vmirr_hyp.npy")
    ell_v = np.load(r"C:\Users\K.Hanada\Desktop\AKBRaytracing\output_20260116_103710_wolter_3_1_correct\vmirr_ell.npy")
    hyp_h = np.load(r"C:\Users\K.Hanada\Desktop\AKBRaytracing\output_20260116_103710_wolter_3_1_correct\hmirr_hyp.npy")
    ell_h = np.load(r"C:\Users\K.Hanada\Desktop\AKBRaytracing\output_20260116_103710_wolter_3_1_correct\hmirr_ell.npy")
    data_size = int(np.sqrt(hyp_v.shape[1]))
    print("data_size:", data_size)
    print("divergence check:",np.arctan(hyp_v[2,0]/hyp_v[0,0]) , np.arctan(hyp_v[2,-1]/hyp_v[0,-1]))

    wolter1 = np.loadtxt(r"C:\Users\K.Hanada\Desktop\AKBRaytracing\wolter1_rotated_before_offset.csv", delimiter=',', skiprows=1)

    # # ### ４列データの確認
    # # wolter3 = pd.read_csv(r"C:\Users\K.Hanada\Dropbox\EUV_Kpro\20250612設計パラメータ\確定\wolterIII\光線追跡データ\wolter3_mirrorPosition.csv", header=None)
    # # wolter3 = wolter3.to_numpy()
    # # side_wolter3_hyp_x = wolter3[:,2]
    # # side_wolter3_hyp_y = wolter3[:,3]
    # # side_wolter3_ell_x = wolter3[:,0]
    # # side_wolter3_ell_y = wolter3[:,1]
    
    # # ### nanを除去
    # # side_wolter3_hyp_x = side_wolter3_hyp_x[~np.isnan(side_wolter3_hyp_x)]
    # # side_wolter3_hyp_y = side_wolter3_hyp_y[~np.isnan(side_wolter3_hyp_y)]
    # # side_wolter3_ell_x = side_wolter3_ell_x[~np.isnan(side_wolter3_ell_x)]
    # # side_wolter3_ell_y = side_wolter3_ell_y[~np.isnan(side_wolter3_ell_y)]

    # # divergence_mirrorpos = np.arctan(side_wolter3_hyp_y[0]  / side_wolter3_hyp_x[0]) - np.arctan(side_wolter3_hyp_y[-1]  / side_wolter3_hyp_x[-1])
    # # print("divergence_mirrorpos (rad):", divergence_mirrorpos)
    # # divergence_ray = np.arctan(hyp_v[2,0]/hyp_v[0,0]) - np.arctan(hyp_v[2,-1]/hyp_v[0,-1])
    # # print("divergence_ray (rad):", divergence_ray)


    # # rot_angle_wolter3_hyp = np.arctan((side_wolter3_hyp_y[-1]-side_wolter3_hyp_y[0]) / (side_wolter3_hyp_x[-1]-side_wolter3_hyp_x[0]))
    # # rot_angle_wolter3_ell = np.arctan((side_wolter3_ell_y[-1]-side_wolter3_ell_y[0]) / (side_wolter3_ell_x[-1]-side_wolter3_ell_x[0]))
    # # wolter3_hyp_rotated_x, wolter3_hyp_rotated_y = rotation_2D(side_wolter3_hyp_x, side_wolter3_hyp_y, -rot_angle_wolter3_hyp)
    # # wolter3_ell_rotated_x, wolter3_ell_rotated_y = rotation_2D(side_wolter3_ell_x, side_wolter3_ell_y, -rot_angle_wolter3_ell)
    # # wolter3_hyp_rotated_x -= np.min(wolter3_hyp_rotated_x)
    # # wolter3_ell_rotated_x -= np.min(wolter3_ell_rotated_x)

    # # wolter3_hyp_rotated_y -= np.max(wolter3_hyp_rotated_y)
    # # wolter3_ell_rotated_y = -wolter3_ell_rotated_y
    # # wolter3_ell_rotated_y -= np.max(wolter3_ell_rotated_y)
    # # plt.figure()
    # # plt.plot(wolter3_hyp_rotated_x*1e3, wolter3_hyp_rotated_y*1e3, label="side_wolter3_hyp")
    # # plt.plot(wolter3_ell_rotated_x*1e3, wolter3_ell_rotated_y*1e3, label="side_wolter3_ell")
    # # plt.xlabel("x (mm)")
    # # plt.ylabel("y (mm)")
    # # plt.show()
    # vmirr_hyp_optaxis = np.load(r"vmirr_hyp_optaxis.npy")
    # vmirr_ell_optaxis = np.load(r"vmirr_ell_optaxis.npy")
    # vmirr_hyp_optaxis[0,:] -= 146
    # vmirr_ell_optaxis[0,:] -= 146
    # vmirr_hyp_optaxis[2,:] = -vmirr_hyp_optaxis[2,:]
    # vmirr_ell_optaxis[2,:] = -vmirr_ell_optaxis[2,:]
    # plt.figure()
    # plt.plot(vmirr_hyp_optaxis[0,:]*1e3, vmirr_hyp_optaxis[2,:]*1e3, label="hyp_v_optaxis before rotation")
    # plt.plot(vmirr_ell_optaxis[0,:]*1e3, vmirr_ell_optaxis[2,:]*1e3, label="ell_v_optaxis before rotation")
    # plt.xlabel("x (mm)")
    # plt.ylabel("z (mm)")
    # rot_hypv_opt = np.arctan((vmirr_hyp_optaxis[2,-1]-vmirr_hyp_optaxis[2,0]) / (vmirr_hyp_optaxis[0,-1]-vmirr_hyp_optaxis[0,0]))
    # rot_ellv_opt = np.arctan((vmirr_ell_optaxis[2,-1]-vmirr_ell_optaxis[2,0]) / (vmirr_ell_optaxis[0,-1]-vmirr_ell_optaxis[0,0]))
    # hyp_v_sidevie_opt = np.vstack((vmirr_hyp_optaxis[0,:], vmirr_hyp_optaxis[2,:]))
    # ell_v_sidevie_opt = np.vstack((vmirr_ell_optaxis[0,:], vmirr_ell_optaxis[2,:]))
    # hyp_v_rotated_x_opt, hyp_v_rotated_y_opt = rotation_2D(hyp_v_sidevie_opt[0,:], hyp_v_sidevie_opt[1,:], -rot_hypv_opt)
    # ell_v_rotated_x_opt, ell_v_rotated_y_opt = rotation_2D(ell_v_sidevie_opt[0,:], ell_v_sidevie_opt[1,:], -rot_ellv_opt)
    
    # hyp_v_rotated_x_opt -=np.min(hyp_v_rotated_x_opt)
    # ell_v_rotated_x_opt -=np.min(ell_v_rotated_x_opt)
    # hyp_v_rotated_y_opt = -hyp_v_rotated_y_opt
    # hyp_v_rotated_y_opt -=np.max(hyp_v_rotated_y_opt)
    # # ell_v_rotated_y_opt = -ell_v_rotated_y_opt
    # ell_v_rotated_y_opt -=np.max(ell_v_rotated_y_opt)


    hyp_v[0,:] -= 146
    ell_v[0,:] -= 146
    hyp_h[0,:] -= 146
    ell_h[0,:] -= 146
    hyp_v[2,:] = -hyp_v[2,:]
    ell_v[2,:] = -ell_v[2,:]
    hyp_h[2,:] = -hyp_h[2,:]
    ell_h[2,:] = -ell_h[2,:]

    # rot_hypv = np.arctan((hyp_v[2, -1] - hyp_v[2,0]) / (hyp_v[0,-1] - hyp_v[0,0]))
    # rot_ellv = np.arctan((ell_v[2, -1] - ell_v[2,0]) / (ell_v[0,-1] - ell_v[0,0]))
    # print("rot_hypv:", rot_hypv)
    # print("rot_ellv:", rot_ellv)
    # hypv_u = hyp_v[:,0]
    # hypv_l = hyp_v[:,-1]
    # ellv_u = ell_v[:,0]
    # ellv_l = ell_v[:,-1]
    # print("hypv_u x,y,z (mm):", hypv_u*1e3)
    # print("hypv_l x,y,z (mm):", hypv_l*1e3)
    # print("ellv_u x,y,z (mm):", ellv_u*1e3)
    # print("ellv_l x,y,z (mm):", ellv_l*1e3)

    # angle_ray_1to2_u = np.arctan((ellv_u[2]-hypv_u[2]) / (ellv_u[0]-hypv_u[0]))
    # angle_ray_1to2_l = np.arctan((ellv_l[2]-hypv_l[2]) / (ellv_l[0]-hypv_l[0]))
    # print("angle_ray_1to2_u (rad):", angle_ray_1to2_u)
    # print("angle_ray_1to2_l (rad):", angle_ray_1to2_l)
    # hyp_v_sidevie = np.vstack((hyp_v[0,:], hyp_v[2,:]))
    # ell_v_sidevie = np.vstack((ell_v[0,:], ell_v[2,:]))
    # print("hyp_v_sidevie.shape:", hyp_v_sidevie.shape)
    
    # ### dartasizeで間引き
    # hyp_v_sidevie = hyp_v_sidevie[:,data_size//2::data_size]
    # ell_v_sidevie = ell_v_sidevie[:,data_size//2::data_size]

    # plt.figure()
    # plt.plot(hyp_v_sidevie[0,:]*1e3, hyp_v_sidevie[1,:]*1e3, label="hyp_v_sidevie before rotation")
    # plt.plot(ell_v_sidevie[0,:]*1e3, ell_v_sidevie[1,:]*1e3, label="ell_v_sidevie before rotation")
    # plt.xlabel("x (mm)")
    # plt.ylabel("y (mm)")
    # plt.title("Side View Before Rotation")
    # plt.legend()

    # hyp_v_rotated_x, hyp_v_rotated_y = rotation_2D(hyp_v_sidevie[0,:], hyp_v_sidevie[1,:], -rot_hypv)
    # ell_v_rotated_x, ell_v_rotated_y = rotation_2D(ell_v_sidevie[0,:], ell_v_sidevie[1,:], -rot_ellv)
    # hyp_v_rotated_x -=np.min(hyp_v_rotated_x)
    # ell_v_rotated_x -=np.min(ell_v_rotated_x)
    # hyp_v_rotated_y = -hyp_v_rotated_y
    # hyp_v_rotated_y -=np.max(hyp_v_rotated_y)
    # # ell_v_rotated_y = -ell_v_rotated_y
    # ell_v_rotated_y -=np.max(ell_v_rotated_y)
    # print("After rotation:")
    # print("effective length hyp_v_rotated (mm):", (np.max(hyp_v_rotated_x)-np.min(hyp_v_rotated_x))*1e3)
    # print("effective length ell_v_rotated (mm):", (np.max(ell_v_rotated_x)-np.min(ell_v_rotated_x))*1e3)
    # print("shape pv hyp_v_rotated (mm):", (np.max(hyp_v_rotated_y)-np.min(hyp_v_rotated_y))*1e3)
    # print("shape pv ell_v_rotated (mm):", (np.max(ell_v_rotated_y)-np.min(ell_v_rotated_y))*1e3)

    # np.savetxt("hyp_v_rotated53.csv", np.vstack((hyp_v_rotated_x, hyp_v_rotated_y)).T*1e3, fmt="%.6f", header="x(mm),y(mm)",delimiter=',')
    # np.savetxt("ell_v_rotated53.csv", np.vstack((ell_v_rotated_x, ell_v_rotated_y)).T*1e3, fmt="%.6f", header="x(mm),y(mm)",delimiter=',')
    # # np.savetxt("hyp_v_rotated_optaxis.csv", np.vstack((hyp_v_rotated_x_opt, hyp_v_rotated_y_opt)).T*1e3, fmt="%.6f", header="x(mm),y(mm)",delimiter=',')
    # # np.savetxt("ell_v_rotated_optaxis.csv", np.vstack((ell_v_rotated_x_opt, ell_v_rotated_y_opt)).T*1e3, fmt="%.6f", header="x(mm),y(mm)",delimiter=',')
    # plt.figure()
    # plt.plot(hyp_v_rotated_x*1e3, hyp_v_rotated_y*1e3, label="hyp_v_rotated")
    # plt.plot(ell_v_rotated_x*1e3, ell_v_rotated_y*1e3, label="ell_v_rotated")
    # # plt.plot(hyp_v_rotated_x_opt*1e3, hyp_v_rotated_y_opt*1e3, label="hyp_v_rotated_optaxis", linestyle='dashed')
    # # plt.plot(ell_v_rotated_x_opt*1e3, ell_v_rotated_y_opt*1e3, label="ell_v_rotated_optaxis", linestyle='dashed')
    # plt.xlabel("x (mm)")
    # plt.ylabel("y (mm)")
    # plt.legend()
    # print("hyp_v_rotated length (mm):", (np.max(hyp_v_rotated_x)-np.min(hyp_v_rotated_x))*1e3)
    # print("ell_v_rotated length (mm):", (np.max(ell_v_rotated_x)-np.min(ell_v_rotated_x))*1e3)
    # print("hyp_v_rotated datapitch_max (mm):", np.max(np.diff(hyp_v_rotated_x))*1e3)
    # print("ell_v_rotated datapitch_max (mm):", np.max(np.diff(ell_v_rotated_x))*1e3)

    hyp_h_u_u = hyp_h[:,0] # right up
    hyp_h_l_u = hyp_h[:, -data_size] # right down
    ell_h_u_u = ell_h[:,0] # left up
    ell_h_l_u = ell_h[:, -data_size] # left down


    yaw = -1.05000000e-02+0.25302868738419665
    roll= 3.59399021e-05

    # wolter1_normal = normal_from_quad(hyp_h_u_u, hyp_h_l_u, ell_h_l_u, ell_h_u_u)

    center = (hyp_h_u_u + hyp_h_l_u + ell_h_l_u + ell_h_u_u) /4
    # hyp_h_rotated = point_rotate_x(hyp_h, -roll, center)
    # ell_h_rotated = point_rotate_x(ell_h, -roll, center)
    hyp_h_rotated = point_rotate_y(hyp_h, -yaw, center)
    ell_h_rotated = point_rotate_y(ell_h, -yaw, center)
    hyp_h_rotated = point_rotate_x(hyp_h_rotated, roll, center)
    ell_h_rotated = point_rotate_x(ell_h_rotated, roll, center)
    

    # ref_n = np.array([0,0,1])
    # hyp_h_rotated = rotate_point_cloud(hyp_h, ref_n, wolter1_normal)
    # ell_h_rotated = rotate_point_cloud(ell_h, ref_n, wolter1_normal)
    # # 確認：回転後の法線方向をチェック
    # new_normal = rotate_point_cloud(ref_n.reshape(3,1), ref_n, wolter1_normal).ravel()

    hyp_h_rotated_side = np.vstack((hyp_h_rotated[0,:], hyp_h_rotated[1,:]))
    ell_h_rotated_side = np.vstack((ell_h_rotated[0,:], ell_h_rotated[1,:]))
    
    rot_hyp_h_side = np.arctan((np.max(hyp_h_rotated_side[1,:]) - np.min(hyp_h_rotated_side[1,:])) / (np.max(hyp_h_rotated_side[0,:]) - np.min(hyp_h_rotated_side[0,:])))
    rot_ell_h_side = np.arctan((np.max(ell_h_rotated_side[1,:]) - np.min(ell_h_rotated_side[1,:])) / (np.max(ell_h_rotated_side[0,:]) - np.min(ell_h_rotated_side[0,:])))
    print("rot_hyp_h_side (rad):", rot_hyp_h_side)
    print("rot_ell_h_side (rad):", rot_ell_h_side)
    rot_wolter_side = (-rot_hyp_h_side - rot_ell_h_side) /2
    hyp_h_side_x, hyp_h_side_y = rotation_2D(hyp_h_rotated_side[0,:], hyp_h_rotated_side[1,:], -rot_wolter_side)
    ell_h_side_x, ell_h_side_y = rotation_2D(ell_h_rotated_side[0,:], ell_h_rotated_side[1,:], -rot_wolter_side)
    hyp_h_side_y = -hyp_h_side_y
    ell_h_side_y = -ell_h_side_y
    offset_x = np.min(ell_h_side_x)
    offset_y = np.max(ell_h_side_y)

    ell_h_side_x -= offset_x
    hyp_h_side_x -= offset_x
    ell_h_side_y -= offset_y
    hyp_h_side_y -= offset_y
    wolter1[:,0] -= np.min(wolter1[:,0])
    # print("回転後法線 (should be close to tgt_n):", new_normal)
    # print("回転後点群:", hyp_h_rotated[:,0])
    hyp_h_side_x_u = hyp_h_side_x[:data_size]
    hyp_h_side_y_u = hyp_h_side_y[:data_size:]
    ell_h_side_x_u = ell_h_side_x[:data_size]
    ell_h_side_y_u = ell_h_side_y[:data_size]
    print("length hyp_h_side_x (mm):", (np.max(hyp_h_side_x)-np.min(ell_h_side_x))*1e3)
    print("length wolter1 (mm):", (np.max(wolter1[:,0])-np.min(wolter1[:,0]))*1e3)
    plt.figure()
    plt.plot(hyp_h_side_x*1e3, hyp_h_side_y*1e3, label="hyp_h_rotated side view")
    plt.plot(ell_h_side_x*1e3, ell_h_side_y*1e3, label="ell_h_rotated side view")
    plt.plot(hyp_h_side_x_u*1e3, hyp_h_side_y_u*1e3, marker='o', label="hyp_h_rotated side view upper edge")
    plt.plot(ell_h_side_x_u*1e3, ell_h_side_y_u*1e3, marker='o', label="ell_h_rotated side view upper edge")
    
    plt.plot(wolter1[:,0]*1e3, wolter1[:,1]*1e3, label="wolter1 from csv")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")



    plt.figure()
    plt.plot(hyp_h_rotated[0,:]*1e3, hyp_h_rotated[2,:]*1e3, label="hyp_h_rotated")
    plt.plot(ell_h_rotated[0,:]*1e3, ell_h_rotated[2,:]*1e3, label="ell_h_rotated")
    plt.xlabel("x (mm)")
    plt.ylabel("z (mm)")
    plt.title("Top View After Rotation")
    plt.legend()
    # plt.savefig("surface_topview.png", dpi=300, transparent=True)
    plt.figure()
    plt.plot(hyp_h_rotated[0,:]*1e3, hyp_h_rotated[1,:]*1e3, label="hyp_h_rotated")
    plt.plot(ell_h_rotated[0,:]*1e3, ell_h_rotated[1,:]*1e3, label="ell_h_rotated")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Side View After Rotation")
    plt.legend()
    # plt.savefig("surface_sideview.png", dpi=300, transparent=True)
    
    plt.figure()
    plt.plot(hyp_v[0,:]*1e3, hyp_v[1, :]*1e3, label="hyp_v")
    plt.plot(ell_v[0,:]*1e3, ell_v[1, :]*1e3, label="ell_v")
    plt.plot(hyp_h[0,:]*1e3, hyp_h[1, :]*1e3, label="hyp_h")
    plt.plot(ell_h[0,:]*1e3, ell_h[1, :]*1e3, label="ell_h")
    plt.plot(hyp_h_u_u[0]*1e3, hyp_h_u_u[1]*1e3,marker='o', label="hyp_h_u_u")
    plt.plot(hyp_h_l_u[0]*1e3, hyp_h_l_u[1]*1e3,marker='o', label="hyp_h_l_u")
    plt.plot(ell_h_u_u[0]*1e3, ell_h_u_u[1]*1e3,marker='o', label="ell_h_u_u")
    plt.plot(ell_h_l_u[0]*1e3, ell_h_l_u[1]*1e3,marker='o', label="ell_h_l_u")
    # plt.plot(center[0]*1e3, center[1]*1e3, marker='x',label="center")
    # plt.plot([center[0]*1e3, center[0]*1e3 + wolter1_normal[0]*10], [center[1]*1e3, center[1]*1e3 + wolter1_normal[1]*10], label="normal vector")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Surface Detail AKB")
    plt.legend()
    plt.axis("equal")
    plt.savefig("surface_xy.png", dpi=300, transparent=True)

    plt.figure()
    plt.plot(hyp_v[0,:]*1e3, hyp_v[2, :]*1e3, label="hyp_v")
    plt.plot(ell_v[0,:]*1e3, ell_v[2, :]*1e3, label="ell_v")
    plt.plot(hyp_h[0,:]*1e3, hyp_h[2, :]*1e3, label="hyp_h")
    plt.plot(ell_h[0,:]*1e3, ell_h[2, :]*1e3, label="ell_h")
    plt.plot(hyp_h_u_u[0]*1e3, hyp_h_u_u[2]*1e3,marker='o', label="hyp_h_u_u")
    plt.plot(hyp_h_l_u[0]*1e3, hyp_h_l_u[2]*1e3,marker='o', label="hyp_h_l_u")
    plt.plot(ell_h_u_u[0]*1e3, ell_h_u_u[2]*1e3,marker='o', label="ell_h_u_u")
    plt.plot(ell_h_l_u[0]*1e3, ell_h_l_u[2]*1e3,marker='o', label="ell_h_l_u")
    # plt.plot(center[0]*1e3, center[2]*1e3, marker='x',label="center")
    # plt.plot([center[0]*1e3, center[0]*1e3 + wolter1_normal[0]*10], [center[2]*1e3, center[2]*1e3 + wolter1_normal[2]*10], label="normal vector")
    plt.xlabel("x (mm)")
    plt.ylabel("z (mm)")
    plt.title("Surface Detail AKB")
    plt.legend()
    plt.axis("equal")
    plt.savefig("surface_xz.png", dpi=300, transparent=True)
    plt.show()