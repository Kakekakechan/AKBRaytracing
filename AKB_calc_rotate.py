import numpy as np
from numpy import sin, cos, tan, arctan, sqrt, arcsin, arccos, pi
import matplotlib.pyplot as plt
def Yvalue_hyperbola(a, b, x0, x):
    return b * np.sqrt(((x - x0) / a) ** 2 - 1)
def Yvalue_ellipse(a, b, x0, x):
    return b * np.sqrt(1 - ((x - x0) / a) ** 2)
### 回転の定義
def rotation_2D(x, y, angle_rad):
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rot, y_rot
def calc_theta3(x,y,a):
    return arcsin(y/(sqrt(x**2 + y**2)-2*a))

def extrapolate_parabola(x0,y0,x1,y1,dydx1,xx):
    a = dydx1/(2*(x1-x0))
    yy = a*(xx - x0)**2 - a*(x1 - x0)**2 + y1
    return yy
def extrapolate_linear(x0,y0,x1,y1,dydx1,xx):
    a = dydx1
    yy = a*(xx - x1) + y1
    return yy
def calc_ell_theta(a,b,theta3,x0):
    A = (cos(theta3)**2)/a**2 + (sin(theta3)**2)/b**2
    B = -2*x0*cos(theta3)/a**2
    C = (x0**2)/a**2 - 1
    D = B**2 - 4*A*C
    if D.any() < 0:
        print("Error in calc_ell_theta: D<0")
        return None
    x1 = (-B + sqrt(D)) / (2*A)
    x2 = (-B - sqrt(D)) / (2*A)
    return x1, x2

def calc_hyp_theta(a,b,theta3,x0):
    A = (cos(theta3)**2)/a**2 - (sin(theta3)**2)/b**2
    B = -2*x0*cos(theta3)/a**2
    C = (x0**2)/a**2 - 1
    D = B**2 - 4*A*C
    if D.any() < 0:
        print("Error in calc_hyp_theta: D<0")
        return None
    x1 = (-B + sqrt(D)) / (2*A)
    x2 = (-B - sqrt(D)) / (2*A)
    return x1, x2
def interp_nan(arr):
    """
    1 次元の NumPy 配列 arr の NaN を線形補間で埋める。
    端点が NaN の場合は、最初／最後の非 NaN 値で埋める。
    """
    arr = arr.copy()
    n = arr.size
    idx = np.arange(n)

    # NaN のマスク
    mask_nan = np.isnan(arr)
    mask_notnan = ~mask_nan

    if mask_notnan.sum() == 0:
        # 全部 NaN の場合はそのまま返す
        return arr

    # 端点処理：先頭／末尾が NaN の場合は最初／最後の非 NaN 値で埋める
    first_valid = idx[mask_notnan][0]
    last_valid  = idx[mask_notnan][-1]
    arr[:first_valid] = arr[first_valid]
    arr[last_valid+1:] = arr[last_valid]

    # 中間の NaN を線形補間
    mask_nan = np.isnan(arr)          # 更新したマスク
    arr[mask_nan] = np.interp(
        idx[mask_nan],
        idx[~mask_nan],
        arr[~mask_nan]
    )
    return arr
def merge_array(x_array, y_array,x_array1, y_array1,x_array2, y_array2):
    xmin = min(np.nanmin(x_array), np.nanmin(x_array1), np.nanmin(x_array2))
    xmax = max(np.nanmax(x_array), np.nanmax(x_array1), np.nanmax(x_array2))
    datapitch = x_array[1]-x_array[0]
    x_merged = np.arange(xmin, xmax, datapitch)
    y_merged = np.full_like(x_merged, np.nan)
    for i, xx in enumerate(x_merged):
        y_values = []
        for x_arr, y_arr in [(x_array, y_array), (x_array1, y_array1), (x_array2, y_array2)]:
            if xx >= np.nanmin(x_arr) and xx <= np.nanmax(x_arr):
                index = (np.abs(x_arr - xx)).argmin()
                y_values.append(y_arr[index])
        if y_values:
            y_merged[i] = max(y_values)
        else:
            print(f"Warning: No data for x={xx}")
    if np.isnan(y_merged).sum() > 0:
        print("Merging resulted in NaN values, applying interpolation.")
        y_merged_before = y_merged.copy()
        y_merged = interp_nan(y_merged)
        print(f"Number of NaNs before interpolation: {np.isnan(y_merged_before).sum()}, after interpolation: {np.isnan(y_merged).sum()}")
        print("datapitch:", datapitch)
        plt.figure(figsize=(10,5))
        plt.plot(x_merged, y_merged_before, label='Before Interpolation')
        plt.plot(x_merged, y_merged, label='After Interpolation', linestyle='dashed')
        plt.legend()
        plt.show()
    return x_merged, y_merged
if __name__ == "__main__":
    ### 3型 Setting12
    a_hyp_v  =   np.float64(72.9825)
    b_hyp_v  =   np.float64(0.263879113520857)
    f_hyp_v  =   sqrt(a_hyp_v**2 + b_hyp_v**2)
    a_ell_v  =   np.float64(0.1175)
    b_ell_v  =   np.float64(0.0283168369674688)
    f_ell_v  =   sqrt(a_ell_v**2 - b_ell_v**2)
    hyp_length_v  =   np.float64(0.043)
    ell_length_v  =   np.float64(0.0809220387326922)
    theta1_v  =   np.float64(5.55983241203018E-05)
    theta2_v  =   np.float64(0.117)
    theta3_v  =   np.float64(0.23405559832412)
    theta4_v  =   np.float64(0.243572583671924)
    theta5_v  =   np.float64(-0.253089569019727)

    num=10000
    pre_margin_l = 0.012
    pre_margin_u = 0.02

    margin_u = 0.001
    margin_l = 0.015
    margin_u_extra = 0.014
    height_margin = 0.5


    datapitch = 0.1/10000  # 10um

    l1_plus_l2_prime = 2*a_hyp_v
    l1_times_l2_prime = -b_hyp_v**2/sin(theta2_v)**2
    l1_v = (l1_plus_l2_prime + sqrt(l1_plus_l2_prime**2 - 4*l1_times_l2_prime))/2
    l2_v = -l1_plus_l2_prime + l1_v
    print("l1_v, l2'_v", l1_v, l2_v)
    xc_hyp_v = l1_v * cos(theta1_v)
    yc_hyp_v = l1_v * sin(theta1_v)
    print("xc_hyp_v, yc_hyp_v (mm):", xc_hyp_v*1e3, yc_hyp_v*1e3) ### correct above here

    xx_hyp_v = np.linspace(-hyp_length_v/2, hyp_length_v/2, num)+xc_hyp_v
    yy_hyp_v = Yvalue_hyperbola(a_hyp_v, b_hyp_v, f_hyp_v, xx_hyp_v)
    print("xx_hyp_v min/max (mm):", np.min(xx_hyp_v)*1e3, np.max(xx_hyp_v)*1e3)
    print("yy_hyp_v min/max (mm):", np.min(yy_hyp_v)*1e3, np.max(yy_hyp_v)*1e3)
    plt.figure(figsize=(10,5))
    plt.plot(xx_hyp_v*1e3, yy_hyp_v*1e3)
    plt.title("hyp_v before rotation")
    plt.xlabel("Position (mm)")
    plt.ylabel("Shape (mm)")### correct above here
    # plt.show()
    print('divergence_mirrorpos (rad):', np.arctan(yy_hyp_v[0]/xx_hyp_v[0]) - np.arctan(yy_hyp_v[-1]/xx_hyp_v[-1]))
    rot_hyp_v = -arctan((yy_hyp_v[-1]-yy_hyp_v[0]) / (xx_hyp_v[-1]-xx_hyp_v[0]))
    xx_hyp_v_roted, yy_hyp_v_roted = rotation_2D(xx_hyp_v, yy_hyp_v, rot_hyp_v)
    print("effective length hyp_v (mm):", (np.max(xx_hyp_v_roted)-np.min(xx_hyp_v_roted))*1e3)
    print("shape pv hyp_v (mm):", (np.max(yy_hyp_v_roted)-np.min(yy_hyp_v_roted))*1e3)
    plt.figure(figsize=(10,5))
    plt.plot(xx_hyp_v_roted*1e3, yy_hyp_v_roted*1e3)
    plt.show()

    min_xx_hyp_v_roted = np.nanmin(xx_hyp_v_roted)
    max_xx_hyp_v_roted = np.nanmax(xx_hyp_v_roted)

    xx_hyp_v_margin = np.linspace(-hyp_length_v/2-pre_margin_l, hyp_length_v/2+pre_margin_u, num)+xc_hyp_v
    yy_hyp_v_margin = Yvalue_hyperbola(a_hyp_v, b_hyp_v, f_hyp_v, xx_hyp_v_margin)

    xx_hyp_v_roted_margin, yy_hyp_v_roted_margin = rotation_2D(xx_hyp_v_margin, yy_hyp_v_margin, rot_hyp_v)

    print("data pitch", xx_hyp_v_roted_margin[1]-xx_hyp_v_roted_margin[0])
    x_offset = (xx_hyp_v_roted[0]+xx_hyp_v_roted[-1])/2
    y_offset = np.nanmax(yy_hyp_v_roted)
    xx_hyp_v_roted -= x_offset
    yy_hyp_v_roted -= y_offset
    xx_hyp_v_roted_margin -= x_offset
    yy_hyp_v_roted_margin -= y_offset

    np.savetxt("hyp_v_rotated_before_offset.csv", np.column_stack((xx_hyp_v_roted*1e3, yy_hyp_v_roted*1e3)), fmt="%.6f", header="x(mm),y(mm)",delimiter=',')

    xx_hyp_v_roted_min = np.nanmin(xx_hyp_v_roted)
    xx_hyp_v_roted_max = np.nanmax(xx_hyp_v_roted)

    xx_hyp_v_resampled = np.arange(round(np.min(xx_hyp_v_roted_margin), 5), round(np.max(xx_hyp_v_roted_margin), 5), datapitch)
    yy_hyp_v_resampled = np.interp(xx_hyp_v_resampled, xx_hyp_v_roted_margin, yy_hyp_v_roted_margin)

    mask = (xx_hyp_v_resampled > (xx_hyp_v_roted_min-margin_u)) & (xx_hyp_v_resampled < (xx_hyp_v_roted_max+margin_l))
    xx_hyp_v_resampled = xx_hyp_v_resampled[mask]
    yy_hyp_v_resampled = yy_hyp_v_resampled[mask]

    xx_hyp_v_extrapolated_u = np.arange(np.min(xx_hyp_v_resampled)-margin_u_extra, np.min(xx_hyp_v_resampled), datapitch)
    dydx1 = (yy_hyp_v_resampled[1]-yy_hyp_v_resampled[0])/(xx_hyp_v_resampled[1]-xx_hyp_v_resampled[0])

    # yy_hyp_v_extrapolated_u_linear = extrapolate_linear(xx_hyp_v_resampled[0]-margin_u_extra, yy_hyp_v_resampled[0]-height_margin, xx_hyp_v_resampled[0], yy_hyp_v_resampled[0], dydx1, xx_hyp_v_extrapolated_u)
    yy_hyp_v_extrapolated_u_parabola = extrapolate_parabola(xx_hyp_v_resampled[0]-margin_u_extra, yy_hyp_v_resampled[0]-height_margin, xx_hyp_v_resampled[0], yy_hyp_v_resampled[0], dydx1, xx_hyp_v_extrapolated_u)

    plt.figure(figsize=(10,5))
    plt.plot(xx_hyp_v_roted*1e3, yy_hyp_v_roted*1e3)
    plt.plot(xx_hyp_v_resampled*1e3, yy_hyp_v_resampled*1e3, linestyle='dashed')
    # plt.plot(xx_hyp_v_extrapolated_u*1e3, yy_hyp_v_extrapolated_u_linear*1e3, linestyle='dashed')
    plt.plot(xx_hyp_v_extrapolated_u*1e3, yy_hyp_v_extrapolated_u_parabola*1e3, linestyle='dashed')
    plt.title(f"hyp_v rot_hyp_v={np.degrees(rot_hyp_v):.2f} deg")
    plt.xlabel("Position (mm)")
    plt.ylabel("Shape (mm)")
    plt.savefig("hyp_v.png")
    # plt.show()

    hyp_v_merge_x, hyp_v_merge_y = merge_array(xx_hyp_v_resampled, yy_hyp_v_resampled,
                                    xx_hyp_v_extrapolated_u, yy_hyp_v_extrapolated_u_parabola,
                                    xx_hyp_v_resampled, yy_hyp_v_resampled)

    datapitch_interp = 0.1  # 0.1mm
    hyp_v_merge_x_interp = np.arange(np.nanmin(hyp_v_merge_x*1e3), np.nanmax(hyp_v_merge_x*1e3), datapitch_interp)
    hyp_v_merge_y_interp = np.interp(hyp_v_merge_x_interp, hyp_v_merge_x*1e3, hyp_v_merge_y*1e3)
    print("hyp_v_merge_x_interp nan index", np.isnan(hyp_v_merge_x_interp).sum())
    print("hyp_v_merge_y_interp nan index", np.isnan(hyp_v_merge_y_interp).sum())
    np.savetxt("hyp_v_rotated_0.1mmpitch.csv", np.column_stack((hyp_v_merge_x_interp, hyp_v_merge_y_interp)), fmt="%.6f", header="x(mm),y(mm)",delimiter=',')
    # plt.figure(figsize=(10,5))
    plt.plot(hyp_v_merge_x*1e3, hyp_v_merge_y*1e3)
    plt.plot(hyp_v_merge_x_interp, hyp_v_merge_y_interp, linestyle='dashed')
    plt.title(f"hyp_v merged rot_hyp_v={np.degrees(rot_hyp_v):.2f} deg")
    plt.xlabel("Position (mm)")
    plt.ylabel("Shape (mm)")
    plt.show()


    margin_u = 0.001
    margin_l = 0.001
    margin_u_extra = 0.014
    margin_l_extra = 0.014

    theta3_v_array = calc_theta3(xx_hyp_v, yy_hyp_v, a_hyp_v)
    theta3_v_array_margin = calc_theta3(xx_hyp_v_margin, yy_hyp_v_margin, a_hyp_v)
    print("theta3_v calced", theta3_v_array)

    l3_array,_ = calc_ell_theta(a_ell_v, b_ell_v, theta3_v_array, f_ell_v)
    l3_array_margin,_ = calc_ell_theta(a_ell_v, b_ell_v, theta3_v_array_margin, f_ell_v)

    xx_ell_v = l3_array*cos(theta3_v_array)
    yy_ell_v = l3_array*sin(theta3_v_array)
    rot_ell_v = -arctan((yy_ell_v[-1]-yy_ell_v[0]) / (xx_ell_v[-1]-xx_ell_v[0]))
    xx_ell_v_roted, yy_ell_v_roted = rotation_2D(xx_ell_v, yy_ell_v, rot_ell_v)

    xx_ell_v_margin = l3_array_margin*cos(theta3_v_array_margin)
    yy_ell_v_margin = l3_array_margin*sin(theta3_v_array_margin)

    xx_ell_v_roted_margin, yy_ell_v_roted_margin = rotation_2D(xx_ell_v_margin, yy_ell_v_margin, rot_ell_v)

    x_offset = (xx_ell_v_roted[0]+xx_ell_v_roted[-1])/2
    y_offset = np.nanmax(yy_ell_v_roted)

    xx_ell_v_roted -= x_offset
    yy_ell_v_roted -= y_offset
    xx_ell_v_roted_margin -= x_offset
    yy_ell_v_roted_margin -= y_offset

    xx_ell_v_roted_min = np.nanmin(xx_ell_v_roted)
    xx_ell_v_roted_max = np.nanmax(xx_ell_v_roted)

    yy_ell_v_roted = -yy_ell_v_roted  # y軸反転
    yy_ell_v_roted_margin = -yy_ell_v_roted_margin  # y軸反転


    np.savetxt("ell_v_rotated_before_offset.csv", np.column_stack((xx_ell_v_roted*1e3, yy_ell_v_roted*1e3)), fmt="%.6f", header="x(mm),y(mm)",delimiter=',')

    xx_ell_v_resampled = np.arange(round(np.min(xx_ell_v_roted_margin), 5), round(np.max(xx_ell_v_roted_margin), 5), datapitch)
    yy_ell_v_resampled = np.interp(xx_ell_v_resampled, xx_ell_v_roted_margin, yy_ell_v_roted_margin)

    mask = (xx_ell_v_resampled > (xx_ell_v_roted_min-margin_u)) & (xx_ell_v_resampled < (xx_ell_v_roted_max+margin_l))
    xx_ell_v_resampled = xx_ell_v_resampled[mask]
    yy_ell_v_resampled = yy_ell_v_resampled[mask]

    xx_ell_v_extrapolated_u = np.arange(np.min(xx_ell_v_resampled)-margin_u_extra, np.min(xx_ell_v_resampled), datapitch)
    dydx1 = (yy_ell_v_resampled[1]-yy_ell_v_resampled[0])/(xx_ell_v_resampled[1]-xx_ell_v_resampled[0])
    yy_ell_v_extrapolated_u_parabola = extrapolate_parabola(xx_ell_v_resampled[0]-margin_u_extra, yy_ell_v_resampled[0]+height_margin, xx_ell_v_resampled[0], yy_ell_v_resampled[0], dydx1, xx_ell_v_extrapolated_u)

    xx_ell_v_extrapolated_l = np.arange(np.max(xx_ell_v_resampled), np.max(xx_ell_v_resampled)+margin_l_extra, datapitch)
    dydx1 = (yy_ell_v_resampled[-1]-yy_ell_v_resampled[-2])/(xx_ell_v_resampled[-1]-xx_ell_v_resampled[-2])
    yy_ell_v_extrapolated_l_parabola = extrapolate_parabola(xx_ell_v_resampled[-1]+margin_l_extra, yy_ell_v_resampled[-1]+height_margin, xx_ell_v_resampled[-1], yy_ell_v_resampled[-1], dydx1, xx_ell_v_extrapolated_l)


    # print("data pitch", xx_ell_v_resampled[1]-xx_ell_v_resampled[0])
    print("")
    plt.figure(figsize=(10,5))
    plt.plot(xx_ell_v_roted*1e3, yy_ell_v_roted*1e3)
    plt.plot(xx_ell_v_resampled*1e3, yy_ell_v_resampled*1e3, linestyle='dashed')
    plt.plot(xx_ell_v_extrapolated_u*1e3, yy_ell_v_extrapolated_u_parabola*1e3, linestyle='dashed')
    plt.plot(xx_ell_v_extrapolated_l*1e3, yy_ell_v_extrapolated_l_parabola*1e3, linestyle='dashed')
    plt.title(f"ell_v rot_ell_v={np.degrees(rot_ell_v):.2f} deg")
    plt.xlabel("Position (mm)")
    plt.ylabel("Shape (mm)")
    plt.savefig("ell_v.png")
    # plt.show()

    print("ell_v effective length",np.nanmax(xx_ell_v_roted)-np.nanmin(xx_ell_v_roted))
    print("ell_v data length",np.nanmax(xx_ell_v_resampled)-np.nanmin(xx_ell_v_resampled))
    print("ell_v effective position min max",np.nanmin(xx_ell_v_roted), np.nanmax(xx_ell_v_roted))

    ell_v_merge_x, ell_v_merge_y = merge_array(xx_ell_v_resampled, yy_ell_v_resampled,
                                    xx_ell_v_extrapolated_u, yy_ell_v_extrapolated_u_parabola,
                                    xx_ell_v_extrapolated_l, yy_ell_v_extrapolated_l_parabola)
    print("xx_ell_v_resampled.dtype", xx_ell_v_resampled.dtype)
    print("yy_ell_v_resampled.dtype", yy_ell_v_resampled.dtype)
    print("xx_ell_v_extrapolated_u.dtype", xx_ell_v_extrapolated_u.dtype)
    print("yy_ell_v_extrapolated_u_parabola.dtype", yy_ell_v_extrapolated_u_parabola.dtype)
    print("xx_ell_v_extrapolated_l.dtype", xx_ell_v_extrapolated_l.dtype)
    print("yy_ell_v_extrapolated_l_parabola.dtype", yy_ell_v_extrapolated_l_parabola.dtype)
    print("ell_v_merge_x.dtype", ell_v_merge_x.dtype)
    print("ell_v_merge_y.dtype", ell_v_merge_y.dtype)

    datapitch_interp = 0.1  # 0.1mm
    ell_v_merge_x_interp = np.arange(np.nanmin(ell_v_merge_x*1e3), np.nanmax(ell_v_merge_x*1e3), datapitch_interp)
    ell_v_merge_y_interp = np.interp(ell_v_merge_x_interp, ell_v_merge_x*1e3, ell_v_merge_y*1e3)

    print("ell_v_merge_x_interp.dtype", ell_v_merge_x_interp.dtype)
    print("ell_v_merge_y_interp.dtype", ell_v_merge_y_interp.dtype)
    print("ell_v_merge_x_interp nan index", np.isnan(ell_v_merge_x_interp).sum())
    print("ell_v_merge_y_interp nan index", np.isnan(ell_v_merge_y_interp).sum())

    np.savetxt("ell_v_rotated_0.1mmpitch.csv", np.column_stack((ell_v_merge_x_interp, ell_v_merge_y_interp)), fmt="%.6f", header="x(mm),y(mm)",delimiter=',')
    # plt.figure(figsize=(10,5))
    plt.plot(ell_v_merge_x*1e3, ell_v_merge_y*1e3)
    plt.plot(ell_v_merge_x_interp, ell_v_merge_y_interp, linestyle='dashed')
    plt.title(f"ell_v merged rot_ell_v={np.degrees(rot_ell_v):.2f} deg")
    plt.xlabel("Position (mm)")
    plt.ylabel("Shape (mm)")
    plt.show()


    ### 1型 setting11
    a_ell_h  =   np.float64(73.1076714403445)
    b_ell_h  =   np.float64(0.517019631143022)
    f_ell_h  =   sqrt(a_ell_h**2 - b_ell_h**2)
    a_hyp_h  =   np.float64(0.0077)
    b_hyp_h  =   np.float64(0.00432051448679384)
    f_hyp_h  =   sqrt(a_hyp_h**2 + b_hyp_h**2)
    hyp_length_h  =   np.float64(0.01380360633)
    ell_length_h  =   np.float64(0.030)
    theta1_h  =   np.float64(0.000145746388538841)
    theta2_h  =   np.float64(0.17)
    theta3_h  =   np.float64(0.339854253611461)
    theta4_h  =   np.float64(0.182330449161024)
    theta5_h  =   np.float64(0.757889356272919)

    num=100000
    pre_margin_l = 0.5
    pre_margin_u = 0.018

    margin_u = 0.001
    margin_l = 0.001
    margin_u_extra = 0.020
    margin_l_extra = 0.020
    margin = 0.015

    xc_ell_h,_ = calc_ell_theta(a_ell_h, b_ell_h, theta1_h, f_ell_h)
    yc_ell_h = Yvalue_ellipse(a_ell_h, b_ell_h, f_ell_h, xc_ell_h)
    xx_ell_h = np.linspace(-ell_length_h/2, ell_length_h/2, num)+xc_ell_h
    yy_ell_h = Yvalue_ellipse(a_ell_h, b_ell_h, f_ell_h, xx_ell_h)
    xx_ell_h_margin = np.linspace(-ell_length_h/2-pre_margin_l, ell_length_h/2+pre_margin_u, num)+xc_ell_h
    yy_ell_h_margin = Yvalue_ellipse(a_ell_h, b_ell_h, f_ell_h, xx_ell_h_margin)

    l1_h = sqrt((xx_ell_h)**2 + (yy_ell_h)**2)
    theta3_h_array = arccos((2*f_ell_h - xx_ell_h)/(2*a_ell_h - l1_h))
    l1_h_margin = sqrt((xx_ell_h_margin)**2 + (yy_ell_h_margin)**2)
    theta3_h_array_margin = arccos((2*f_ell_h - xx_ell_h_margin)/(2*a_ell_h - l1_h_margin))
    print(f"theta3_h calced: {theta3_h_array[:5]} ... {theta3_h_array[-5:]}")
    l1_h_c = sqrt((xc_ell_h)**2 + (yc_ell_h)**2)
    theta3_h_c = arccos((2*f_ell_h - xc_ell_h)/(2*a_ell_h - l1_h_c))
    print("theta3_h_c calced", theta3_h_c)

    l3_h_array,_ = calc_hyp_theta(a_hyp_h, b_hyp_h, theta3_h_array, f_hyp_h)
    l3_h_c1,l3_h_c2 = calc_hyp_theta(a_hyp_h, b_hyp_h, theta3_h_c, f_hyp_h)
    l3_h_array_margin,_ = calc_hyp_theta(a_hyp_h, b_hyp_h, theta3_h_array_margin, f_hyp_h)
    print("l3_h_c calced", l3_h_c1, l3_h_c2)

    xx_hyp_h = 2*f_ell_h-l3_h_array*cos(theta3_h_array)
    yy_hyp_h = l3_h_array*sin(theta3_h_array)
    xx_hyp_h_margin = 2*f_ell_h-l3_h_array_margin*cos(theta3_h_array_margin)
    yy_hyp_h_margin = l3_h_array_margin*sin(theta3_h_array_margin)

    rot_hyp_h = -arctan((yy_hyp_h[-1]-yy_hyp_h[0]) / (xx_hyp_h[-1]-xx_hyp_h[0]))
    rot_ell_h = -arctan((yy_ell_h[-1]-yy_ell_h[0]) / (xx_ell_h[-1]-xx_ell_h[0]))
    hyplen = xx_hyp_h[0]-xx_hyp_h[-1]
    elllen = xx_ell_h[-1]-xx_ell_h[0]
    print("rot_hyp_h, rot_ell_h", np.degrees(rot_hyp_h), np.degrees(rot_ell_h))
    print("hyplen, elllen", hyplen, elllen)
    rot_wolter1 = (rot_hyp_h + rot_ell_h)/2
    rot_wolter1 = (rot_hyp_h*hyplen + rot_ell_h*elllen)/(hyplen + elllen)*1.2
    # rot_wolter1 = -arctan((yy_hyp_h[0]-yy_ell_h[0]) / (xx_hyp_h[0]-xx_ell_h[0]))
    xx_ell_h_roted, yy_ell_h_roted = rotation_2D(xx_ell_h, yy_ell_h, rot_wolter1)
    xx_hyp_h_roted, yy_hyp_h_roted = rotation_2D(xx_hyp_h, yy_hyp_h, rot_wolter1)

    xx_ell_h_roted_margin, yy_ell_h_roted_margin = rotation_2D(xx_ell_h_margin, yy_ell_h_margin, rot_wolter1)
    xx_hyp_h_roted_margin, yy_hyp_h_roted_margin = rotation_2D(xx_hyp_h_margin, yy_hyp_h_margin, rot_wolter1)

    x_offset = (xx_ell_h_roted[0]+xx_hyp_h_roted[-1])/2
    y_offset = np.nanmin(yy_ell_h_roted)
    xx_ell_h_roted -= x_offset
    yy_ell_h_roted -= y_offset
    xx_hyp_h_roted -= x_offset
    yy_hyp_h_roted -= y_offset
    yy_ell_h_roted = -yy_ell_h_roted  # y軸反転
    yy_hyp_h_roted = -yy_hyp_h_roted  # y軸反転
    xx_ell_h_roted_min = np.nanmin(xx_ell_h_roted)
    xx_ell_h_roted_max = np.nanmax(xx_ell_h_roted)
    xx_hyp_h_roted_min = np.nanmin(xx_hyp_h_roted)
    xx_hyp_h_roted_max = np.nanmax(xx_hyp_h_roted)

    xx_ell_h_roted_margin -= x_offset
    yy_ell_h_roted_margin -= y_offset
    xx_hyp_h_roted_margin -= x_offset
    yy_hyp_h_roted_margin -= y_offset
    yy_ell_h_roted_margin = -yy_ell_h_roted_margin  # y軸反転
    yy_hyp_h_roted_margin = -yy_hyp_h_roted_margin  # y軸反転

    plt.figure()
    plt.plot(xx_ell_h_roted*1e3, yy_ell_h_roted*1e3, label='Ellipse before offset')
    plt.plot(xx_hyp_h_roted*1e3, yy_hyp_h_roted*1e3, label='Hyperbola before offset')
    wolter_data_origin = np.column_stack((np.concatenate((xx_ell_h_roted, xx_hyp_h_roted)), np.concatenate((yy_ell_h_roted, yy_hyp_h_roted))))
    np.savetxt("wolter1_rotated_before_offset.csv", wolter_data_origin, fmt="%.6f", header="x(mm),y(mm)",delimiter=',')
    plt.title(f"1st type Wolter rot_wolter1={np.degrees(rot_wolter1):.2f} deg")
    plt.xlabel("Position (mm)")
    plt.ylabel("Shape (mm)")
    plt.legend()
    plt.show()

    xx_ell_h_resampled = np.arange(round(np.min(xx_ell_h_roted_margin), 5), round(np.max(xx_hyp_h_roted_margin), 5), datapitch)
    yy_ell_h_resampled = np.interp(xx_ell_h_resampled, xx_ell_h_roted_margin, yy_ell_h_roted_margin)
    xx_hyp_h_resampled = np.arange(round(np.min(xx_ell_h_roted_margin), 5), round(np.max(xx_hyp_h_roted_margin), 5), datapitch)
    yy_hyp_h_resampled = np.interp(xx_hyp_h_resampled, xx_hyp_h_roted_margin[::-1], yy_hyp_h_roted_margin[::-1])

    mask_ell = (xx_ell_h_resampled > (xx_ell_h_roted_min-margin_u)) & (xx_ell_h_resampled < (xx_ell_h_roted_max+margin))
    mask_hyp = (xx_hyp_h_resampled > (xx_hyp_h_roted_min-margin)) & (xx_hyp_h_resampled < (xx_hyp_h_roted_max+margin_l))
    xx_ell_h_resampled = xx_ell_h_resampled[mask_ell]
    yy_ell_h_resampled = yy_ell_h_resampled[mask_ell]
    xx_hyp_h_resampled = xx_hyp_h_resampled[mask_hyp]
    yy_hyp_h_resampled = yy_hyp_h_resampled[mask_hyp]

    xx_wolter1_resampled = np.arange(min(np.nanmin(xx_ell_h_resampled), np.nanmin(xx_hyp_h_resampled)), max(np.nanmax(xx_ell_h_resampled), np.nanmax(xx_hyp_h_resampled)), datapitch)
    yy_wolter1_resampled = np.full_like(xx_wolter1_resampled, np.nan)
    for i, xx in enumerate(xx_wolter1_resampled):
        index_ell = (np.abs(xx_ell_h_resampled - xx)).argmin()
        index_hyp = (np.abs(xx_hyp_h_resampled - xx)).argmin()
        if yy_ell_h_resampled[index_ell] > yy_hyp_h_resampled[index_hyp]:
            yy_wolter1_resampled[i] = yy_ell_h_resampled[index_ell]
        else:
            yy_wolter1_resampled[i] = yy_hyp_h_resampled[index_hyp]

    xx_wolter1_extrapolated_u = np.arange(np.min(xx_wolter1_resampled)-margin_u_extra, np.min(xx_wolter1_resampled), datapitch)
    dydx1 = (yy_wolter1_resampled[1]-yy_wolter1_resampled[0])/(xx_wolter1_resampled[1]-xx_wolter1_resampled[0])
    yy_wolter1_extrapolated_u_parabola = extrapolate_parabola(xx_wolter1_resampled[0]-margin_u_extra, yy_wolter1_resampled[0]+height_margin, xx_wolter1_resampled[0], yy_wolter1_resampled[0], dydx1, xx_wolter1_extrapolated_u)

    xx_wolter1_extrapolated_l = np.arange(np.max(xx_wolter1_resampled), np.max(xx_wolter1_resampled)+margin_l_extra, datapitch)
    dydx1 = (yy_wolter1_resampled[-1]-yy_wolter1_resampled[-2])/(xx_wolter1_resampled[-1]-xx_wolter1_resampled[-2])
    yy_wolter1_extrapolated_l_parabola = extrapolate_parabola(xx_wolter1_extrapolated_l[-1] + margin_l_extra, yy_wolter1_resampled[-1]+height_margin, xx_wolter1_resampled[-1], yy_wolter1_resampled[-1], dydx1, xx_wolter1_extrapolated_l)

    print("hyp_v margin_upper", np.nanmin(xx_hyp_v_roted)-np.nanmin(xx_hyp_v_resampled))
    print("hyp_v margin_lower", np.nanmax(xx_hyp_v_resampled)-np.nanmax(xx_hyp_v_roted))
    print("ell_v margin_upper", np.nanmin(xx_ell_v_roted)-np.nanmin(xx_ell_v_resampled))
    print("ell_v margin_lower", np.nanmax(xx_ell_v_resampled)-np.nanmax(xx_ell_v_roted))
    print("wolter1 margin_upper", np.nanmin(xx_ell_h_roted)-np.nanmin(xx_wolter1_resampled))
    print("wolter1 margin_lower", np.nanmax(xx_wolter1_resampled)-np.nanmax(xx_hyp_h_roted))
    print("wolter1 gap", np.nanmin(xx_hyp_h_roted)-np.nanmax(xx_ell_h_roted))
    plt.figure(figsize=(10,5))
    plt.plot(xx_ell_h_roted*1e3, yy_ell_h_roted*1e3)
    plt.plot(xx_hyp_h_roted*1e3, yy_hyp_h_roted*1e3)
    # plt.plot(xx_ell_h_resampled*1e3, yy_ell_h_resampled*1e3, linestyle='dashed')
    # plt.plot(xx_hyp_h_resampled*1e3, yy_hyp_h_resampled*1e3, linestyle='dashed')
    plt.plot(xx_wolter1_resampled*1e3, yy_wolter1_resampled*1e3, linestyle='dashed')
    plt.plot(xx_wolter1_extrapolated_u*1e3, yy_wolter1_extrapolated_u_parabola*1e3, linestyle='dashed')
    plt.plot(xx_wolter1_extrapolated_l*1e3, yy_wolter1_extrapolated_l_parabola*1e3, linestyle='dashed')
    plt.title(f"wolter1 rot_wolter1={np.degrees(rot_wolter1):.2f} deg")
    plt.xlabel("Position (mm)")
    plt.ylabel("Shape (mm)")
    plt.savefig("wolter1.png")


    wolter1_x, wolter1_y = merge_array(xx_wolter1_resampled, yy_wolter1_resampled,
                                    xx_wolter1_extrapolated_u, yy_wolter1_extrapolated_u_parabola,
                                    xx_wolter1_extrapolated_l, yy_wolter1_extrapolated_l_parabola)


    np.savetxt("wolter1_rotated.csv", np.column_stack((wolter1_x*1e3, wolter1_y*1e3)), fmt="%.6f", header="x(mm),y(mm)",delimiter=',')

    ### wolter1_x*1e3, wolter1_y*1e3の datpitchを0.1mmに変更して
    datapitch_interp = 0.1  # 0.1mm
    wolter1_x_interp = np.arange(np.nanmin(wolter1_x*1e3), np.nanmax(wolter1_x*1e3), datapitch_interp)
    wolter1_y_interp = np.interp(wolter1_x_interp, wolter1_x*1e3, wolter1_y*1e3)
    print("wolter1_x_interp nan index", np.isnan(wolter1_x_interp).sum())
    print("wolter1_y_interp nan index", np.isnan(wolter1_y_interp).sum())
    np.savetxt("wolter1_rotated_0.1mmpitch.csv", np.column_stack((wolter1_x_interp, wolter1_y_interp)), fmt="%.6f", header="x(mm),y(mm)",delimiter=',')
    plt.figure(figsize=(10,5))
    plt.plot(wolter1_x*1e3, wolter1_y*1e3)
    plt.plot(wolter1_x_interp, wolter1_y_interp, linestyle='dashed')
    plt.title(f"wolter1 merged rot_wolter1={np.degrees(rot_wolter1):.2f} deg")
    plt.xlabel("Position (mm)")
    plt.ylabel("Shape (mm)")
    plt.show()