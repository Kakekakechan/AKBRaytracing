import numpy as np
from numpy import cos,sin,tan,arccos,arctan,sqrt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
class Ell:
    def __init__(self, l_i1, l_o1, theta_g1, na_o, allcalc=True):
        self.allcalc = allcalc
        self.na_o = na_o  # NA of the output beam
        self.l_i1 = l_i1
        self.l_o1 = l_o1
        self.theta_g1 = theta_g1
        self.ell_design()
    def print(self):
        print('self.a',self.a)
        print('self.b',self.b)
        print('self.f',self.f)
        print('self.theta_i1',self.theta_i1)
        print('self.theta_i2',self.theta_i2)
        print('self.theta_o1',self.theta_o1)
        print('self.theta_o2',self.theta_o2)
        print('self.l_i1',self.l_i1)
        print('self.l_o1',self.l_o1)
        print('self.l_i2',self.l_i2)
        print('self.l_o2',self.l_o2)
        print('self.x_1',self.x_1)
        print('self.x_2',self.x_2)
        print('self.x_3',self.x_3)
        print('self.edge',self.edge)
        print('self.y_1',self.y_1)
        print('self.y_2',self.y_2)
        print('self.m1',self.m1)
        print('self.m2',self.m2)
        print('self.theta_i_cnt_angle',self.theta_i_cnt_angle)
        print('self.theta_i_cnt_m_wid',self.theta_i_cnt_m_wid)
        print("self.theta_i_cnt_o_angle",self.theta_i_cnt_o_angle)
        print('self.x_cnt_m',self.x_cnt_m_wid)
        print('self.y_cnt_m',self.y_cnt_m_wid)
        print('self.omega_default',self.omega_default)
        print('self.omega_cnt_m_wid',self.omega_cnt_m_wid)
        print('self.omega_cnt_o_angle',self.omega_cnt_o_angle)
        print('self.x_cent_o_angle',self.x_cent_o_angle)
        print('self.y_cent_o_angle',self.y_cent_o_angle)
        print('=====mirror parameter=====')
        print('self.p_centre',self.p_centre)
        print('self.q_centre',self.q_centre)
        print('self.theta_centre',self.theta_centre)
        print('self.mirr_length',self.mirr_length)
        print('Aperture',self.mirr_length*self.theta_centre)
        print('self.na_o',self.na_o)
        print('==========================')
        
    def calc_mirror(self):
        self.x_datas = np.linspace(self.x_1, self.x_1+self.x_2, 100)
        self.y_datas = self.b * np.sqrt(1 - ((self.x_datas-self.f)/self.a)**2)
        plt.plot(self.x_datas, self.y_datas, 'b--')
        plt.show()
        # 2次関数フィッティング
        coeffs = np.polyfit(self.x_datas, self.y_datas, deg=2)  # [a, b, c]
        a, b, c = coeffs
        print(f"y = {a:.3f} x^2 + {b:.3f} x + {c:.3f}")
        print(f"RoC:{1/(2*a):.3f} m")
        return
    def raytrace(self, size, n_points=100):
        source = np.hstack((np.linspace(-size/2, size/2, n_points//2),-np.linspace(-size/2, size/2, n_points//2)))
        sin_here = np.sin((self.theta_i1+self.theta_i2)/2)
        cos_here = np.cos((self.theta_i1+self.theta_i2)/2)
        self.x_datas = np.linspace(self.x_1, self.x_1+self.x_2, n_points)
        self.y_datas = self.b * np.sqrt(1 - ((self.x_datas-self.f)/self.a)**2)

        # 2次関数フィッティング
        coeffs = np.polyfit(self.x_datas, self.y_datas, deg=2)  # [a, b, c]
        a, b, c = coeffs
        print(f"y = {a:.3f} x^2 + {b:.3f} x + {c:.3f}")
        print(f"RoC:{1/(2*a):.3f} m")

        self.dydx_datas = -self.b * ((self.x_datas-self.f)/(self.a**2 * np.sqrt(1 - ((self.x_datas-self.f)/self.a)**2)))
        self.nvector_datas = np.vstack((np.ones(n_points), -1/self.dydx_datas))
        self.nvector_datas = self.nvector_datas / np.linalg.norm(self.nvector_datas, axis=0)

        self.ivector_datas = np.vstack((self.x_datas-source*sin_here, self.y_datas-source*cos_here))

        dot_in = np.sum(self.ivector_datas * self.nvector_datas, axis=0)  # スカラー積（i・n）
        self.rvector_datas = self.ivector_datas - 2 * self.nvector_datas * dot_in
        self.rvector_datas = self.rvector_datas / np.linalg.norm(self.rvector_datas, axis=0)

        plt.plot(self.x_datas, self.y_datas, 'b')

        for i in range(n_points):
            plt.plot([self.x_datas[i], self.x_datas[i] - self.ivector_datas[0, i]], [self.y_datas[i], self.y_datas[i] - self.ivector_datas[1, i]], 'k',linewidth=0.1)
            plt.plot([self.x_datas[i], self.x_datas[i] + self.rvector_datas[0, i]], [self.y_datas[i], self.y_datas[i] + self.rvector_datas[1, i]], 'k',linewidth=0.1)
            # plt.plot([self.x_datas[i], self.x_datas[i] + self.nvector_datas[0, i]], [self.y_datas[i], self.y_datas[i] + self.nvector_datas[1, i]], 'k--')      
        # plt.show()

        def spot(x, y, r_x, r_y, f):
            y_dash = (f-x)/r_x * r_y + y
            return y_dash
        # 収束点
        fig, ax = plt.subplots(1, 5,sharey=True)
        for i in range(5):
            Focus = 2*self.f + (i-2) * 1e-5/self.na_o
            y_dash = spot(self.x_datas, self.y_datas, self.rvector_datas[0], self.rvector_datas[1], Focus)
            ax[i].scatter((Focus-2*self.f)*np.ones(n_points), y_dash-(np.min(y_dash)+np.max(y_dash))/2, c='r', s=1)

        plt.show()
        



    def ell_design(self):
        self.a = (self.l_i1 + self.l_o1)/2
        self.b2 = self.l_i1 * self.l_o1 * np.sin(self.theta_g1)**2
        self.f = np.sqrt(self.a**2 - self.b2)
        A = self.l_i1**2 * (1/self.a**2 - 1/self.b2)
        B = -2 * self.l_i1 * self.f / self.a**2
        C = self.f**2 / self.a**2 + self.l_i1**2/self.b2 - 1
        t = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
        self.theta_i1 = np.arccos(t)
        self.x_1 = self.l_i1 * cos(self.theta_i1)
        self.theta_o1 = 2*self.theta_g1 - self.theta_i1
        self.theta_o2 = self.theta_o1 + self.na_o
        A2 = cos(self.theta_o2)**2/self.a**2 + sin(self.theta_o2)**2/self.b2
        B2 = -2*self.f*cos(self.theta_o2)/self.a**2
        C2 = (self.f**2)/self.a**2 - 1
        self.l_o2 = (-B2 + np.sqrt(B2**2 - 4*A2*C2))/(2*A2)
        if self.allcalc:
            self.b = np.sqrt(self.b2)
            self.l_i2 = 2 * self.a - self.l_o2
            self.x_3 = self.l_o2 * np.cos(self.theta_o2)
            self.x_2 = self.l_o1 * np.cos(self.theta_o1) - self.x_3
            self.theta_i2 = arccos((self.x_1 + self.x_2)/self.l_i2)
            self.m1 = tan(self.theta_o1) / tan(self.theta_i1)
            self.m2 = tan(self.theta_o2) / tan(self.theta_i2)
            self.y_1 = self.l_i1 * sin(self.theta_i1)
            self.y_2 = self.l_o2 * sin(self.theta_o2)
            self.edge = self.x_1 + self.x_2
            self.omega_default = (self.theta_i1 + self.theta_o1 + self.theta_i2 + self.theta_o2)/2
            ### mirror center for conventional code
            self.x_cnt_m_wid = self.x_1 + self.x_2/2
            self.y_cnt_m_wid = np.sqrt(self.b2 * (1 - ((self.x_cnt_m_wid-self.f)/self.a)**2))
            self.theta_i_cnt_m_wid = arctan(self.y_cnt_m_wid / self.x_cnt_m_wid )
            self.theta_o_cnt_m_wid = arctan(self.y_cnt_m_wid / (2*self.f - self.x_cnt_m_wid))
            self.omega_cnt_m_wid = self.theta_i_cnt_m_wid + self.theta_o_cnt_m_wid
            ### output center
            self.theta_o_cnt_o_angle = (self.theta_o1 + self.theta_o2)/2
            A3 = cos(self.theta_o_cnt_o_angle)**2/self.a**2 + sin(self.theta_o_cnt_o_angle)**2/self.b2
            B3 = -2*self.f*cos(self.theta_o_cnt_o_angle)/self.a**2
            C3 = (self.f**2)/self.a**2 - 1
            self.l_o_cnt_o_angle = (-B3 + np.sqrt(B3**2 - 4*A3*C3))/(2*A3)
            self.x_cent_o_angle = 2*self.f - self.l_o_cnt_o_angle * cos(self.theta_o_cnt_o_angle)
            self.y_cent_o_angle = self.l_o_cnt_o_angle * sin(self.theta_o_cnt_o_angle)
            self.l_i_cnt_o_angle = 2 * self.a - self.l_o_cnt_o_angle
            self.theta_i_cnt_o_angle = arccos(self.x_cent_o_angle /self.l_i_cnt_o_angle)
            self.omega_cnt_o_angle = self.theta_i_cnt_o_angle + self.theta_o_cnt_o_angle
            ### input center
            self.theta_i_cnt_angle = (self.theta_i1+self.theta_i2)/2

            self.x_centre = self.x_1 + self.x_2/2
            self.y_centre = np.sqrt(self.b2 * (1 - ((self.x_centre-self.f)/self.a)**2))
            self.p_centre = np.sqrt(self.x_centre**2 + self.y_centre**2)
            self.q_centre = 2 * self.a -self.p_centre
            self.theta_centre = np.arcsin(np.sqrt(self.b2/(self.p_centre*self.q_centre)))
            self.mirr_length = self.x_2


        pass  # Placeholder for actual calculations
def ELL_V_design(l_i1, l_o1, theta_g1, na_o_sin):
    na_o = np.float64(np.arcsin(na_o_sin)*2)
    Ell1 = Ell(l_i1, l_o1, theta_g1, na_o)
    return Ell1
def ELL_H_design(Ell1, target_l_o2, target_gap, ast,na_o_sin_h,display=True):
    
    target_x_1 = Ell1.edge + target_gap
    target_f = Ell1.f + ast
    l_i1 = Ell1.l_i2 + target_gap
    theta_g1 = Ell1.theta_g1/3
    if display:
        print('target_x_1',target_x_1)
        print('l_i1',l_i1)
    # 最小化対象関数（目的関数）
    na_o = np.float64(np.arcsin(na_o_sin_h)*2)
    # na_o = Ell1.na_o
    
    def objective(params):
        l_i1, l_o1, theta_g1 = params
        ell = Ell(l_i1, l_o1, theta_g1, na_o,allcalc=False)
        err_l_o2 = ell.l_o2 - target_l_o2
        err_x_1 = ell.x_1 - target_x_1
        err_f = ell.f - target_f
        return np.sqrt((err_l_o2 / target_l_o2)**2 + (err_f / target_f)**2 + (err_x_1 / target_x_1)**2)

    # 初期値
    init_params = [l_i1, Ell1.l_o2, theta_g1]

    # 範囲制限
    bounds = [(l_i1-1, l_i1+1),(0.001,2), (1e-9, np.pi / 4)]
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=10000,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        polish=True,
        disp=False
    )
    # 結果出力
    opt_i1, opt_l_o1, opt_theta_g1 = result.x
    Ell2 = Ell(l_i1, opt_l_o1, opt_theta_g1, na_o)
    if display:
        print("反復回数:", result.nit)
        print(f"\n✅ 最適解:")
        print(f"  l_i1        = {opt_i1:.6f}")
        print(f"  l_o1        = {opt_l_o1:.6f}")
        print(f"  theta_g1    = {np.rad2deg(opt_theta_g1):.4f} deg")
        print(f"  l_i1        = {opt_i1:.6f}")
        print(f"  最終誤差    = {result.fun:.6e}")
        print(f"　誤差の詳細")
        print(f"  l_o2        = {Ell2.l_o2-target_l_o2:.6f}")
        print(f"  x_1         = {Ell2.x_1-target_x_1:.6f}")
        print(f"  f           = {Ell2.f-target_f:.6f}")
    return Ell1, Ell2
def KB_design(l_i1, l_o1, theta_g1, na_o_sin_v, na_o_sin_h,target_l_o2, target_gap, ast):
    Ell1 = ELL_V_design(l_i1, l_o1, theta_g1, na_o_sin_v)
    print('Ell1 design')
    Ell1.print()
    Ell1, Ell2 = ELL_H_design(Ell1, target_l_o2, target_gap, ast,na_o_sin_h)
    print('Ell2 design')
    Ell2.print()
    return Ell1, Ell2
def plot_ellipses(Ell1, Ell2):
    print('Ell1 design')
    Ell1.print()
    print('Ell2 design')
    Ell2.print()
    plt.figure()
    # plt.plot([0, Ell1.x_1], [0, Ell1.y_1], 'r', label='input beam')
    plt.plot([Ell1.x_1, Ell1.x_1+Ell1.x_2], [Ell1.y_1, Ell1.y_2], 'r--')
    plt.plot(2*Ell1.f, 0, 'ro')
    # plt.plot([0, Ell2.x_1], [0, Ell2.y_1], 'r', label='input beam')
    plt.plot([Ell2.x_1, Ell2.x_1+Ell2.x_2], [Ell2.y_1, Ell2.y_2], 'b--')
    plt.plot(2*Ell2.f, 0, 'bo')

    # fig, ax = plt.subplots(1,2)
    # ax[0].plot([0, Ell1.x_2],[np.rad2deg((Ell1.theta_i1+Ell1.theta_o1)/2), np.rad2deg((Ell1.theta_i2+Ell1.theta_o2)/2)], 'r--')
    # ax[1].plot([0, Ell2.x_2],[np.rad2deg((Ell2.theta_i1+Ell2.theta_o1)/2), np.rad2deg((Ell2.theta_i2+Ell2.theta_o2)/2)], 'r--')
    # ax[0].set_ylabel('incident angle (deg)')
    # # plt.show()
    fig1, ax1 = plt.subplots(1,2)
    ax1[0].plot([0, Ell1.x_2],[((Ell1.theta_i1+Ell1.theta_o1)/2), ((Ell1.theta_i2+Ell1.theta_o2)/2)], 'r--')
    ax1[1].plot([0, Ell2.x_2],[((Ell2.theta_i1+Ell2.theta_o1)/2), ((Ell2.theta_i2+Ell2.theta_o2)/2)], 'b--')
    ax1[0].set_xlabel('distance (m)')
    ax1[1].set_xlabel('distance (m)')
    ax1[0].set_ylabel('incident angle (rad)')
    ax1[0].set_title('Ell1 incident angle')
    ax1[1].set_title('Ell2 incident angle')
    print('Ell1 diverge angle', Ell1.theta_i1-Ell1.theta_i2)
    print('Ell1 mirror length', Ell1.mirr_length)
    print('Ell1 mirror angle', [(Ell1.theta_i1+Ell1.theta_o1)/2, (Ell1.theta_i2+Ell1.theta_o2)/2])
    print('Ell1 demagnification', [Ell1.m1, Ell1.m2, np.mean([Ell1.m1, Ell1.m2])])
    print('Ell2 diverge angle', Ell2.theta_i1-Ell2.theta_i2)
    print('Ell2 mirror length', Ell2.mirr_length)
    print('Ell2 mirror angle', [(Ell2.theta_i1+Ell2.theta_o1)/2, (Ell2.theta_i2+Ell2.theta_o2)/2])
    print('Ell2 demagnification', [Ell2.m1, Ell2.m2, np.mean([Ell2.m1, Ell2.m2])])
    print('===========================')
    print('Ell1 aperture',Ell1.mirr_length*Ell1.theta_centre)
    print('Ell2 aperture',Ell2.mirr_length*Ell2.theta_centre)
    print('Area aperture',Ell1.mirr_length*Ell1.theta_centre*Ell2.mirr_length*Ell2.theta_centre)
    print('Focus distance',Ell1.f-Ell2.f)
    plt.show()
    return
def plot_mirrors(Ell1, Ell2):
    
    return



if __name__ == '__main__':
    # 初期値
    

    # # ### 初期値
    # # # Ell1 aperture 0.01090493366342567
    # # # Ell2 aperture 0.002279544746263799
    # # # Area aperture 2.4858284240817232e-05
    # # l_o1 = np.float64(0.17)
    # # theta_g1 = np.float64(0.06)

    # l_i1 = np.float64(145.7500024376426)
    # ### 初期値
    # # Ell1 aperture 0.009622771816525613
    # # Ell2 aperture 0.002612969252157888
    # # Area aperture 2.5144006877112932e-05
    # l_o1 = np.float64(0.085)
    # theta_g1 = np.float64(0.2)

    # # ### 初期値
    # # Ell1 aperture 0.009740844713758013
    # # Ell2 aperture 0.002571216723527687
    # # Area aperture 2.5045822829300865e-05
    # # l_o1 = np.float64(0.0785)
    # # theta_g1 = np.float64(0.3)

    l_i1 = np.float64(146.)
    # ### 初期値
    # l_o1 = np.float64(0.3)
    # theta_g1 = np.float64(0.225)
    ### 初期値
    # l_o1 = np.float64(0.12)
    # theta_g1 = np.float64(0.20)
    # target_l_o2 = np.float64(0.024) ### WD

    na_o_sin_v = np.float64(0.082)
    na_o_sin_h = np.float64(0.082)
    
    target_gap = np.float64(0.030)
    ast = np.float64(0.)


    ### apertureが5になるように調整
    var_l_o1 = np.float64(0.3)
    var_theta_g1 = np.float64(0.14)
    var_target_l_o2 = np.float64(0.03) ### WD

    # ### apertureが5になるように調整
    # var_l_o1 = np.float64(0.15)
    # theta_g1 = np.float64(0.16)
    # target_l_o2 = np.float64(0.02125) ### WD


    target_aperture1 = np.float64(0.010)
    target_aperture2 = np.float64(0.010)
    l_o1 = var_l_o1.copy()  # 初期値の設定
    theta_g1 = var_theta_g1.copy()  # 初期値の設定
    target_l_o2 = var_target_l_o2.copy()  # 初期値の設定

    Ell1 = ELL_V_design(l_i1, l_o1, theta_g1, na_o_sin_v)
    Ell1, Ell2 = ELL_H_design(Ell1, target_l_o2, target_gap, ast, na_o_sin_h)  
    plot_ellipses(Ell1, Ell2)

    def objective(params):
        # l_o1 = params[0]
        l_o1, theta_g1, target_l_o2 = params
        Ell1 = ELL_V_design(l_i1, l_o1, theta_g1, na_o_sin_v)
        Ell1, Ell2 = ELL_H_design(Ell1, target_l_o2, target_gap, ast, na_o_sin_h,display=False)
        Focus_gap = Ell1.f - Ell2.f
        if Focus_gap > 1e-3:
            return np.inf  # 適切な焦点距離でない場合は無限大を返す
        if Ell1.m1 < 1000:
            return np.inf
        theta_g1_max = np.max([ (Ell1.theta_i1+ Ell1.theta_o1)/2, (Ell1.theta_i2 + Ell1.theta_o2)/2 ])
        theta_g2_max = np.max([ (Ell2.theta_i1+ Ell2.theta_o1)/2, (Ell2.theta_i2 + Ell2.theta_o2)/2 ])
        if theta_g1_max > 0.28 or theta_g2_max > 0.28:
            # print(f"theta_g1_max: {theta_g1_max:.6f}, theta_g2_max: {theta_g2_max:.6f}")
            return np.inf
        aperture_Ell1 = Ell1.mirr_length * Ell1.theta_centre
        aperture_Ell2 = Ell2.mirr_length * Ell2.theta_centre
        err_aperture1 = aperture_Ell1 - target_aperture1
        err_aperture2 = aperture_Ell2 - target_aperture2
        if np.abs(err_aperture2) < 0.001:
            print(f"l_o1: {l_o1:.6f}, theta_g1: {theta_g1:.6f}, target_l_o2: {target_l_o2:.6f}, aperture_Ell1: {aperture_Ell1:.6f}, aperture_Ell2: {aperture_Ell2:.6f}")
        eval_err = np.sqrt(err_aperture2**2)
        return eval_err

    # 探索範囲を定義（各パラメータに対して）
    bounds = [
        (0.2, 0.4),           # var_l_o1 の範囲
        (0.1, 0.26),           # var_theta_g1 の範囲
        (0.03, 0.05)           # var_target_l_o2 の範囲
    ]

    # differential evolution で最適化
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=1e-4,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        polish=True,
        disp=False
    )
    
    optimized_params = result.x

    Ell1 = ELL_V_design(l_i1, optimized_params[0], optimized_params[1], na_o_sin_v)
    Ell1, Ell2 = ELL_H_design(Ell1, optimized_params[2], target_gap, ast, na_o_sin_h)
    print("最適化結果:")
    print(f"l_o1: {optimized_params[0]:.6f}, theta_g1: {optimized_params[1]:.6f}, target_l_o2: {optimized_params[2]:.6f}")
    print(f"最終誤差: {result.fun:.6e}")

    # Ell1 = ELL_V_design(l_i1, optimized_params[0], theta_g1, na_o_sin_v)
    # Ell1, Ell2 = ELL_H_design(Ell1, target_l_o2, target_gap, ast, na_o_sin_h)
    # print("最適化結果:")
    # print(f"l_o1: {optimized_params[0]:.6f}, theta_g1: {theta_g1:.6f}")
    # print(f"最終誤差: {result.fun:.6e}")

    # Ell1 = ELL_V_design(l_i1, l_o1, theta_g1, na_o_sin_v)
    # Ell1, Ell2 = ELL_H_design(Ell1, target_l_o2, target_gap, ast, na_o_sin_h)    
    
    plot_ellipses(Ell1, Ell2)

    Ell1.raytrace(2e-3)
    Ell2.raytrace(2e-3)

    Ell1.calc_mirror()
    Ell2.calc_mirror()
