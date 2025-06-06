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
        print('Apperture',self.mirr_length*self.theta_centre)
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
def ELL_H_design(Ell1, target_l_o2, target_gap, ast,na_o_sin_h):
    
    target_x_1 = Ell1.edge + target_gap
    print('target_x_1',target_x_1)
    target_f = Ell1.f + ast
    l_i1 = Ell1.l_i2 + target_gap
    theta_g1 = Ell1.theta_g1/3
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
    bounds = [(l_i1-0.1, l_i1+0.1),(0.01,2), (1e-9, np.pi / 4)]
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=1000,
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
    print(f"\n✅ 最適解:")
    print(f"  l_i1        = {opt_i1:.6f}")
    print(f"  l_o1        = {opt_l_o1:.6f}")
    print(f"  theta_g1    = {np.rad2deg(opt_theta_g1):.4f} deg")
    print(f"  l_i1        = {opt_i1:.6f}")
    print(f"  最終誤差    = {result.fun:.6e}")

    Ell2 = Ell(l_i1, opt_l_o1, opt_theta_g1, na_o)
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
    plt.plot([Ell2.x_1, Ell2.x_1+Ell2.x_2], [Ell2.y_1, Ell2.y_2], 'r--')
    plt.plot(2*Ell2.f, 0, 'ro')

    # fig, ax = plt.subplots(1,2)
    # ax[0].plot([0, Ell1.x_2],[np.rad2deg((Ell1.theta_i1+Ell1.theta_o1)/2), np.rad2deg((Ell1.theta_i2+Ell1.theta_o2)/2)], 'r--')
    # ax[1].plot([0, Ell2.x_2],[np.rad2deg((Ell2.theta_i1+Ell2.theta_o1)/2), np.rad2deg((Ell2.theta_i2+Ell2.theta_o2)/2)], 'r--')
    # ax[0].set_ylabel('incident angle (deg)')
    # # plt.show()
    fig1, ax1 = plt.subplots(1,2)
    ax1[0].plot([0, Ell1.x_2],[((Ell1.theta_i1+Ell1.theta_o1)/2), ((Ell1.theta_i2+Ell1.theta_o2)/2)], 'r--')
    ax1[1].plot([0, Ell2.x_2],[((Ell2.theta_i1+Ell2.theta_o1)/2), ((Ell2.theta_i2+Ell2.theta_o2)/2)], 'r--')
    ax1[0].set_ylabel('incident angle (rad)')
    plt.show()
    return
def plot_mirrors(Ell1, Ell2):
    
    return
if __name__ == '__main__':
    # # 初期値
    # l_i1 = np.float64(145.7500024376426)
    # l_o1 = np.float64(1.0499975623574187)
    # theta_g1 = np.float64(0.211)
    # na_o_sin = np.float64(0.082)
    # target_l_o2 = np.float64(0.04) ### WD
    # target_gap = np.float64(0.02)

    # 初期値
    l_i1 = np.float64(110)
    l_o1 = np.float64(1.55)
    theta_g1 = np.float64(3.95e-3)
    na_o_sin_v = np.float64(0.00085)*0.95
    na_o_sin_h = np.float64(0.0008)*1.75
    target_l_o2 = np.float64(0.51) ### WD
    target_gap = np.float64(0.03)

    ast = np.float64(0.)
    # Ell1, Ell2 = KB_design(l_i1, l_o1, theta_g1, na_o_sin,target_l_o2, target_gap, ast)
    Ell1 = ELL_V_design(l_i1, l_o1, theta_g1, na_o_sin_v)
    Ell1, Ell2 = ELL_H_design(Ell1, target_l_o2, target_gap, ast,na_o_sin_h)
    plot_ellipses(Ell1, Ell2)

    Ell1.raytrace(2e-3)
    Ell2.raytrace(2e-3)

    Ell1.calc_mirror()
    Ell2.calc_mirror()
