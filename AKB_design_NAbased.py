import numpy as np
from numpy import cos,sin,tan,arccos,arcsin,arctan,sqrt
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


        pass  # Placeholder for actual calculations
class Wolter3:
    # def __init__(self, l_1a, na_o, x_3, l_2a,theta_2a, na_i,estimate_theta_4a):
    #     self.l_1a = l_1a ### constant
    #     self.l_2a = l_2a ### variable
    #     self.theta_2a = theta_2a ### variable
    #     self.na_i = na_i ### variable
    #     self.hyp_design()

    #     # self.na_o = na_o ### constant

    #     self.x_3 = x_3 ### constant
    #     ### constant values based on hyp_design
    #     self.l_3a = (self.x_1 + self.x_2 + self.x_3 - self.f_hyp*2)

    #     ### variable values for estimation
    #     self.l_4a = 0.11196128532403424

    #     self.theta_4a = estimate_theta_4a

    #     self.theta_5a = self.theta_4a*2 - self.theta_3a
    #     self.ell_design()

    #     target_na_o = na_o ### target constant
    def __init__(self):
        pass

    def optimize(self, l_1a, x_3, tar_na_o, tar_l_4b, var_theta_2a, var_na_i, est_l_2a, est_theta_4a):
        init_params =[est_l_2a, est_theta_4a]
        self.target_params = [tar_na_o, tar_l_4b]
        self.l_1a = l_1a ### constant
        self.theta_2a = var_theta_2a ### variable
        self.na_i = var_na_i ### variable
        self.x_3 = x_3 ### constant

        # self.l_3a(constant) self.l_4a(constant) self.theta_4a(variable) ### variable
        # 最小化対象関数（目的関数）
        def objective(params):
            self.l_2a = init_params[0] ### estimate variable
            self.theta_4a = params[1]
            ### set hyp_design
            self.hyp_design()
            ### constant values based on hyp_design
            self.l_3a = (self.x_1 + self.x_2 + self.x_3 - self.f_hyp*2)/cos(self.theta_3a)
            # self.l_4a = params[1]
            self.theta_5a = self.theta_4a*2 - self.theta_3a
            self.ell_design()
            ### error calculation
            err_na_o = self.na_o - self.target_params[0]
            err_l_4b = self.l_4b - self.target_params[1]
            return np.sqrt((err_na_o)**2 + (err_l_4b)**2)

        # 初期値

        # 範囲制限
        bounds = [(1e-5, init_params[0]+1),(init_params[1]/20, 0.3)]
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
        self.l_2a, self.theta_4a = result.x
        print(f"\n✅ 最適解:")
        print(f"  l_2a        = {self.l_2a:.6f}")
        print(f"  theta_4a        = {self.theta_4a:.6f}")
        print(f"  最終誤差    = {result.fun:.6e}")

        self.hyp_design()
        ### constant values based on hyp_design
        self.l_3a = (self.x_1 + self.x_2 + self.x_3 - self.f_hyp*2)
        self.theta_5a = self.theta_4a*2 - self.theta_3a
        self.ell_design()
        print(f"　誤差の詳細")
        print(f"  na_o        = {self.na_o - self.target_params[0]:.6f}")
        print(f"  l_4b         = {self.l_4b - self.target_params[1]:.6f}")

    def hyp_design(self):
        self.a_hyp = (self.l_1a - self.l_2a)/2
        self.b_hyp = sqrt(self.l_1a*self.l_2a)*sin(self.theta_2a)
        self.f_hyp = sqrt(self.a_hyp**2 + self.b_hyp**2)

        self.theta_1a = arccos((self.l_1a - self.l_2a*cos(2*self.theta_2a))/(2*self.f_hyp))
        self.theta_1b = self.theta_1a + self.na_i
        self.theta_3a = self.theta_1a + self.theta_2a*2

        A = 1./self.a_hyp**2 * cos(self.theta_1b)**2 - 1./self.b_hyp**2 * sin(self.theta_1b)**2
        B = -2 * self.f_hyp/self.a_hyp**2 * cos(self.theta_1b)
        C = self.f_hyp**2/self.a_hyp**2 - 1
        self.l_1b = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
        self.l_2b = self.l_1b - self.a_hyp*2

        self.theta_2b = arcsin(self.b_hyp/sqrt(self.l_1b*self.l_2b))
        self.theta_3b = self.theta_1b + self.theta_2b*2
        self.na_int = self.theta_3a - self.theta_3b
        self.x_1 = self.l_1a*cos(self.theta_1a)
        self.x_2 = self.l_1b*cos(self.theta_1b) - self.x_1
        pass
    def ell_design(self):
        self.a_ell = (self.l_3a + self.l_4a)/2
        self.b_ell = sqrt(self.l_3a * self.l_4a) * np.sin(self.theta_4a)
        self.f_ell = np.sqrt(self.a_ell**2 - self.b_ell**2)
        self.l_4a =  self.a_ell*2 - self.l_3a


        A = 1./self.a_ell**2 * cos(self.theta_3b)**2 + 1./self.b_ell**2 * sin(self.theta_3b)**2
        B = -2 * self.f_ell/self.a_ell**2 * cos(self.theta_3b)
        C = self.f_ell**2/self.a_ell**2 - 1
        self.l_3b = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
        self.l_4b =  self.a_ell*2 - self.l_3b

        self.theta_4b = arcsin(self.b_ell/sqrt(self.l_3b * self.l_4b))
        self.theta_5b = self.theta_4b*2 - self.theta_3b
        self.na_o = self.theta_5b - self.theta_5a
        self.x_4 = self.l_3b*cos(self.theta_3b) - self.l_3a*cos(self.theta_3a)
        self.x_5 = self.l_4a*cos(self.theta_5a) - self.x_4
        # print('self.l_3b*cos(self.theta_3b)',self.l_3b*cos(self.theta_3b))
        # print('self.l_3a*cos(self.theta_3a)',self.l_3a*cos(self.theta_3a))
    def print(self):
        print('self.a_hyp',self.a_hyp)
        print('self.b_hyp',self.b_hyp)
        print('self.f_hyp',self.f_hyp)
        print('self.a_ell',self.a_ell)
        print('self.b_ell',self.b_ell)
        print('self.f_ell',self.f_ell)

        print('self.theta_1a',self.theta_1a)
        print('self.theta_1b',self.theta_1b)
        print('self.theta_2a',self.theta_2a)
        print('self.theta_2b',self.theta_2b)
        print('self.theta_3a',self.theta_3a)
        print('self.theta_3b',self.theta_3b)
        print('self.theta_4a',self.theta_4a)
        print('self.theta_4b',self.theta_4b)
        print('self.theta_5a',self.theta_5a)
        print('self.theta_5b',self.theta_5b)
        print('self.na_int',self.na_int)
        print('self.na_i',self.na_i)
        print('self.na_o',self.na_o)
        print('NA_o',sin(self.na_o)/2)

        print('self.l_1a',self.l_1a)
        print('self.l_1b',self.l_1b)
        print('self.l_2a',self.l_2a)
        print('self.l_2b',self.l_2b)
        print('self.l_3a',self.l_3a)
        print('self.l_3b',self.l_3b)
        print('self.l_4a',self.l_4a)
        print('self.l_4b',self.l_4b)

        print('self.x_1',self.x_1)
        print('self.x_2',self.x_2)
        print('self.x_3',self.x_3)
        print('self.x_4',self.x_4)
        print('self.x_5',self.x_5)

def ELL_V_design(l_i1, l_o1, theta_g1, na_o_sin):
    na_o = np.float64(np.arcsin(na_o_sin)*2)
    Ell1 = Ell(l_i1, l_o1, theta_g1, na_o)
    return Ell1
def ELL_H_design(Ell1, target_l_o2, target_gap, ast):
    target_x_1 = Ell1.edge + target_gap
    print('target_x_1',target_x_1)
    target_f = Ell1.f + ast
    l_i1 = Ell1.l_i2 + target_gap
    theta_g1 = Ell1.theta_g1/3
    print('l_i1',l_i1)
    # 最小化対象関数（目的関数）
    def objective(params):
        l_i1, l_o1, theta_g1 = params
        ell = Ell(l_i1, l_o1, theta_g1, Ell1.na_o,allcalc=False)
        err_l_o2 = ell.l_o2 - target_l_o2
        err_x_1 = ell.x_1 - target_x_1
        err_f = ell.f - target_f
        return np.sqrt((err_l_o2 / target_l_o2)**2 + (err_f / target_f)**2 + (err_x_1 / target_x_1)**2)

    # 初期値
    init_params = [l_i1, Ell1.l_o2, theta_g1]

    # 範囲制限
    bounds = [(l_i1-0.1, l_i1+0.1),(0.01,2), (1e-4, np.pi / 4)]
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

    Ell2 = Ell(l_i1, opt_l_o1, opt_theta_g1, Ell1.na_o)
    print(f"　誤差の詳細")
    print(f"  l_o2        = {Ell2.l_o2-target_l_o2:.6f}")
    print(f"  x_1         = {Ell2.x_1-target_x_1:.6f}")
    print(f"  f           = {Ell2.f-target_f:.6f}")
    return Ell1, Ell2
def KB_design(l_i1, l_o1, theta_g1, na_o_sin,target_l_o2, target_gap, ast):
    Ell1 = ELL_V_design(l_i1, l_o1, theta_g1, na_o_sin)
    print('Ell1 design')
    Ell1.print()
    Ell1, Ell2 = ELL_H_design(Ell1, target_l_o2, target_gap, ast)
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

    fig, ax = plt.subplots(1,2)
    ax[0].plot([0, Ell1.x_2],[np.rad2deg((Ell1.theta_i1+Ell1.theta_o1)/2), np.rad2deg((Ell1.theta_i2+Ell1.theta_o2)/2)], 'r--')
    ax[1].plot([0, Ell2.x_2],[np.rad2deg((Ell2.theta_i1+Ell2.theta_o1)/2), np.rad2deg((Ell2.theta_i2+Ell2.theta_o2)/2)], 'r--')
    ax[0].set_ylabel('incident angle (deg)')
    plt.show()
    return
if __name__ == '__main__':
    # 初期値
    # l_i1 = np.float64(145.7500024376426)
    # l_o1 = np.float64(1.0499975623574187)
    # theta_g1 = np.float64(0.211)
    # na_o_sin = np.float64(0.082)
    # target_l_o2 = np.float64(0.04) ### WD
    # target_gap = np.float64(0.02)
    # ast = np.float64(0.)
    # # Ell1, Ell2 = KB_design(l_i1, l_o1, theta_g1, na_o_sin,target_l_o2, target_gap, ast)
    # Ell1 = ELL_V_design(l_i1, l_o1, theta_g1, na_o_sin)
    # Ell1, Ell2 = ELL_H_design(Ell1, target_l_o2, target_gap, ast)
    # plot_ellipses(Ell1, Ell2)
    l_1a = np.float64(145.98274882106028)
    tar_na_o = np.float64(arcsin(0.0820569070569736)*2)
    x_3 = np.float64(0.01)
    est_l_2a = np.float64(0.01274882106028354)
    tar_l_4b = np.float64(0.09)
    var_theta_2a = np.float64(0.18470626604945578)
    var_na_i = np.float64(2.993372063373417e-05)
    est_theta_4a = 0.2732948272725705
    # Wolter3_V = Wolter3(l_1a, tar_na_o, x_3, est_l_2a, var_theta_2a, var_na_i,est_theta_4a)
    Wolter3_V = Wolter3()
    Wolter3_V.optimize(l_1a, x_3, tar_na_o, tar_l_4b, var_theta_2a, var_na_i, est_l_2a, est_theta_4a)
    Wolter3_V.print()
