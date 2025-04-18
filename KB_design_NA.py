import numpy as np
import matplotlib.pyplot as plt

class Ell_V:
    def __init__(self, na_o, na_i, x1, wd):
        self.na_o = na_o  # NA of the output beam
        self.na_i = na_i  # NA of the input beam
        self.x1 = x1  # source to mirror edge distance
        self.wd = wd  # Working distance
    def ell_design(self):
        self.theta_i1
        pass  # Placeholder for actual calculations

    def ell_estimate(self):
        
        pass  # Placeholder for actual plotting logic


from scipy.optimize import fsolve
from scipy.optimize import least_squares
# --- 入力パラメータ --- 
x_1 = np.float64(146.0)  # 任意の長さ
x_3 = np.float64(0.55)   # 任意の長さ
na_i = np.float64(1e-4)
na_o = np.float64(0.02)

# --- 制約式の定義 ---
def equations(p):
    theta_i1, theta_o2 = p

    # 角度が負の値にならないように制約を加える
    if theta_i1 <= 0 or theta_o2 <= 0:
        return [1e10, 1e10]  # 不正な解には大きな値を返して無効にする

    theta_i2 = theta_i1 - na_i
    theta_o1 = theta_o2 - na_o

    # 各長さの定義
    l_i1 = x_1 / np.cos(theta_i1)
    l_o2 = x_3 / np.cos(theta_o2)

    # x_2 を介して l_i2, l_o1 を定義
    alpha = (l_o2 * (np.sin((theta_i2 + theta_o2)/2))**2) / (l_i1 * (np.sin((theta_i1 + theta_o1)/2))**2)
    l_i2 = (x_1 -x_3) / (np.cos(theta_i2) - alpha * np.cos(theta_o1))
    x_2 = l_i2 * np.cos(theta_i2) - x_1
    l_o1 = (x_3 + x_2) / np.cos(theta_o1)

    # l_i2 = (x_1 + x_2_i) / np.cos(theta_i2)
    # x_2 = l_i2 * np.cos(theta_i2) - x_1
    # l_o1 = (x_3 + x_2) / np.cos(theta_o1)

    # 半長軸 a, 半短軸 b
    a = (l_i1 + l_o1) / 2
    b2 = l_i1 * l_o1 * np.sin((theta_i1 + theta_o1) / 2) ** 2

    # 式1: 長さの合計一致
    eq1 = (l_i1 + l_o1) - (l_i2 + l_o2)

    # 式2: 楕円の焦点条件
    eq2 = (x_1 + x_2 + x_3) - 2 * np.sqrt(a**2 - b2)

    print("eq1:", eq1)
    print("eq2:", eq2)

    return [eq1, eq2]

# --- fsolveで非線形連立方程式を解く ---
initial_guess = [na_i*4, na_o*4]  # 初期推定（ラジアン）
# initial_guess = [4e-4, 0.072]  # 初期推定（ラジアン）
print("初期推定値:", initial_guess)
eq1, eq2 = equations(initial_guess)
print("初期推定値の方程式の残差:", eq1, eq2)
# theta_i1_sol, theta_o2_sol = fsolve(equations, initial_guess)
# print("解:", theta_i1_sol, theta_o2_sol)

# --- 初期推定と制約 ---
bounds = ([1e-6, 1e-6], [np.pi/2 - 1e-6, np.pi/2 - 1e-6])  # 正の範囲
# --- 最適化実行 ---
result = least_squares(equations, initial_guess, bounds=bounds, xtol=1e-15, ftol=1e-15, gtol=1e-15, verbose=2)
theta_i1_sol, theta_o2_sol = result.x
eq1, eq2 = equations([theta_i1_sol, theta_o2_sol])
print("結果の方程式の残差:", eq1, eq2)


# --- 解を使って他の変数を計算 --- na_o, x1, theta_i1_sol, theta_o2_sol 固定
theta_o1 = theta_o2_sol - na_o
print("theta_o1:", theta_o1)
print("theta_o2_sol:", theta_o2_sol)
print("theta_i1_sol:", theta_i1_sol)
theta_i1_o1 = theta_i1_sol + theta_o1
const1 = x_1 * (np.sin(theta_i1_o1/2))**2 / (np.cos(theta_i1_sol) * np.cos(theta_o1))

# print("const1:", const1)
# print("theta_i1_sol + theta_o1:", theta_i1_o1)
# print("cos_i1_o1:", cos_i1_o1)
# print("cos_i1_o1 error:", cos_i1_o1 - np.cos(theta_i1_o1))
print("x_1:", x_1)
A = 1/(np.cos(theta_o1)**2) - 1
B = 4 * (x_1 / (2 *np.cos(theta_o1) * np.cos(theta_i1_sol)) - x_1 / 2 - const1)
C = x_1**2 * (1 / (np.cos(theta_i1_sol)**2) - 1)
X = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A) ### X = x_2 + x_3
print("X:", X)
b2 = X * const1
f = (x_1 + X) / 2
a = np.sqrt(f**2 + b2)

l_i1 = x_1 / np.cos(theta_i1_sol)
l_o1 = 2 * a - l_i1

A2 = 1 / a**2 + np.tan(theta_o2_sol)**2 / b2
B2 = -2 * f / a**2
C2 = f**2 / a**2 - 1
x_3_result = (-B2 + np.sqrt(B2**2 - 4 * A2 * C2)) / (2 * A2)  # x_3の計算
x_2 = X - x_3_result  # x_2の計算
theta_i2 = np.arctan(x_3_result * np.tan(theta_o2_sol) / (x_1 + x_2))
l_i2 = (x_1 + x_2) / np.cos(theta_i2)
l_o2 = x_3_result / np.cos(theta_o2_sol)

na_i_result = theta_i1_sol - theta_i2

### check
print("check a error", a - (l_i2 + l_o2) / 2)
print("check na_i error", na_i_result - na_i)
print("check x_3 error", x_3_result - x_3)

# --- 結果表示 ---
print("theta_i1 :", theta_i1_sol)
print("theta_i2 :", theta_i2)
print("theta_o1 :", theta_o1)
print("theta_o2 :", theta_o2_sol)
print("x_1:", x_1)
print("x_2:", x_2)
print("x_3:", x_3_result)
print("l_i1:", l_i1)
print("l_i2:", l_i2)
print("l_o1:", l_o1)
print("l_o2:", l_o2)
print("a:", a)
print("b^2:", b2)
print("f (焦点距離):", f)
print("x1 + x2 + x3:", x_1 + x_2 + x_3)
print("2f:", 2 * f)

print("na_i_result:", na_i_result)
print("na_o:", na_o)