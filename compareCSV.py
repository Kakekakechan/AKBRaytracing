import numpy as np
import matplotlib.pyplot as plt

# --- 設定 ---
file1 = 'hyp_v_rotated53.csv'   # CSV1 のパス ### raytrace
# file1 = 'side_wolter3_hyp_rotated.csv'   # CSV1 のパス
# file2 = 'hyp_v_rotated_0.1mmpitch.csv'   # CSV2 のパス ### process data
file2 = 'hyp_v_rotated_before_offset.csv'   # CSV2 のパス
dx = 0.1              # 補間後の x ピッチ

# --- CSV 読み込み (ヘッダ1行スキップ, 区切り文字はカンマ) ---
# データ形式は "x,y" と仮定
d1 = np.loadtxt(file1, delimiter=',', skiprows=1)
d2 = np.loadtxt(file2, delimiter=',', skiprows=1)
x1, y1 = d1[:,0], d1[:,1]  # mm に変換
x2, y2 = d2[:,0], d2[:,1]  # mm に変換
x1-= np.min(x1)
x2-= np.min(x2) # オフセット分
y1-= np.max(y1)
y2-= np.max(y2)

print('length xq1:', x1[-1]- x1[0])
print('length xq2:', x2[-1]- x2[0])

# print('minimum arg y1:', x1[np.argmin(y1)])
# print('minimum arg y2:', x2[np.argmin(y2)])
# x2-= x2[np.argmin(y2)]- x1[np.argmin(y1)] # オフセット分

print('maximum arg y1:', x1[np.argmax(y1)])
print('maximum arg y2:', x2[np.argmax(y2)])
x2-= x2[np.argmax(y2)]- x1[np.argmax(y1)] # オフセット分

plt.figure()
plt.plot(x1, y1, label='raytrace raw', alpha=0.5)
plt.plot(x2, y2, label='process data raw', alpha=0.5)
plt.legend()
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Raw Data Comparison')
plt.grid(True)
plt.show()
# --- 共通の x 範囲を決定 ---
# 重なりのある部分だけを対象にしたい場合は下記を使う
x_min = max(x1.min(), x2.min())
x_max = min(x1.max(), x2.max())
print("x_min:", x_min)
print("x_max:", x_max)
# もしどちらかの全範囲を使いたいなら、以下のようにする
# x_min = min(x1.min(), x2.min())
# x_max = max(x1.max(), x2.max())

# x 軸の共通グリッドを作成
x_common = np.arange(x_min, x_max + dx, dx)

# --- 線形補間 ---
y1_i = np.interp(x_common, x1, y1)
y2_i = np.interp(x_common, x2, y2)

# --- 差分計算 ---
diff = y1_i - y2_i
### 1次関数フィッティングして引く場合
coeffs = np.polyfit(x_common, diff, 1)
fit_line = np.polyval(coeffs, x_common)
diff -= fit_line

# --- プロット ---
plt.figure(figsize=(10, 6))

# 上段：2 曲線の比較
plt.subplot(2, 1, 1)
plt.plot(x_common, y1_i, label='raytrace (interp)', color='C0')
plt.plot(x_common, y2_i, label='process data (interp)', color='C1')
plt.scatter(x1, y1, s=10, alpha=0.3, color='C0', label='raytrace raw')
plt.scatter(x2, y2, s=10, alpha=0.3, color='C1', label='process data raw')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

# 下段：差分プロット
plt.subplot(2, 1, 2)
plt.plot(x_common, diff, label='raytrace - process data', color='k')
plt.axhline(0, color='gray', linestyle='--')
plt.legend()
plt.title('difference between raytrace and process data')
plt.xlabel('x')
plt.ylabel('residual')
plt.grid(True)

plt.tight_layout()
plt.show()