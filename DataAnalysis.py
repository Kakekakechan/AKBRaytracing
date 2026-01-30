import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import legendre_fit as lf
base_dir = r"C:\Users\DN0089\Desktop\git\AKBRaytracing\Hyp_v_single_pitch"
param_number = 2
# ** で再帰的探索、末尾に / をつけることでディレクトリのみにマッチさせる
# recursive=True が必要
subfolders = glob.glob(os.path.join(base_dir, "**/"), recursive=True)

print(subfolders)
print(len(subfolders))

n = len(subfolders)
inner_products = []
params = []

for folder in subfolders:
    param_file = os.path.join(folder, "optical_params.txt")
    inner_product_file = os.path.join(folder, "inner_products.csv")

    if not os.path.isfile(param_file) or not os.path.isfile(inner_product_file):
        continue
    
    with open(param_file, 'r') as pf:
        param_values = [float(line.split(":")[1].strip()) for line in pf if ":" in line]
        params.append(param_values)

    inner_products_values = np.loadtxt(inner_product_file, delimiter=',')
    inner_products.append(inner_products_values)
params = np.array(params)
inner_products = np.array(inner_products)

orders_file = os.path.join(folder, "orders.csv")
orders = np.loadtxt(orders_file, delimiter=',')
orders = orders.astype(int)
print("Parameters shape:", params.shape)
print("Inner Products shape:", inner_products.shape)
coeffs = []
plt.figure(figsize=(10, 6))
for n, order in enumerate(orders):
    print(order)
    if order[1] == 0:
        print(n)
        param_here = params[:, param_number]
        inner_product_here = inner_products[:, n]
        coeff = np.polyfit(param_here, inner_product_here, 1)
        coeffs.append(coeff)
        plt.plot(param_here, inner_product_here, 'o-', label=f'ny={order[0]},nx={order[1]} fit: {coeff[0]:.4e}x + {coeff[1]:.4e}')
        sample_legendre = lf.output_legendre_data(inner_products[-1, n], order)
        pv = (np.nanmax(sample_legendre) - np.nanmin(sample_legendre)) * np.sign(inner_products[-1, n])
        print(f'Order ny={order[0]}, nx={order[1]}: innerproduct = {inner_products[-1, n]:.4e} PV = {pv:.4e} lambda')
plt.legend()
coeffs = np.array(coeffs)
np.savetxt(os.path.join(base_dir, f"legendre_fit_coeffs_param{param_number}.csv"), coeffs, delimiter=',')
# n=10
# sample_legendre = lf.output_legendre_data(inner_products[-1, n], orders[10])
# plt.figure(figsize=(12, 8))
# plt.imshow(sample_legendre, extent=(-1, 1, -1, 1), cmap='jet', vmin=-1/4, vmax=1/4)
# plt.colorbar(label='Wave error (lambda)')
# plt.title(f'Legendre ny={orders[n][0]},nx={orders[n][1]} \n inner product: {inner_products[-1, n]:.4e}')
plt.show()