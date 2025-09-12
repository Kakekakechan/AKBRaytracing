import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

folder_path = r"C:\Users\K.Hanada\Desktop\AKBRaytracing\output_20250912_114743_KB_sourcepos"
folders = os.listdir(folder_path)
folders = [f for f in folders if os.path.isdir(os.path.join(folder_path, f))]
### folder number sort folder is like 0, 1, 2, ..., 9, 10, 11, ...
folders.sort(key=lambda x: int(x))  # Sort folders by their integer value


num_folders = len(folders)
numsqrt = int(np.ceil(np.sqrt(num_folders)))
print(f"Number of folders: {num_folders}, Grid size: {numsqrt}x{numsqrt}")
fig, axs = plt.subplots(numsqrt, numsqrt, figsize=(15, 15))
axs = axs.ravel()

NA_array = []
pv_array = []
M_array = []
l1h_array = []
l2h_array = []
inc_h_array = []
mlen_h_array = []
wd_v_array = []
inc_v_array = []
mlen_v_array = []
accept_h_array = []
NA_h_array = []
accept_v_array = []
NA_v_array = []
s2f_h_array = []
Key_folder = 'AlignmentFit'
Div_h_array = []
# Key_folder = 'Default'


for i, folder in enumerate(folders):
    folder_full_path = os.path.join(folder_path, folder)
    if not os.path.isdir(folder_full_path):
        continue

    print(f"Processing folder: {folder_full_path}")
    
    l1h = None
    l2h = None
    inc_h = None
    mlen_h = None
    wd_v = None
    inc_v = None
    mlen_v = None
    accept_h = None
    NA_h = None
    accept_v = None
    NA_v = None
    s2f_h = None

    with open(os.path.join(folder_full_path, rf'{Key_folder}\kb_design.txt'), encoding="utf-8") as f:
        for line in f:
            if line.startswith("l1h:"):
                l1h = float(line.split(":")[1].strip())
            if line.startswith("l2h:"):
                l2h = float(line.split(":")[1].strip())
            if line.startswith("inc_h:"):
                inc_h = float(line.split(":")[1].strip())
            if line.startswith("mlen_h:"):
                mlen_h = float(line.split(":")[1].strip())
            if line.startswith("wd_v:"):
                wd_v = float(line.split(":")[1].strip())
            if line.startswith("inc_v:"):
                inc_v = float(line.split(":")[1].strip())
            if line.startswith("mlen_v:"):
                mlen_v = float(line.split(":")[1].strip())
            if line.startswith("accept_h:"):
                accept_h = float(line.split(":")[1].strip())
            if line.startswith("NA_h:"):
                NA_h = float(line.split(":")[1].strip())
            if line.startswith("accept_v:"):
                accept_v = float(line.split(":")[1].strip())
            if line.startswith("NA_v:"):
                NA_v = float(line.split(":")[1].strip())
            if line.startswith("s2f_h:"):
                s2f_h = float(line.split(":")[1].strip())
            if line.startswith("xv_s:"):
                xv_s = float(line.split(":")[1].strip())
            if line.startswith("xv_e:"):
                xv_e = float(line.split(":")[1].strip())
            if line.startswith("yv_s:"):
                yv_s = float(line.split(":")[1].strip())
            if line.startswith("yv_e:"):
                yv_e = float(line.split(":")[1].strip())
                break

    print("NA_h =", NA_h)
    Div_h = np.arctan(yv_s/xv_s) - np.arctan(yv_e/xv_e)
    Div_h_array.append(Div_h)

    # Load the data from the text files
    fit_sum = np.loadtxt(os.path.join(folder_full_path, rf'{Key_folder}\fit_sum.txt'))
    pvs = np.loadtxt(os.path.join(folder_full_path, rf'{Key_folder}\pvs.txt'))
    ### extract pv from pvs
    pv = pvs[12]
    # pv = np.loadtxt(os.path.join(folder_full_path, r'AlignmentFit\pv.txt'))

    # M = np.loadtxt(os.path.join(folder_full_path, rf'{Key_folder}\M.txt'))
    # M_array.append(M)
    # fit sumのimshow
    im = axs[i].imshow(fit_sum, aspect='auto', cmap='jet')
    axs[i].set_title(f'Fit Sum - {folder}')
    axs[i].set_xlabel('Index')
    axs[i].set_ylabel('Value')


    NA_array.append(NA_h)
    l1h_array.append(l1h)
    l2h_array.append(l2h)
    inc_h_array.append(inc_h)
    mlen_h_array.append(mlen_h)
    wd_v_array.append(wd_v)
    inc_v_array.append(inc_v)
    mlen_v_array.append(mlen_v)
    accept_h_array.append(accept_h)
    NA_h_array.append(NA_h)
    accept_v_array.append(accept_v)
    NA_v_array.append(NA_v)
    s2f_h_array.append(s2f_h)

    pv_array.append(pv)
    fig.colorbar(im, ax=axs[i], label='Wave error (lambda)')
    # plt.savefig(os.path.join(folder_full_path, 'fit_sum_heatmap.png'))
    # plt.show()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(NA_array, pv_array, marker='o', linestyle='', color='b')

# Polyfit: only 4th and 0th order (a*x^4 + b)
# X = np.vstack([np.array(NA_array)**4, np.ones_like(NA_array)]).T
# coeffs, residuals, rank, s = np.linalg.lstsq(X, pv_array, rcond=None)
# poly_fit = X @ coeffs
coeffs = np.polyfit(NA_array, pv_array, 4)
poly_fit = np.polyval(coeffs, NA_array)
print("Fitted coefficients (4th order polynomial):", coeffs)
# R^2
r2 = r2_score(pv_array, poly_fit)
plt.plot(NA_array, poly_fit, color='r', linestyle='--', label='y = {:.2e}x^4 + {:.2e}x^3 + {:.2e}x^2 + {:.2e}x + {:.2e}\n$R^2$={:.4f}'.format(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], r2))
plt.legend()
plt.title('NA vs PV_legendre')
plt.xlabel('NA')
plt.ylabel('PV (lambda)')

def plt_func(NA_array, pv_array, name, folder_path):
    ### NA**4
    plt.figure(figsize=(6, 4))
    NA_m1 = np.array(NA_array)**-1
    NA_2 = np.array(NA_array)**2
    NA_3 = np.array(NA_array)**3
    NA_4 = np.array(NA_array)**4
    coeffsm1 = np.polyfit(NA_m1, pv_array, 1)
    poly_fitm1 = np.polyval(coeffsm1, NA_m1)
    r2_m1 = r2_score(pv_array, poly_fitm1)
    coeffs1 = np.polyfit(NA_array, pv_array, 1)
    poly_fit1 = np.polyval(coeffs1, NA_array)
    r2_1 = r2_score(pv_array, poly_fit1)
    coeffs2 = np.polyfit(NA_2, pv_array, 1)
    poly_fit2 = np.polyval(coeffs2, NA_2)
    r2_2 = r2_score(pv_array, poly_fit2)
    coeffs3 = np.polyfit(NA_3, pv_array, 1)
    poly_fit3 = np.polyval(coeffs3, NA_3)
    r2_3 = r2_score(pv_array, poly_fit3)
    coeffs4 = np.polyfit(NA_4, pv_array, 1)
    poly_fit4 = np.polyval(coeffs4, NA_4)
    r2_4 = r2_score(pv_array, poly_fit4)

    plt.figure(figsize=(6, 4))
    plt.plot(NA_m1, pv_array, marker='o', linestyle='', color='b')
    plt.plot(NA_m1, poly_fitm1, color='r', linestyle='--', label='y = {:.2e}x + {:.2e}\n$R^2$={:.4f}'.format(coeffsm1[0], coeffsm1[1], r2_m1))
    plt.legend()
    plt.title(f'{name}^-1 vs PV_legendre')
    plt.xlabel(f'{name}^-1')
    plt.ylabel('PV (lambda)')
    ### 軸目盛数字大きく
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(folder_path, f'{name}m1_vs_PV_legendre.png'), dpi=300)

    plt.figure(figsize=(6, 4))
    plt.plot(NA_array, pv_array-poly_fitm1)
    plt.title(f'{name}^-1 vs Residuals of PV_legendre')
    plt.xlabel(f'{name}^1')
    plt.ylabel('Residuals (lambda)')
    # plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(NA_array, pv_array, marker='o', linestyle='', color='b')
    plt.plot(NA_array, poly_fit1, color='r', linestyle='--', label='y = {:.2e}x + {:.2e}\n$R^2$={:.4f}'.format(coeffs1[0], coeffs1[1], r2_1))
    plt.legend()
    plt.title(f'{name}^1 vs PV_legendre')
    plt.xlabel(f'{name}^1')
    plt.ylabel('PV (lambda)')
    ### 軸目盛数字大きく
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(folder_path, f'{name}1_vs_PV_legendre.png'), dpi=300)
    
    plt.figure(figsize=(6, 4))
    plt.plot(NA_2, pv_array, marker='o', linestyle='', color='b')
    plt.plot(NA_2, poly_fit2, color='r', linestyle='--', label='y = {:.2e}x + {:.2e}\n$R^2$={:.4f}'.format(coeffs2[0], coeffs2[1], r2_2))
    plt.legend()
    plt.title(f'{name}^2 vs PV_legendre')
    plt.xlabel(f'{name}^2')
    plt.ylabel('PV (lambda)')
    ### 軸目盛数字大きく
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(folder_path, f'{name}2_vs_PV_legendre.png'), dpi=300)

    plt.figure(figsize=(6, 4))
    plt.plot(NA_3, pv_array, marker='o', linestyle='', color='b')
    plt.plot(NA_3, poly_fit3, color='r', linestyle='--', label='y = {:.2e}x + {:.2e}\n$R^2$={:.4f}'.format(coeffs3[0], coeffs3[1], r2_3))
    plt.legend()
    plt.title(f'{name}^3 vs PV_legendre')
    plt.xlabel(f'{name}^3')
    plt.ylabel('PV (lambda)')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(os.path.join(folder_path, f'{name}3_vs_PV_legendre.png'), dpi=300)

    plt.figure(figsize=(6, 4))
    plt.plot(NA_4, pv_array, marker='o', linestyle='-', color='b')
    plt.plot(NA_4, poly_fit4, color='r', linestyle='--', label='y = {:.2e}x + {:.2e}\n$R^2$={:.4f}'.format(coeffs4[0], coeffs4[1], r2_4))
    plt.legend()
    plt.title(f'{name}^4 vs PV_legendre')
    plt.xlabel(f'{name}^4')
    plt.ylabel('PV (lambda)')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(os.path.join(folder_path, f'{name}4_vs_PV_legendre.png'), dpi=300)
# plt.show()

# plt_func(NA_array, pv_array, 'NA', folder_path)
# plt_func(np.array(Div_h_array)*1e6, pv_array, 'Div_h', folder_path)

plt_func(np.array(l1h_array), pv_array, 'l1h', folder_path)

# M_array = np.array(M_array)
# fig2, ax2 = plt.subplots(3,3,figsize=(15, 15))
# name_col = ['Oblique Astigmatism', 'Coma', 'Edge Power']
# name_row = ['Yaw', 'Roll', 'Pitch']

# for M_array_here ,NA in zip(M_array, NA_array):
#     print(f"Processing M_array for NA = {NA}")  
#     for i in range(3):
#         for j in range(3):
        
#             ax2[i, j].scatter(NA, M_array_here[i, j],c ='b', marker='o')
#             # ax2[i, j].set_title()
#             ax2[i, j].set_xlabel('NA')
#             ax2[i, j].set_ylabel(f'{name_col[j]} / {name_row[i]}')
# plt.title('NA vs Matrix Coefficients')
# plt.tight_layout()
plt.show()