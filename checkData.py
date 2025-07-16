import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

folder_path = r"C:\Users\K.Hanada\Desktop\AKBRaytracing\output_20250715_170055_KB_NAdependence"
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
                break

    print("NA_h =", NA_h)

    # Load the data from the text files
    fit_sum = np.loadtxt(os.path.join(folder_full_path, rf'{Key_folder}\fit_sum.txt'))
    pvs = np.loadtxt(os.path.join(folder_full_path, rf'{Key_folder}\pvs.txt'))
    ### extract pv from pvs
    pv = pvs[12]
    # pv = np.loadtxt(os.path.join(folder_full_path, r'AlignmentFit\pv.txt'))

    M = np.loadtxt(os.path.join(folder_full_path, rf'{Key_folder}\M.txt'))
    M_array.append(M)
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
X = np.vstack([np.array(NA_array)**4, np.ones_like(NA_array)]).T
coeffs, residuals, rank, s = np.linalg.lstsq(X, pv_array, rcond=None)
poly_fit = X @ coeffs
# R^2
r2 = r2_score(pv_array, poly_fit)
plt.plot(NA_array, poly_fit, color='r', linestyle='--', label='y = {:.2e}x^4 + {:.2e}\nR^2 = {:.2f}'.format(coeffs[0], coeffs[1], r2))
plt.legend()
plt.title('NA vs PV')
plt.xlabel('NA')
plt.ylabel('PV (lambda)')

### NA**4
plt.figure(figsize=(10, 5))
plt.plot(np.array(NA_array)**4, pv_array, marker='o', linestyle='-', color='b')
plt.title('NA^4 vs PV')
plt.xlabel('NA^4')
plt.ylabel('PV (lambda)')
# plt.show()

M_array = np.array(M_array)
fig2, ax2 = plt.subplots(3,3,figsize=(15, 15))
name_col = ['Oblique Astigmatism', 'Coma', 'Edge Power']
name_row = ['Yaw', 'Roll', 'Pitch']

for M_array_here ,NA in zip(M_array, NA_array):
    print(f"Processing M_array for NA = {NA}")  
    for i in range(3):
        for j in range(3):
        
            ax2[i, j].scatter(NA, M_array_here[i, j],c ='b', marker='o')
            # ax2[i, j].set_title()
            ax2[i, j].set_xlabel('NA')
            ax2[i, j].set_ylabel(f'{name_col[j]} / {name_row[i]}')
plt.title('NA vs Matrix Coefficients')
plt.tight_layout()
plt.show()