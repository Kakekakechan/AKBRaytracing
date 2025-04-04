import os
import shutil
import sys
import numpy as np
from numpy import abs, sin, cos, tan, arcsin, arccos, arctan, sqrt, pi
from datetime import datetime
import h5py
import gc
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Add tqdm import for progress bar
from concurrent.futures import ThreadPoolExecutor  # Add import for parallel execution
from concurrent.futures import ProcessPoolExecutor  # Replace Pool with ProcessPoolExecutor
from numba import njit, prange  # Add numba imports

### CPU ###
class WaveField3D:
    def __init__(self, num, _lambda, wave_num_H, wave_num_V):
        # 各値をNumPy配列で初期化
        self.u = np.zeros(num, dtype=np.complex128)
        self.x = np.zeros(num, dtype=np.float64)
        self.y = np.zeros(num, dtype=np.float64)
        self.z = np.zeros(num, dtype=np.float64)

        self.lambda_ = np.float64(_lambda)
        self.wave_num_H = wave_num_H
        self.wave_num_V = wave_num_V

    def setdata(self, data):
        # データをNumPy配列に変換
        self.x = np.array(data[0, :], dtype=np.float64)
        self.y = np.array(data[1, :], dtype=np.float64)
        self.z = np.array(data[2, :], dtype=np.float64)

    def set_ds(self, data):
        self.ds = np.array(data, dtype=np.float64)

    def forward_propagation(self, u_back, num_cores=None):
        k = 2.0 * np.pi / self.lambda_  # 波数の計算
        start_time = time.time()  # 開始時間を記録
        self.u = forward_propagation_numpy_batch(
            self.x, self.y, self.z,
            u_back.x, u_back.y, u_back.z,
            u_back.u,  # 複素数配列を直接渡す
            k, u_back.ds,  # 事前計算したdsを渡す
            num_cores=num_cores  # コア数を指定
        )
        end_time = time.time()  # 終了時間を記録

        # 実行時間を表示
        elapsed_time = end_time - start_time
        print(f"計算時間: {elapsed_time:.6f} 秒")

def compute_u(i, x, y, z, u_back_x, u_back_y, u_back_z, u_back_u, k):
    dist = np.sqrt(
        (x[i] - u_back_x) ** 2 +
        (y[i] - u_back_y) ** 2 +
        (z[i] - u_back_z) ** 2
    )
    amplitude = 1. / dist
    phase = -k * dist
    factor = amplitude * np.exp(1j * phase)
    return np.sum(factor * u_back_u)

def compute_u_wrapper(args):
    """
    Wrapper function to unpack arguments for compute_u.
    """
    return compute_u(*args)

@njit(parallel=True)
def compute_u_parallel(x, y, z, u_back_x, u_back_y, u_back_z, u_back_u, k):
    total = len(x)
    u = np.zeros(total, dtype=np.complex128)
    for i in prange(total):
        dist = np.sqrt(
            (x[i] - u_back_x) ** 2 +
            (y[i] - u_back_y) ** 2 +
            (z[i] - u_back_z) ** 2
        )
        amplitude = 1. / dist
        phase = -k * dist
        factor = amplitude * np.exp(1j * phase)
        u[i] = np.sum(factor * u_back_u)
    return u

def forward_propagation_numpy_batch(x, y, z, u_back_x, u_back_y, u_back_z, u_back_u, k, ds, num_cores=None):
    """
    Perform forward propagation using parallel processing.

    Parameters:
    x, y, z: Coordinates of the current wave field.
    u_back_x, u_back_y, u_back_z: Coordinates of the previous wave field.
    u_back_u: Complex amplitude of the previous wave field.
    k: Wave number.
    ds: Precomputed ds values.
    num_cores: Number of cores to use for parallel processing (optional).

    Returns:
    u: Computed complex amplitude of the current wave field.
    """
    u_back_u = u_back_u * ds
    del ds

    if num_cores:
        import os
        os.environ["NUMBA_NUM_THREADS"] = str(num_cores)
        print(f"Using {num_cores} cores for parallel processing.")
    else:
        print("Using default number of cores for parallel processing.")

    print("Using numba prange for parallel processing.")
    print("Progress will be displayed in 1/100 increments.")
    total = len(x)
    progress_step = max(total // 100, 1)  # Calculate step for 1/100 progress

    u = np.zeros(total, dtype=np.complex128)
    for i in range(0, total, progress_step):
        u[i:i+progress_step] = compute_u_parallel(
            x[i:i+progress_step], y[i:i+progress_step], z[i:i+progress_step],
            u_back_x, u_back_y, u_back_z, u_back_u, k
        )
        print(f"Progress: {((i + progress_step) / total) * 100:.2f}%")
    return u

##########
# Add a cache dictionary to store loaded files
file_cache = {}

def load_file(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    """
    ファイルパスを受け取り、拡張子に応じて読み込む関数。

    Parameters:
    file_path (str): 読み込むファイルのパス

    Returns:
    内容またはデータ（適切な場合）
    """
    # Check if the file is already cached
    if file_path in file_cache:
        return file_cache[file_path]

    # ファイルが存在するか確認
    if os.path.exists(file_path):
        print(f"Reading file: {file_path}")

        # ファイル拡張子によって処理を分ける
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                content = file.read()
            file_cache[file_path] = content  # Cache the content
            return content

        elif file_path.endswith('.npy'):
            data = np.load(file_path)
            file_cache[file_path] = data  # Cache the data
            return data

        elif file_path.endswith('.npz'):
            data = np.load(file_path)
            file_cache[file_path] = data  # Cache the data
            return data

        else:
            print(f"Unsupported file type: {file_path}")
            return None
    else:
        print(f"File not found: {file_path}")
        return None

def load_npz_data(filename):
    """
    指定したファイル名の npz ファイルを読み込み、データを返す関数。

    Parameters:
    filename (str): 読み込みたい npz ファイルの名前（パスを含む）。

    Returns:
    numpy.ndarray: 読み込んだデータの配列。ファイルが存在しない場合は None を返す。
    """
    if os.path.exists(filename):
        with np.load(filename) as data:  # コンテキストマネージャで開くと自動的に閉じる
            return data['data']  # 保存時のキーワード 'data' でデータを取得
    else:
        print(f"{filename} が存在しません。")
        return None

if __name__ == '__main__':
    # folder_path = r'\\HPC-PC3\Users\OP_User\Desktop\akb\output_20241129_4096_4096'  # 読み込みたいフォルダ名を指定
    folder_path = r'output_20250404_sNAAKB701'
    file_names = ['points_source.npy','points_M1.npy','points_M2.npy','points_gridImage.npy','points_gridDefocus.npy']  # 読み込みたいファイル名をリストで指定
    source = load_file(folder_path, file_names[0])
    vmirr_hyp = load_file(folder_path, file_names[1])

    # フォルダ名として使用する現在時刻の文字列を取得
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory_name = f"output_{timestamp}"
    # 新しいフォルダを作成
    os.makedirs(directory_name, exist_ok=True)

    # データを新規フォルダ内に保存
    np.save(os.path.join(directory_name, 'points_source.npy'), source)
    np.save(os.path.join(directory_name, 'points_M1.npy'), vmirr_hyp)

    # ファイルを読み込んで値を抽出
    with open(os.path.join(folder_path, 'calculation_conditions.txt'), 'r') as file:
        for line in file:
            if 'grid pix_y:' in line:
                pix_y = int(line.split(':')[1].strip())  # 値を整数として取得
                ray_num_H1 = int(line.split(':')[1].strip())  # 値を整数として取得
                ray_num_H2 = int(line.split(':')[1].strip())  # 値を整数として取得
            elif 'grid pix_z:' in line:
                pix_z = int(line.split(':')[1].strip())  # 値を整数として取得
                ray_num_V1 = int(line.split(':')[1].strip())  # 値を整数として取得
                ray_num_V2 = int(line.split(':')[1].strip())  # 値を整数として取得
            elif 'grid pix_H:' in line:
                ray_num_H1 = int(line.split(':')[1].strip())  # 値を整数として取得
                ray_num_V2 = int(line.split(':')[1].strip())  # 値を整数として取得
            elif 'grid pix_V:' in line:
                ray_num_V1 = int(line.split(':')[1].strip())  # 値を整数として取得
                ray_num_H2 = int(line.split(':')[1].strip())  # 値を整数として取得
            elif 'grid pix_H1:' in line:
                ray_num_H1 = int(line.split(':')[1].strip())  # 値を整数として取得
            elif 'grid pix_V1:' in line:
                ray_num_V1 = int(line.split(':')[1].strip())  # 値を整数として取得
            elif 'grid pix_H2:' in line:
                ray_num_H2 = int(line.split(':')[1].strip())  # 値を整数として取得
            elif 'grid pix_V2:' in line:
                ray_num_V2 = int(line.split(':')[1].strip())  # 値を整数として取得
            elif 'option_AKB:' in line:
                option_AKB = line.split(':')[1].strip().lower() == "true"
                print("option_AKB", option_AKB)
            elif 'option_HighNA:' in line:
                option_HighNA = line.split(':')[1].strip().lower() == "true"
                print("option_HighNA", option_HighNA)

    shutil.copy(os.path.join(folder_path, 'calculation_conditions.txt'), os.path.join(directory_name, 'calculation_conditions.txt'))

    print(source.reshape(3,1))
    if option_HighNA:
        wavelength  =13.5e-9 # 波長
    else:
        wavelength  =13.5e-9 * 1e-1 # 波長
    print("WaveLength nm",wavelength*1e9)
    LightSource = WaveField3D(1, wavelength,1,1)
    LightSource.u[0] = 1.

    # print('vmirr_hyp',vmirr_hyp[:,:10])
    # sys.pause(10)

    Field_1 = WaveField3D(vmirr_hyp.shape[1], wavelength,ray_num_H1,ray_num_V1)

    LightSource.setdata(source.reshape(3,1))
    LightSource.set_ds(np.ones(1))
    Field_1.setdata(vmirr_hyp)


    # M1計算
    data = load_npz_data("complex_data_M1.npz")
    if data is not None:
        print("complex_data_M1.npz を正常に読み込みました。")
        Field_1.u = data
    else:
        print("source => M1 計算を行います")
        num_cores = 17  # Adjust this value as needed
        Field_1.forward_propagation(LightSource, num_cores=num_cores)
        np.savez_compressed(os.path.join(directory_name, "complex_data_M1.npz"), data=Field_1.u)
        del data, LightSource # メモリ解放


    # sys.exit()

    # M2計算
    hmirr_hyp = load_file(folder_path, file_names[2])
    np.save(os.path.join(directory_name, 'points_M2.npy'), hmirr_hyp)
    Field_2 = WaveField3D(hmirr_hyp.shape[1], wavelength,ray_num_H2,ray_num_V2)
    Field_2.setdata(hmirr_hyp)
    Field_1.set_ds(vmirr_hyp[3, :])
    data = load_npz_data("complex_data_M2.npz")
    if data is not None:
        print("complex_data_M2.npz を正常に読み込みました。")
        Field_2.u = data
    else:
        print("M1 => M2 計算を行います")

        Field_2.forward_propagation(Field_1)
        np.savez_compressed(os.path.join(directory_name, "complex_data_M2.npz"), data=Field_2.u)
        del Field_1, data  # 不要な変数を削除

    # 画像グリッド計算
    Image_grid_org = load_file(folder_path, file_names[3])

    if option_AKB:
        # M3計算
        vmirr_ell = load_file(folder_path, 'points_M3.npy')
        np.save(os.path.join(directory_name, 'points_M3.npy'), vmirr_ell)
        Field_3 = WaveField3D(vmirr_ell.shape[1], wavelength,ray_num_H1,ray_num_V1)
        Field_3.setdata(vmirr_ell)
        Field_2.set_ds(hmirr_hyp[3, :])
        data = load_npz_data("complex_data_M3.npz")
        if data is not None:
            print("complex_data_M3.npz を正常に読み込みました。")
            Field_3.u = data
        else:
            print("M2 => M3 計算を行います")

            Field_3.forward_propagation(Field_2)
            np.savez_compressed(os.path.join(directory_name, "complex_data_M3.npz"), data=Field_3.u)
            del Field_2, data  # 不要な変数を削除

        # M4計算
        hmirr_ell = load_file(folder_path, 'points_M4.npy')
        np.save(os.path.join(directory_name, 'points_M4.npy'), hmirr_ell)
        Field_4 = WaveField3D(hmirr_ell.shape[1], wavelength,ray_num_H2,ray_num_V2)
        Field_4.setdata(hmirr_ell)
        Field_3.set_ds(vmirr_ell[3, :])
        data = load_npz_data("complex_data_M4.npz")
        if data is not None:
            print("complex_data_M4.npz を正常に読み込みました。")
            Field_4.u = data
        else:
            print("M3 => M4 計算を行います")

            Field_4.forward_propagation(Field_3)
            np.savez_compressed(os.path.join(directory_name, "complex_data_M4.npz"), data=Field_4.u)
            del Field_3, data  # 不要な変数を削除
    #リサイズ
    if True:
        mean_val = [np.mean(Image_grid_org[0,:]),np.mean(Image_grid_org[1,:]),np.mean(Image_grid_org[2,:])]
        Image_grid_org[0,:] = (Image_grid_org[0,:] - mean_val[0])*2. + mean_val[0]
        Image_grid_org[1,:] = (Image_grid_org[1,:] - mean_val[1])*2. + mean_val[1]
        Image_grid_org[2,:] = (Image_grid_org[2,:] - mean_val[2])*2. + mean_val[2]

    np.save(os.path.join(directory_name, 'points_gridImage.npy'), Image_grid_org)
    Image_grid = WaveField3D(Image_grid_org.shape[1], wavelength, pix_y,pix_z)
    Image_grid.setdata(Image_grid_org)
    if option_AKB:
        Field_4.set_ds(hmirr_ell[3, :])
        print("M4 => Image 計算を行います")
        Image_grid.forward_propagation(Field_4)
    else:
        Field_2.set_ds(hmirr_hyp[3, :])
        print("M2 => Image 計算を行います")
        Image_grid.forward_propagation(Field_2)
    np.savez_compressed(os.path.join(directory_name, "complex_data_Image.npz"), data=Image_grid.u)
    del Image_grid # メモリ解放

    # 画像グリッド計算
    Image_grid_org2 = load_file(folder_path, file_names[4])

    #リサイズ
    if True:
        mean_val2 = [np.mean(Image_grid_org2[0,:]),np.mean(Image_grid_org2[1,:]),np.mean(Image_grid_org2[2,:])]
        Image_grid_org2[0,:] = (Image_grid_org2[0,:] - mean_val2[0]) + mean_val2[0]
        Image_grid_org2[1,:] = (Image_grid_org2[1,:] - mean_val2[1]) + mean_val2[1]
        Image_grid_org2[2,:] = (Image_grid_org2[2,:] - mean_val2[2]) + mean_val2[2]

    np.save(os.path.join(directory_name, 'points_gridImage2.npy'), Image_grid_org2)
    Image_grid2 = WaveField3D(Image_grid_org2.shape[1], wavelength, pix_y,pix_z)
    Image_grid2.setdata(Image_grid_org2)

    if option_AKB:
        print("M4 => Image2 計算を行います")
        Image_grid2.forward_propagation(Field_4)
    else:
        print("M2 => Image2 計算を行います")
        Image_grid2.forward_propagation(Field_2)
    np.savez_compressed(os.path.join(directory_name, "complex_data_Image2.npz"), data=Image_grid2.u)

    if option_AKB:
        del Field_4, Image_grid2  # メモリ解放
    else:
        del Field_2, Image_grid2  # メモリ解放

    print("すべての計算が完了しました。")
