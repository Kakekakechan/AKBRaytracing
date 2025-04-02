import os
import shutil
import sys
import numpy as np
from numpy import abs, sin, cos,tan, arcsin,arccos,arctan, sqrt, pi
import cupy as cp
from datetime import datetime
import h5py
import gc
import time

### GPU ###
print(cp.cuda.runtime.getDeviceCount())  # 使用可能なGPU数を表示
class WaveField3D:
    def __init__(self, num, _lambda, wave_num_H, wave_num_V):
        # 各値をCuPy配列で初期化
        ### 64
        self.u = cp.zeros(num, dtype=cp.complex128)
        self.x = cp.zeros(num, dtype=cp.float64)
        self.y = cp.zeros(num, dtype=cp.float64)
        self.z = cp.zeros(num, dtype=cp.float64)
        ### 32
        # self.u = cp.zeros(num, dtype=cp.complex64)
        # self.x = cp.zeros(num, dtype=cp.float32)
        # self.y = cp.zeros(num, dtype=cp.float32)
        # self.z = cp.zeros(num, dtype=cp.float32)

        self.lambda_ = cp.float64(_lambda)
        self.wave_num_H = wave_num_H
        self.wave_num_V = wave_num_V
        # self.ds = cp.ones(num, dtype=cp.float64)

    def setdata(self, data):
        # データをCuPy配列に変換
        ### 64
        self.x = cp.array(data[0, :], dtype=cp.float64)
        self.y = cp.array(data[1, :], dtype=cp.float64)
        self.z = cp.array(data[2, :], dtype=cp.float64)
        ### 32
        # self.x = cp.array(data[0, :], dtype=cp.float32)
        # self.y = cp.array(data[1, :], dtype=cp.float32)
        # self.z = cp.array(data[2, :], dtype=cp.float32)

    def set_ds(self, data):
        self.ds = cp.array(data, dtype=cp.float64)
        # self.ds = cp.ones(data.shape[1], dtype=cp.float64)
        # if data.shape[1] > 1:
        #     ds_np = calc_dS(data,self.wave_num_V, self.wave_num_H)
        #     self.ds = cp.asarray(ds_np)


    # def _calculate_ds(self, u_back):
    #     """面積要素 ds を計算してキャッシュ"""
    #     num_points = len(u_back.x)
    #     ds = cp.ones(num_points, dtype=cp.float64) ### 64
    #
    #     # if num_points > 1:
    #     #     dx1 = cp.diff(u_back.x.reshape(u_back.wave_num_V, u_back.wave_num_H), axis=1)
    #     #     dy1 = cp.diff(u_back.y.reshape(u_back.wave_num_V, u_back.wave_num_H), axis=1)
    #     #     dz1 = cp.diff(u_back.z.reshape(u_back.wave_num_V, u_back.wave_num_H), axis=1)
    #     #     dx2 = cp.diff(u_back.x.reshape(u_back.wave_num_V, u_back.wave_num_H), axis=0)
    #     #     dy2 = cp.diff(u_back.y.reshape(u_back.wave_num_V, u_back.wave_num_H), axis=0)
    #     #     dz2 = cp.diff(u_back.z.reshape(u_back.wave_num_V, u_back.wave_num_H), axis=0)
    #     #
    #     #     # ds_y = cp.sqrt(dx_y[1:, :]**2 + dy[1:, :]**2)
    #     #     # ds_z = cp.sqrt(dx_z[:, 1:]**2 + dz[:, 1:]**2)
    #     #     vector1 = cp.stack((dx1[1:, :], dy1[1:, :], dz1[1:, :]), axis=0)
    #     #     vector2 = cp.stack((dx2[:, 1:], dy2[:, 1:], dz2[:, 1:]), axis=0)
    #     #     print('vector1',vector1[0,0])
    #     #     print('vector1',vector1.shape)
    #     #     cross_product = cp.cross(vector1, vector2, axis=0)
    #     #     ds_0 = cp.linalg.norm(cross_product, axis=0)
    #     #
    #     #     ds_matrix = cp.ones((u_back.wave_num_V, u_back.wave_num_H), dtype=cp.float64) ### 64
    #     #     # ds_matrix = cp.ones((u_back.wave_num_V, u_back.wave_num_H), dtype=cp.float32) ### 32
    #     #     ds_matrix[1:, 1:] = ds_0
    #     #     ds_matrix[0, 1:] = ds_matrix[1, 1:]
    #     #     ds_matrix[:, 0] = ds_matrix[:, 1]
    #     #     ds = ds_matrix.flatten()
    #     #     print('ds',ds[0])
    #
    #     if num_points > 1:
    #         dS = calc_dS(vmirr_hyp,size_v1, size_h1)
    #
    #     return ds

    def forward_propagation(self, u_back):
        k = 2.0 * cp.pi / self.lambda_  # 波数の計算
        start_time = time.time()  # 開始時間を記録
        # GPUで計算を呼び出す（バッチ処理対応）
        # 面積要素（ds）を初期化

        # self.ds = self._calculate_ds(u_back)

        print('u_back.u',u_back.u[0])
        print('u_back.ds',u_back.ds[0])
        print('k',k)

        self.u = forward_propagation_cupy_batch(
            self.x, self.y, self.z,
            u_back.x, u_back.y, u_back.z,
            u_back.u,  # 複素数配列を直接渡す
            k, u_back.ds  # 事前計算したdsを渡す
        )
        end_time = time.time()  # 終了時間を記録

        # 実行時間を表示
        elapsed_time = end_time - start_time
        print(f"計算時間: {elapsed_time:.6f} 秒")

def forward_propagation_cupy_batch(x, y, z, u_back_x, u_back_y, u_back_z, u_back_u, k, ds):
    if not isinstance(u_back_u, cp.ndarray):
        u_back_u = cp.asarray(u_back_u)
    u_back_u = u_back_u * ds
    del ds
    num = len(x)
    num_back = len(u_back_x)

    # GPUメモリの空き容量を取得してバッチサイズを決定
    mem_info = cp.cuda.Device().mem_info
    free_mem = mem_info[0]
    overhead = 4.0  # メモリの余裕率
    element_size = 16  # 複素数1要素あたりのメモリ使用量（バイト）
    max_batch_size = int((free_mem / overhead) / element_size / num_back)
    print(f"free memory: {free_mem}")
    print(f"Max batch size based on free memory: {max_batch_size}")

    # 出力配列を初期化（複素数型）
    u = cp.zeros(num, dtype=cp.complex128)  # 精度許容ならcomplex128を使用

    # ストリームを準備
    compute_stream = cp.cuda.Stream()
    cleanup_stream = cp.cuda.Stream()

    # バッチごとに計算
    for i in range(0, num, max_batch_size):
        batch_end = min(i + max_batch_size, num)
        if i % 23 == 0:
            print(f"batch: {i}/{num}")

        # 部分データをスライス
        x_batch = x[i:batch_end]
        y_batch = y[i:batch_end]
        z_batch = z[i:batch_end]
        # ds_batch = ds[i:batch_end]

        # メインストリームで計算を実行
        with compute_stream:
            dist = cp.sqrt(
                (x_batch[:, None] - u_back_x[None, :]) ** 2 +
                (y_batch[:, None] - u_back_y[None, :]) ** 2 +
                (z_batch[:, None] - u_back_z[None, :]) ** 2
            )
            # amplitude = ds_batch[None, :] / dist
            amplitude = 1. / dist
            phase = -k * dist
            factor = amplitude * cp.exp(1j * phase)
            interaction = cp.dot(u_back_u, factor.T)
            u[i:batch_end] = interaction

        # メモリ解放を別ストリームで処理
        with cleanup_stream:
            del x_batch, y_batch, z_batch, dist, amplitude, phase, factor, interaction
            cp.get_default_memory_pool().free_all_blocks()

        # 同期ポイントを設定
        compute_stream.synchronize()
        cleanup_stream.synchronize()

    return u

def calc_dS(points,ray_num_V, ray_num_H):
    grid_points = points.reshape(3, ray_num_V, ray_num_H)
    """
    grid_points : (3, l, k) の3次元座標
    """
    l, k = grid_points.shape[1:]

    # 面積格納
    dS = np.zeros((l, k))

    # 内部点の面積計算
    for i in range(1, l-1):
        for j in range(1, k-1):
            p = grid_points[:, i, j]

            p_right = grid_points[:, i, j+1]
            p_left = grid_points[:, i, j-1]
            p_up = grid_points[:, i-1, j]
            p_down = grid_points[:, i+1, j]

            triangles = [
                (p, p_right, p_up),
                (p, p_up, p_left),
                (p, p_left, p_down),
                (p, p_down, p_right)
            ]

            area_sum = 0.0

            for v0, v1, v2 in triangles:
                e1 = v1 - v0
                e2 = v2 - v0
                area = np.linalg.norm(np.cross(e1, e2)) / 2
                area_sum += area

            dS[i, j] = area_sum

    # 縁の補間処理（縁以外の最近傍をコピー）
    for i in range(l):
        for j in range(k):
            if i == 0:           # 上端
                dS[i, j] = dS[1, j]
            elif i == l-1:       # 下端
                dS[i, j] = dS[l-2, j]
            elif j == 0:         # 左端
                dS[i, j] = dS[i, 1]
            elif j == k-1:       # 右端
                dS[i, j] = dS[i, k-2]

    # 4隅の補間処理（これは個別指定でOK）
    dS[0, 0]     = dS[1, 1]      # 左上
    dS[0, k-1]   = dS[1, k-2]    # 右上
    dS[l-1, 0]   = dS[l-2, 1]    # 左下
    dS[l-1, k-1] = dS[l-2, k-2]  # 右下

    return dS.flatten()

def split_wave_field(field, num_parts):
    """
    WaveField3D インスタンスを num_parts 個に分割する関数。

    Parameters:
    - field (WaveField3D): 分割対象の WaveField3D インスタンス
    - num_parts (int): 分割数

    Returns:
    - List[WaveField3D]: 分割された WaveField3D インスタンスのリスト
    """
    num_points = len(field.x)
    part_size = num_points // num_parts
    wave_fields = []

    for i in range(num_parts):
        start = i * part_size
        end = (i + 1) * part_size if i != num_parts - 1 else num_points

        # 新しい WaveField3D インスタンスを作成
        part_field = WaveField3D(end - start, field.lambda_, field.wave_num_H, field.wave_num_V)
        part_field.x = field.x[start:end]
        part_field.y = field.y[start:end]
        part_field.z = field.z[start:end]
        part_field.ds = field.ds[start:end]  # 面積要素も分割
        part_field.u = cp.zeros(end - start, dtype=cp.complex128)  # 結果用の初期化
        wave_fields.append(part_field)

    return wave_fields
##########
def load_file(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    """
    ファイルパスを受け取り、拡張子に応じて読み込む関数。

    Parameters:
    file_path (str): 読み込むファイルのパス

    Returns:
    内容またはデータ（適切な場合）
    """
    # ファイルが存在するか確認
    if os.path.exists(file_path):
        print(f"Reading file: {file_path}")

        # ファイル拡張子によって処理を分ける
        if file_path.endswith('.txt'):
            # テキストファイル（.txt）の読み込み
            with open(file_path, 'r') as file:
                content = file.read()
            return content

        elif file_path.endswith('.npy'):
            # NumPyのバイナリファイル（.npy）の読み込み
            data = np.load(file_path)
            return data

        elif file_path.endswith('.npz'):
            # NumPyの圧縮バイナリファイル（.npz）の読み込み
            data = np.load(file_path)
            # .npzファイルは複数の配列を持っているので、キーを確認
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

def SplitGPUdata(Field_2,num_parts):
    # 分割設定
    # num_parts = 5  # 分割数（調整可能）
    part_size = Field_2.u.size // num_parts  # 各パートのサイズ
    Field_2_parts = []

    # Field_2 を NumPy に変換して CPU に保存
    Field_2_np = {
        "x": Field_2.x.get(),  # CuPy -> NumPy
        "y": Field_2.y.get(),
        "z": Field_2.z.get(),
        "ds": Field_2.ds.get(),
        "u": np.zeros_like(Field_2.x.get(), dtype=np.complex128),  # 結果を格納
    }
    del Field_2  # 元の CuPy データを削除
    cp.get_default_memory_pool().free_all_blocks()  # GPU メモリを解放

    # 分割して各部分を NumPy に保存
    for i in range(num_parts):
        start = i * part_size
        end = Field_2_np["x"].size if i == num_parts - 1 else (i + 1) * part_size

        # 分割した部分を Field_2_parts に追加
        part = WaveField3D(end - start, wavelength, Field_2.wave_num_H, Field_2.wave_num_V)
        part.x = Field_2_np["x"][start:end]
        part.y = Field_2_np["y"][start:end]
        part.z = Field_2_np["z"][start:end]
        part.ds = Field_2_np["ds"][start:end]
        Field_2_parts.append(part)

    # Field_2_np を削除して CPU メモリも解放
    del Field_2_np

    # 各分割パートで計算
    results = []  # 計算済みデータを保存するリスト
    for i, part in enumerate(Field_2_parts):
        print(f"Calculating for part {i + 1}/{num_parts}")

        # 必要なデータを CuPy にロード
        part.x = cp.array(part.x)
        part.y = cp.array(part.y)
        part.z = cp.array(part.z)
        part.ds = cp.array(part.ds)

        # 計算を実行
        part.forward_propagation(Field_1)

        # 計算結果を NumPy に戻して保存
        results.append(part.u.get())  # CuPy -> NumPy
        del part  # メモリ解放
        cp.get_default_memory_pool().free_all_blocks()

    # 計算結果を結合
    Field_2_result = np.concatenate(results)

    # 必要であれば、結果を NumPy の .npz ファイルに保存
    np.savez_compressed("Field_2_result.npz", data=Field_2_result)

    print("All calculations completed and results saved.")

# folder_path = r'\\HPC-PC3\Users\OP_User\Desktop\akb\output_20241129_4096_4096'  # 読み込みたいフォルダ名を指定
folder_path = r'output_20250305_sNA_2049cstm_woAlgn'
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
            ray_num_v2 = int(line.split(':')[1].strip())  # 値を整数として取得
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

shutil.copy(os.path.join(folder_path, 'calculation_conditions.txt'), os.path.join(directory_name, 'calculation_conditions.txt'))

print(source.reshape(3,1))
wavelength  =13.5e-9 * 1.e-1
print("WaveLength nm",wavelength*1e9)
LightSource = WaveField3D(1, wavelength,1,1)
LightSource.u[0] = 1.;

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
    Field_1.forward_propagation(LightSource)
    np.savez_compressed(os.path.join(directory_name, "complex_data_M1.npz"), data=Field_1.u)
    del data, LightSource # メモリ解放
    gc.collect()

# sys.exit()

# M2計算
hmirr_hyp = load_file(folder_path, file_names[2])
np.save(os.path.join(directory_name, 'points_M2.npy'), hmirr_hyp)
Field_2 = WaveField3D(hmirr_hyp.shape[1], wavelength,ray_num_H2,ray_num_V2)
Field_2.setdata(hmirr_hyp)
Field_1.set_ds(vmirr_hyp[3, :])
option_split = False
num_parts = 1500  # 分割数（調整可能）
data = load_npz_data("complex_data_M2.npz")
if data is not None:
    print("complex_data_M2.npz を正常に読み込みました。")
    Field_2.u = data
else:
    print("M1 => M2 計算を行います")
    if option_split:
        # 分割設定
        part_size = Field_2.u.size // num_parts  # 各パートのサイズ
        Field_2_parts = []

        # Field_2 を NumPy に変換して CPU に保存
        Field_2_np = {
            "x": Field_2.x.get(),  # CuPy -> NumPy
            "y": Field_2.y.get(),
            "z": Field_2.z.get(),
            "ds": Field_2.ds.get(),
            "u": np.zeros_like(Field_2.x.get(), dtype=np.complex128),  # 結果を格納
        }
        del Field_2  # 元の CuPy データを削除
        cp.get_default_memory_pool().free_all_blocks()  # GPU メモリを解放

        # 分割して各部分を NumPy に保存
        for i in range(num_parts):
            start = i * part_size
            end = Field_2_np["x"].size if i == num_parts - 1 else (i + 1) * part_size

            # 分割した部分を Field_2_parts に追加
            part = WaveField3D(end - start, wavelength,pix_z,pix_y)
            part.x = Field_2_np["x"][start:end]
            part.y = Field_2_np["y"][start:end]
            part.z = Field_2_np["z"][start:end]
            part.ds = Field_2_np["ds"][start:end]
            Field_2_parts.append(part)

        # Field_2_np を削除して CPU メモリも解放
        del Field_2_np

        # 各分割パートで計算
        results = []  # 計算済みデータを保存するリスト
        for i, part in enumerate(Field_2_parts):
            print(f"Calculating for part {i + 1}/{num_parts}")

            # 必要なデータを CuPy にロード
            part.x = cp.array(part.x)
            part.y = cp.array(part.y)
            part.z = cp.array(part.z)
            part.ds = cp.array(part.ds)

            # 計算を実行
            part.forward_propagation(Field_1)
            print('Done part')
            # 計算結果を NumPy に戻して保存
            np.savez_compressed(os.path.join(directory_name, f"complex_data_M2_{i + 1}_{num_parts}.npz"), data=part.u)
            # results.append(part.u.get())  # CuPy -> NumPy
            del part  # メモリ解放
            gc.collect()
            print('Delete part')
            cp.get_default_memory_pool().free_all_blocks()


        # # 計算結果を結合
        # Field_2_result = np.concatenate(results)
        # #
        # # 必要であれば、結果を NumPy の .npz ファイルに保存
        # np.savez_compressed("Field_2_result.npz", data=Field_2_result)

        print("All calculations completed and results saved.")
        sys.exit()

    else:
        Field_2.forward_propagation(Field_1)
    np.savez_compressed(os.path.join(directory_name, "complex_data_M2.npz"), data=Field_2.u)
    del Field_1, data  # 不要な変数を削除
    cp.get_default_memory_pool().free_all_blocks()

# 画像グリッド計算
Image_grid_org = load_file(folder_path, file_names[3])

#リサイズ
if True:
    mean_val = [np.mean(Image_grid_org[0,:]),np.mean(Image_grid_org[1,:]),np.mean(Image_grid_org[2,:])]
    Image_grid_org[0,:] = (Image_grid_org[0,:] - mean_val[0])*2. + mean_val[0]
    Image_grid_org[1,:] = (Image_grid_org[1,:] - mean_val[1])*2. + mean_val[1]
    Image_grid_org[2,:] = (Image_grid_org[2,:] - mean_val[2])*2. + mean_val[2]

np.save(os.path.join(directory_name, 'points_gridImage.npy'), Image_grid_org)
Image_grid = WaveField3D(Image_grid_org.shape[1], wavelength, pix_y,pix_z)
Image_grid.setdata(Image_grid_org)
Field_2.set_ds(hmirr_hyp[3, :])
print("M2 => Image 計算を行います")
if option_split:
    Image_grid_parts = split_wave_field(Image_grid, num_parts)

    # 分割した各部分で forward_propagation を実行
    results = []
    for part in Image_grid_parts:
        print(f"Calculating for part {Image_grid_parts.index(part) + 1}/{num_parts}")
        part.forward_propagation(Field_2)  # Field_1 を使用して計算
        results.append(part.u)  # 各部分の結果を保存

    # 各部分の結果を結合して Field_2.u に保存
    Image_grid.u = cp.concatenate(results)
else:
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

print("M2 => Image2 計算を行います")
if option_split:
    Image_grid_parts2 = split_wave_field(Image_grid2, num_parts)

    # 分割した各部分で forward_propagation を実行
    results = []
    for part in Image_grid_parts2:
        print(f"Calculating for part {Image_grid_parts2.index(part) + 1}/{num_parts}")
        part.forward_propagation(Field_2)  # Field_1 を使用して計算
        results.append(part.u)  # 各部分の結果を保存

    # 各部分の結果を結合して Field_2.u に保存
    Image_grid2.u = cp.concatenate(results)
else:
    Image_grid2.forward_propagation(Field_2)
np.savez_compressed(os.path.join(directory_name, "complex_data_Image2.npz"), data=Image_grid2.u)


del Field_2, Image_grid2  # メモリ解放
cp.get_default_memory_pool().free_all_blocks()

print("すべての計算が完了しました。")
