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
import threading
### GPU ###
print(f"free GPU: {cp.cuda.runtime.getDeviceCount()}")  # 使用可能なGPU数を表示
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

    def forward_propagation(self, u_back):
        k = 2.0 * cp.pi / self.lambda_  # 波数の計算
        start_time = time.time()  # 開始時間を記録
        # GPUで計算を呼び出す（バッチ処理対応）

        self.u = forward_propagation_cupy_batch_multi_gpu(
            self.x, self.y, self.z,
            u_back.x, u_back.y, u_back.z,
            u_back.u,  # 複素数配列を直接渡す
            k, u_back.ds  # 事前計算したdsを渡す
        )
        end_time = time.time()  # 終了時間を記録

        # 実行時間を表示
        elapsed_time = end_time - start_time
        print(f"計算時間: {elapsed_time:.6f} 秒")

def process_on_gpu(gpu_id, split_x, split_y, split_z, u_back_x, u_back_y, u_back_z, u_back_u, k, num_back, results, streams, cleanup_streams):
    with cp.cuda.Device(gpu_id):  # GPUデバイスを切り替え
        streams[gpu_id] = cp.cuda.Stream()  # 現在のデバイスでストリームを作成
        cleanup_streams[gpu_id] = cp.cuda.Stream()  # メモリ解放用ストリームも同様に作成

        with cp.cuda.Device(gpu_id), streams[gpu_id]:
            x_batch = split_x[gpu_id]
            y_batch = split_y[gpu_id]
            z_batch = split_z[gpu_id]

            # GPUメモリの空き容量を取得してバッチサイズを決定
            mem_info = cp.cuda.Device().mem_info
            free_mem = mem_info[0]
            overhead = 4.0  # メモリの余裕率
            element_size = 16  # 複素数1要素あたりのメモリ使用量（バイト）
            max_batch_size = int((free_mem / overhead) / element_size / num_back)
            print(f"GPU {gpu_id} free memory: {free_mem}")
            print(f"GPU {gpu_id} max batch size: {max_batch_size}")

            # 出力配列を初期化（複素数型）
            u_partial = cp.zeros(len(x_batch), dtype=cp.complex128)

            # バッチごとに計算
            for i in range(0, len(x_batch), max_batch_size):
                batch_end = min(i + max_batch_size, len(x_batch))
                x_sub_batch = x_batch[i:batch_end]
                y_sub_batch = y_batch[i:batch_end]
                z_sub_batch = z_batch[i:batch_end]

                # メインストリームで計算を実行
                with streams[gpu_id]:
                    dist = cp.sqrt(
                        (x_sub_batch[:, None] - u_back_x[None, :]) ** 2 +
                        (y_sub_batch[:, None] - u_back_y[None, :]) ** 2 +
                        (z_sub_batch[:, None] - u_back_z[None, :]) ** 2
                    )
                    amplitude = 1. / dist
                    phase = -k * dist
                    factor = amplitude * cp.exp(1j * phase)
                    interaction = cp.dot(u_back_u, factor.T)
                    u_partial[i:batch_end] = interaction

                # 進捗状況を表示
                if i % 23 == 0:
                    print(f"GPU {gpu_id} - Batch {i // max_batch_size + 1}/{(len(x_batch) + max_batch_size - 1) // max_batch_size}")
                    mem_info = cp.cuda.Device().mem_info
                    free_mem = mem_info[0]
                    total_mem = mem_info[1]
                    used_mem = total_mem - free_mem
                    print(f"GPU {gpu_id} - Used memory: {used_mem / 1024**2:.2f} MB / {total_mem / 1024**2:.2f} MB")

                # メモリ解放を別ストリームで処理
                with cleanup_streams[gpu_id]:
                    del x_sub_batch, y_sub_batch, z_sub_batch, dist, amplitude, phase, factor, interaction
                    cp.get_default_memory_pool().free_all_blocks()

            # 部分結果を保存
            results[gpu_id] = u_partial

def forward_propagation_cupy_batch_multi_gpu(x, y, z, u_back_x, u_back_y, u_back_z, u_back_u, k, ds):
    if not isinstance(u_back_u, cp.ndarray):
        u_back_u = cp.asarray(u_back_u)
    u_back_u = u_back_u * ds
    del ds
    num = len(x)
    num_back = len(u_back_x)

    # GPUの数を取得
    num_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"Number of GPUs available: {num_gpus}")

    # データをGPU数で分割
    split_x = cp.array_split(x, num_gpus)
    split_y = cp.array_split(y, num_gpus)
    split_z = cp.array_split(z, num_gpus)

    # # 各GPUで計算結果を格納するリスト
    # results = []
    # streams = [cp.cuda.Stream() for _ in range(num_gpus)]
    # cleanup_streams = [cp.cuda.Stream() for _ in range(num_gpus)]  # メモリ解放用ストリーム

    # # 各GPUで並列計算
    # for gpu_id in range(num_gpus):
    #     with cp.cuda.Device(gpu_id):  # GPUデバイスを切り替え
    #         streams[gpu_id] = cp.cuda.Stream()  # 現在のデバイスでストリームを作成
    #         cleanup_streams[gpu_id] = cp.cuda.Stream()  # メモリ解放用ストリームも同様に作成

    #     with cp.cuda.Device(gpu_id), streams[gpu_id]:
    #         x_batch = split_x[gpu_id]
    #         y_batch = split_y[gpu_id]
    #         z_batch = split_z[gpu_id]

    #         # GPUメモリの空き容量を取得してバッチサイズを決定
    #         mem_info = cp.cuda.Device().mem_info
    #         free_mem = mem_info[0]
    #         overhead = 4.0  # メモリの余裕率
    #         element_size = 16  # 複素数1要素あたりのメモリ使用量（バイト）
    #         max_batch_size = int((free_mem / overhead) / element_size / num_back)
    #         print(f"GPU {gpu_id} free memory: {free_mem}")
    #         print(f"GPU {gpu_id} max batch size: {max_batch_size}")

    #         # 出力配列を初期化（複素数型）
    #         u_partial = cp.zeros(len(x_batch), dtype=cp.complex128)

    #         # バッチごとに計算
    #         for i in range(0, len(x_batch), max_batch_size):
    #             batch_end = min(i + max_batch_size, len(x_batch))
    #             x_sub_batch = x_batch[i:batch_end]
    #             y_sub_batch = y_batch[i:batch_end]
    #             z_sub_batch = z_batch[i:batch_end]

    #             # メインストリームで計算を実行
    #             with streams[gpu_id]:
    #                 dist = cp.sqrt(
    #                     (x_sub_batch[:, None] - u_back_x[None, :]) ** 2 +
    #                     (y_sub_batch[:, None] - u_back_y[None, :]) ** 2 +
    #                     (z_sub_batch[:, None] - u_back_z[None, :]) ** 2
    #                 )
    #                 amplitude = 1. / dist
    #                 phase = -k * dist
    #                 factor = amplitude * cp.exp(1j * phase)
    #                 interaction = cp.dot(u_back_u, factor.T)
    #                 u_partial[i:batch_end] = interaction
                # if i % 23 == 0:
                #     print(f"batch: {i}/{num}")
                #     # メモリ使用量を監視
                #     mem_info = cp.cuda.Device().mem_info
                #     free_mem = mem_info[0]
                #     total_mem = mem_info[1]
                #     used_mem = total_mem - free_mem
                #     print(f"GPU {gpu_id} - Used memory: {used_mem / 1024**2:.2f} MB / {total_mem / 1024**2:.2f} MB")
    #             # メモリ解放を別ストリームで処理
    #             with cleanup_streams[gpu_id]:
    #                 del x_sub_batch, y_sub_batch, z_sub_batch, dist, amplitude, phase, factor, interaction
    #                 cp.get_default_memory_pool().free_all_blocks()

    #         # 部分結果を保存
    #         results.append(u_partial)

    # # ストリームの同期
    # for stream, cleanup_stream in zip(streams, cleanup_streams):
    #     stream.synchronize()
    #     cleanup_stream.synchronize()

    # 各GPUで計算結果を格納するリスト
    results = [None] * num_gpus
    streams = [None] * num_gpus
    cleanup_streams = [None] * num_gpus

    # 各GPUの処理をスレッドで並列実行
    threads = []
    for gpu_id in range(num_gpus):
        thread = threading.Thread(
            target=process_on_gpu,
            args=(gpu_id, split_x, split_y, split_z, u_back_x, u_back_y, u_back_z, u_back_u, k, num_back, results, streams, cleanup_streams)
        )
        threads.append(thread)
        thread.start()

    # すべてのスレッドが終了するのを待機
    for thread in threads:
        thread.join()

    # 各GPUの結果を結合
    u = cp.concatenate(results)
    return u


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
        if i % 239 == 0:
            print(f"batch: {i}/{num}")
            mem_info = cp.cuda.Device().mem_info
            free_mem = mem_info[0]
            print(f"free memory: {free_mem}")

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

class DualOutput:
    def __init__(self, file_path):
        self.console = sys.stdout  # 元の標準出力
        self.file = open(file_path, "w", encoding="utf-8")  # ファイル出力用

    def write(self, message):
        self.console.write(message)  # コンソールに出力
        self.file.write(message)  # ファイルに出力
        self.file.flush()  # バッファをフラッシュしてディスクに書き込む

    def flush(self):
        self.console.flush()
        self.file.flush()

if __name__ == '__main__':
    # コンソール出力をテキストファイルに保存
    # フォルダ名として使用する現在時刻の文字列を取得
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"log_{timestamp}.txt"
    sys.stdout = DualOutput(log_file)  # 標準出力をDualOutputにリダイレクト

    try:
        # folder_path = r'\\HPC-PC3\Users\OP_User\Desktop\akb\output_20241129_4096_4096'  # 読み込みたいフォルダ名を指定
        folder_path = r'output_20250501_LNA4097'
        file_names = ['points_source.npy','points_M1.npy','points_M2.npy','points_gridImage.npy','points_gridDefocus.npy']  # 読み込みたいファイル名をリストで指定
        source = load_file(folder_path, file_names[0])
        vmirr_hyp = load_file(folder_path, file_names[1])


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
            Field_1.forward_propagation(LightSource)
            np.savez_compressed(os.path.join(directory_name, "complex_data_M1.npz"), data=Field_1.u)
            del data, LightSource # メモリ解放
            cp.get_default_memory_pool().free_all_blocks()

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
            cp.get_default_memory_pool().free_all_blocks()

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
                cp.get_default_memory_pool().free_all_blocks()

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
                cp.get_default_memory_pool().free_all_blocks()
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
        cp.get_default_memory_pool().free_all_blocks()
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
            cp.get_default_memory_pool().free_all_blocks()
        else:
            del Field_2, Image_grid2  # メモリ解放
            cp.get_default_memory_pool().free_all_blocks()

        print("すべての計算が完了しました。")
    except KeyboardInterrupt:
        print("計算が中断されました。現在の進捗を保存します...")
    except Exception as e:
        # その他のエラーをキャッチしてログに記録
        print(f"エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()  # エラーの詳細をログに出力
    finally:
        sys.stdout.file.close()  # ファイルを閉じる
        sys.stdout = sys.stdout.console  # 標準出力を元に戻す

    print(f"コンソール出力が {log_file} に保存されました。")
    sys.exit()  # プログラムを終了
