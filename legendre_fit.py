import os
import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt
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
                content = np.loadtxt(file_path)
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
# 収差をルジャンドル多項式で各次数の組み合わせごとに計算
def aberration_legendre_component(x, y, nx, ny):
    """
    各次数の組み合わせごとのルジャンドル多項式で表現した収差
    :param x: x座標の配列（-1から1の範囲）
    :param y: y座標の配列（-1から1の範囲）
    :param nx: x方向のルジャンドル多項式の次数
    :param ny: y方向のルジャンドル多項式の次数
    :param coeff: 係数
    :return: 収差分布の2D配列（特定次数のみ）
    """
    Px = legendre(nx)(x)  # x方向のルジャンドル多項式
    Py = legendre(ny)(y)  # y方向のルジャンドル多項式
    return np.outer(Py, Px)  # y方向のPyとx方向のPxの積

def match_legendre(data, nx, ny):
    """
    収差データをルジャンドル多項式の次数に基づいてマッチングする関数。
    :param data: 収差データの2D配列
    :param nx: x方向のルジャンドル多項式の次数
    :param ny: y方向のルジャンドル多項式の次数
    :return: マッチングされた収差データ
    """
    x = np.linspace(-1, 1, data.shape[0])
    y = np.linspace(-1, 1, data.shape[1])
    Z = aberration_legendre_component(x, y, nx, ny)
    Z /= np.sqrt(np.nansum(Z * Z))
    inner_product = np.nansum(Z * data)
    fit_data = inner_product * Z
    return fit_data, inner_product

def match_legendre_multi(data, order):
    """
    収差データを複数のルジャンドル多項式の次数に基づいてマッチングする関数。
    :param data: 収差データの2D配列
    """
    n_length = order * (order + 1) // 2  # 各次数の組み合わせ数
    fit_datas = np.zeros((n_length, data.shape[0], data.shape[1]))
    inner_products = np.zeros(n_length)
    orders = []
    count = 0
    for i in range(order):
        for j in range(order):
            if j <= i:
                nx, ny = j, i - j
                fit_data, inner_product = match_legendre(data, nx, ny)
                orders.append((int(ny), int(nx)))
                fit_datas[count] = fit_data
                inner_products[count] = inner_product
                count += 1
    return fit_datas, inner_products, orders
def output_legendre_data(inner_product, order,size=129):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    Z = aberration_legendre_component(x, y, order[1], order[0])
    Z /= np.sqrt(np.nansum(Z * Z))
    fit_data = inner_product * Z
    return fit_data

# メインの処理
if __name__ == "__main__":
    folder_path = ''  # 読み込みたいフォルダ名を指定
    file_names = ['ResPhase_matrix.txt', '']  # 読み込みたいファイル名をリストで指定
    data = load_file(folder_path, file_names[0])
    print(type(data))
    # x, yの座標系を生成
    x = np.linspace(-1, 1, data.shape[0])
    y = np.linspace(-1, 1, data.shape[1])

    # プロットの行と列の数を設定
    num_rows = 7
    num_cols = 7  # 最大列数を6に設定

    Z_datas = np.zeros((round((num_rows**2+num_rows)/2), data.shape[0],data.shape[1]))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16))
    set = np.zeros((round((num_rows**2+num_rows)/2), 2))
    count = 0
    # 各次数ごとに収差分布を計算してプロット
    for i in range(num_rows):
        for j in range(num_cols):
            if j <= i:
                nx, ny = j, i - j
                label = f"Order ({nx}, {ny})"
                # 特定の条件に基づく特別な処理
                if i % 2 == 0 and False:
                    if nx == ny:  # (0, i)+(i, 0)
                        Z = aberration_legendre_component(x, y, nx, ny)
                    else:  # (0, i)-(i, 0)
                        Z1 = aberration_legendre_component(x, y, nx, ny)
                        Z2 = aberration_legendre_component(x, y, ny, nx)
                        if nx > ny:
                            Z = Z1 + Z2
                            label = f"Order ({nx}, {ny}) + ({ny}, {nx})"
                        else:
                            Z = Z1 - Z2
                            label = f"Order ({nx}, {ny}) - ({ny}, {nx})"
                else:
                    Z = aberration_legendre_component(x, y, nx, ny)
                # Z = aberration_legendre_component(x, y, nx, ny)
                print(np.sqrt(np.mean(Z*Z)))
                Z_datas[count] = Z/np.sqrt(np.sum(Z*Z))
                ax = axes[i, j]  # 2Dインデックスで指定
                im = ax.imshow(Z, cmap="jet")
                # fig.colorbar(im, ax=ax, label="Aberration")
                ax.set_title(label)
                # ax.set_xlabel("x")
                # ax.set_ylabel("y")
                ax.set_xticks([])
                ax.set_yticks([])
                set[count,0] = i
                set[count,1] = j
                count = count + 1
            else:
                axes[i, j].axis('off')  # 空白の場合は非表示

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.imshow(data, cmap="jet")
    plt.show()

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16))

    print(set)
    # 各次数ごとに収差分布を計算してプロット
    for i in range(Z_datas.shape[0]):
        # 内積を取る
        inner_product = np.mean(Z_datas[i]* data)
        # abs_product = np.dot(Z_datas[i], Z_datas[i])
        fit = inner_product*Z_datas[i]
        # print(fit.shape)
        ax = axes[round(set[i,0]),round(set[i,1])]  # 2Dインデックスで指定
        im = ax.imshow(fit, cmap="jet")
        fig.colorbar(im, ax=ax, label="Aberration")
        ax.set_title(label)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16))
    # 各次数ごとに収差分布を計算してプロット
    for i in range(num_rows):
        for j in range(num_cols):
            if j <= i:
                nx, ny = j, i - j
                label = f"Order ({nx}, {ny})"
                # 特定の条件に基づく特別な処理
                if i % 2 == 0:
                    if nx == ny:  # (0, i)+(i, 0)
                        Z = aberration_legendre_component(x, y, nx, ny)
                    else:  # (0, i)-(i, 0)
                        Z1 = aberration_legendre_component(x, y, nx, ny)
                        Z2 = aberration_legendre_component(x, y, ny, nx)
                        if nx > ny:
                            Z = Z1 + Z2
                            label = f"Order ({nx}, {ny}) + ({ny}, {nx})"
                        else:
                            Z = Z1 - Z2
                            label = f"Order ({nx}, {ny}) - ({ny}, {nx})"
                else:
                    Z = aberration_legendre_component(x, y, nx, ny)
                # Z = aberration_legendre_component(x, y, nx, ny)

                ax = axes[i, j]  # 2Dインデックスで指定
                ax.plot(x,Z[0,:],c='r')
                ax.plot(x,Z[round(len(Z)/2),:],c='y')
                ax.plot(x,Z[-1,:],c='g')

                ax.set_title(label)
                # ax.set_xlabel("x")
                # ax.set_ylabel("y")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                axes[i, j].axis('off')  # 空白の場合は非表示

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16))
    # 各次数ごとに収差分布を計算してプロット
    for i in range(num_rows):
        for j in range(num_cols):
            if j <= i:
                nx, ny = j, i - j
                label = f"Order ({nx}, {ny})"
                # 特定の条件に基づく特別な処理
                if i % 2 == 0:
                    if nx == ny:  # (0, i)+(i, 0)
                        Z = aberration_legendre_component(x, y, nx, ny)
                    else:  # (0, i)-(i, 0)
                        Z1 = aberration_legendre_component(x, y, nx, ny)
                        Z2 = aberration_legendre_component(x, y, ny, nx)
                        if nx > ny:
                            Z = Z1 + Z2
                            label = f"Order ({nx}, {ny}) + ({ny}, {nx})"
                        else:
                            Z = Z1 - Z2
                            label = f"Order ({nx}, {ny}) - ({ny}, {nx})"
                else:
                    Z = aberration_legendre_component(x, y, nx, ny)
                # Z = aberration_legendre_component(x, y, nx, ny)

                ax = axes[i, j]  # 2Dインデックスで指定
                ax.plot(x,Z[:,0],c='r')
                ax.plot(x,Z[:,round(len(Z)/2)],c='y')
                ax.plot(x,Z[:,-1],c='g')

                ax.set_title(label)
                # ax.set_xlabel("x")
                # ax.set_ylabel("y")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                axes[i, j].axis('off')  # 空白の場合は非表示

    plt.tight_layout()
    plt.show()
