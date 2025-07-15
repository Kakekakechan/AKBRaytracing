import numpy as np
from numpy import cos,sin,tan,arccos,arctan,sqrt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import KB_design_NAbased as KB
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


def plot_ellipses_gui(Ell1, Ell2, text_widget, canvas_frames):
    # テキストリセット
    text_widget.delete(1.0, tk.END)

    # === 1枚目のFigure ===
    fig0 = plt.Figure(figsize=(4, 3))
    ax0 = fig0.add_subplot(111)
    ax0.plot([Ell1.x_1, Ell1.x_1 + Ell1.x_2], [Ell1.y_1, Ell1.y_2], 'r--')
    ax0.plot(2 * Ell1.f, 0, 'ro')
    ax0.plot([Ell2.x_1, Ell2.x_1 + Ell2.x_2], [Ell2.y_1, Ell2.y_2], 'b--')
    ax0.plot(2 * Ell2.f, 0, 'bo')
    ax0.set_title("Ellipses")

    # === 2枚目のFigure ===
    fig1 = plt.Figure(figsize=(4, 3))
    ax1_0 = fig1.add_subplot(121)
    ax1_1 = fig1.add_subplot(122)

    ax1_0.plot(
        [0, Ell1.x_2],
        [(Ell1.theta_i1 + Ell1.theta_o1) / 2, (Ell1.theta_i2 + Ell1.theta_o2) / 2],
        'r--'
    )
    ax1_1.plot(
        [0, Ell2.x_2],
        [(Ell2.theta_i1 + Ell2.theta_o1) / 2, (Ell2.theta_i2 + Ell2.theta_o2) / 2],
        'b--'
    )
    ax1_0.set_xlabel("distance (m)")
    ax1_0.set_ylabel("incident angle (rad)")
    ax1_0.set_title("Ell1 incident angle")
    ax1_0.set_ylim((Ell1.theta_i1 + Ell1.theta_o1) / 2, (Ell1.theta_i2 + Ell1.theta_o2) / 2)
    ax1_1.set_xlabel("distance (m)")
    ax1_1.set_title("Ell2 incident angle")
    ax1_1.set_ylim((Ell2.theta_i1 + Ell2.theta_o1) / 2, (Ell2.theta_i2 + Ell2.theta_o2) / 2)

    # === 結果テキスト ===
    lines = [
        f"Ell1 diverge angle: {Ell1.theta_i1 - Ell1.theta_i2}",
        f"Ell1 mirror length: {Ell1.mirr_length}",
        f"Ell1 mirror angle: {[(Ell1.theta_i1 + Ell1.theta_o1) / 2, (Ell1.theta_i2 + Ell1.theta_o2) / 2]}",
        f"Ell1 demagnification: {[Ell1.m1, Ell1.m2, np.mean([Ell1.m1, Ell1.m2])]}",
        f"Ell2 diverge angle: {Ell2.theta_i1 - Ell2.theta_i2}",
        f"Ell2 mirror length: {Ell2.mirr_length}",
        f"Ell2 mirror angle: {[(Ell2.theta_i1 + Ell2.theta_o1) / 2, (Ell2.theta_i2 + Ell2.theta_o2) / 2]}",
        f"Ell2 demagnification: {[Ell2.m1, Ell2.m2, np.mean([Ell2.m1, Ell2.m2])]}",
        "===========================",
        f"Ell1 aperture: {Ell1.mirr_length * Ell1.theta_centre}",
        f"Ell2 aperture: {Ell2.mirr_length * Ell2.theta_centre}",
        f"Area aperture: {Ell1.mirr_length * Ell1.theta_centre * Ell2.mirr_length * Ell2.theta_centre}",
        f"Focus distance: {Ell1.f - Ell2.f}"
    ]
    text_widget.insert(tk.END, "\n".join(lines))

    fig1.tight_layout()

    # === Canvasに貼り付け ===
    figs = [fig0, fig1]
    for frame, fig in zip(canvas_frames, figs):
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

def run():
    try:
        l_i1 = np.float64(entry_l_i1.get())
        na_o_sin_v = np.float64(entry_na_o_sin_v.get())
        na_o_sin_h = np.float64(entry_na_o_sin_h.get())
        target_gap = np.float64(entry_target_gap.get())
        ast = np.float64(entry_ast.get())
        l_o1 = np.float64(entry_l_o1.get())
        theta_g1 = np.float64(entry_theta_g1.get())
        target_l_o2 = np.float64(entry_target_l_o2.get())

        Ell1 = KB.ELL_V_design(l_i1, l_o1, theta_g1, na_o_sin_v)
        Ell1, Ell2 = KB.ELL_H_design(Ell1, target_l_o2, target_gap, ast, na_o_sin_h)

        plot_ellipses_gui(Ell1, Ell2, text_output, [canvas_frame1, canvas_frame2])

    except Exception as e:
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, f"Error: {e}")

def run_other():
    try:
        # 例: 別の処理を呼ぶ
        result = KB.some_other_function()  # <- ここを実際の関数名に書き換える
        text_output.insert(tk.END, "\n--- 別の処理の結果 ---\n")
        text_output.insert(tk.END, f"{result}\n")
        text_output.see(tk.END)  # スクロールする
    except Exception as e:
        text_output.insert(tk.END, f"\nError in other function: {e}\n")
        text_output.see(tk.END)


root = tk.Tk()
root.title("KB設計ツール")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

labels = [
    "l_i1", "na_o_sin_v", "na_o_sin_h", "target_gap",
    "ast", "l_o1", "theta_g1", "target_l_o2"
]
defaults = [
    "48.6", "0.002", "0.002", "0.1",
    "0.", "0.33", "0.006", "0.04"
]
entries = []

for i, (label, default) in enumerate(zip(labels, defaults)):
    ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
    entry = ttk.Entry(frame)
    entry.insert(0, default)
    entry.grid(row=i, column=1, pady=2)
    entries.append(entry)

(
    entry_l_i1, entry_na_o_sin_v, entry_na_o_sin_h, entry_target_gap,
    entry_ast, entry_l_o1, entry_theta_g1, entry_target_l_o2
) = entries

ttk.Button(frame, text="計算して更新", command=run).grid(row=len(labels), column=0, columnspan=2, pady=5)

# 別の処理ボタン
ttk.Button(frame, text="別の処理を実行", command=run_other).grid(row=len(labels)+1, column=0, columnspan=2, pady=5)

# === グラフとテキスト出力領域 ===
canvas_frame1 = ttk.LabelFrame(root, text="グラフ1/2")
canvas_frame1.grid(row=0, column=1, padx=10, pady=5)

canvas_frame2 = ttk.LabelFrame(root, text="グラフ2/2")
canvas_frame2.grid(row=1, column=1, padx=10, pady=5)

text_output = ScrolledText(root, width=60, height=20)
text_output.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()