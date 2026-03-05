import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import io

# --- 1. ページ設定とスタイル ---
st.set_page_config(page_title="Iterated maximal operator", layout="wide")

# Matplotlib の LaTeX 設定 (Streamlit Cloud の標準フォントを使用)
plt.rcParams.update({
    "text.usetex": False,  # サーバー環境でのエラー回避のため False にし、Mathtext を使用
    "font.size": 12,
    "grid.alpha": 0.3
})

# --- 2. 定数と計算キャッシュ ---
MAX_K = 15
ABS_MAX_RANGE = 100
NUM_POINTS = 3000

@st.cache_data
def precompute_data():
    """不連続点付近に点を集中させた高密度グリッドで計算"""
    # 1. 基本となる広域グリッド
    x_base = np.linspace(-ABS_MAX_RANGE, ABS_MAX_RANGE, NUM_POINTS)
    
    # 2. 不連続点 (0 と 1) 付近に指数的に密集した点を追加
    # 1e-9 から 0.1 までの距離に 200 点ずつ配置
    fine_grid = np.logspace(-9, -1, 200) 
    special_points = np.concatenate([
        [0, 1], # ジャストの点
        fine_grid, -fine_grid,           # 0 の前後
        1 + fine_grid, 1 - fine_grid     # 1 の前後
    ])
    
    # 3. すべてを結合してソート
    x = np.unique(np.sort(np.concatenate([x_base, special_points])))
    
    # --- 以下、計算ロジックは同じ ---
    v_pos = np.maximum(np.abs(x), 1e-15)
    v_neg = np.maximum(np.abs(1.0 - x), 1e-15)
    j = np.arange(MAX_K)
    inv_fact = 1.0 / factorial(j)
    
    log_p = np.log(v_pos)[:, np.newaxis]
    m_pos = (1.0 / v_pos)[:, np.newaxis] * np.cumsum((log_p ** j) * inv_fact, axis=1)
    
    log_n = np.log(v_neg)[:, np.newaxis]
    m_neg = (1.0 / v_neg)[:, np.newaxis] * np.cumsum((log_n ** j) * inv_fact, axis=1)
    
    res = np.ones((len(x), MAX_K))
    res[x >= 1, :] = m_pos[x >= 1, :]
    res[x <= 0, :] = m_neg[x <= 0, :]
    return x, res

X_GRID, FULL_DATA = precompute_data()

# --- 3. UI ---
st.title(r"Iterated Maximal Function $M^k [ \chi_{[0,1]} ](x)$")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Comparison Mode of $M^{k_1}\\left[\\chi_{[0,1]} \\right](x)$ and $M^{k_2}\\left[\\chi_{[0,1]} \\right](x)$", "All Layers Mode"])
    
    if mode == "Comparison Mode of $M^{k_1}\\left[\\chi_{[0,1]} \\right](x)$ and $M^{k_2}\\left[\\chi_{[0,1]} \\right](x)$":
        k1 = st.slider(r"$k_1$", 1, MAX_K, 1)
        k2 = st.slider(r"$k_2$", 1, MAX_K, 5)
    else:
        k_max = st.slider(r"Max $k$", 1, MAX_K, 10)
    
    view_range = st.slider("Plot Range", 5, ABS_MAX_RANGE, 50)
    show_unit = st.checkbox("Show line $y=1$", value=True)

# --- 4. 描画 (Matplotlib) ---
fig, ax = plt.subplots(figsize=(10, 6))

mask = (X_GRID >= -view_range) & (X_GRID <= view_range)
x_p, d_p = X_GRID[mask], FULL_DATA[mask, :]

if show_unit:
    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.5, label="$y=1$")

# 基底関数
ax.plot(x_p, np.where((x_p>0)&(x_p<1), 1.0, 0.0), color="black", lw=2, label=r"$\chi_{[0,1]}$")

if mode == "Comparison Mode of $M^{k_1}\\left[\\chi_{[0,1]} \\right](x)$ and $M^{k_2}\\left[\\chi_{[0,1]} \\right](x)$":
    ax.plot(x_p, d_p[:, k1-1], lw=2.5, color="dodgerblue", label=f"$M^{{{k1}}} \\left[ \\chi_{{[0,1]}} \\right](x)$")
    ax.plot(x_p, d_p[:, k2-1], lw=2.5, color="orangered", label=f"$M^{{{k2}}} \\left[ \\chi_{{[0,1]}} \\right](x)$")
else:
    colors = plt.cm.plasma(np.linspace(0, 0.85, k_max)) 
    for i in range(k_max):
        show_leg = (i==0 or i==k_max-1 or (i+1)%(max(1, k_max//5))==0)
        ax.plot(x_p, d_p[:, i], color=colors[i], lw=1, alpha=0.7, 
                label=f"$M^{{{i+1}}} \chi$" if show_leg else "")

ax.set_xlim(-view_range, view_range)
ax.set_ylim(0, 1.1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right", ncol=2, fontsize=10)

# グラフを表示
st.pyplot(fig)

# --- 5. PDF 保存 (Matplotlib の標準機能) ---
buf = io.BytesIO()
fig.savefig(buf, format="pdf", bbox_inches="tight")

st.download_button(
    label="Export graph to PDF",
    data=buf.getvalue(),
    file_name="iterated_maximal_operator.pdf",
    mime="application/pdf"
)

# --- 6. 解説セクション ---
st.write("---")
st.write(r"The definition of the Hardy-Littlewood maximal operator $M$." )
st.latex(r'''Mf(x) :=\sup_{\substack{ I\subset \mathbb{R},\\ I: \text{interval} }} \frac{1}{|I|} \int_{I} | f(y) | dy\cdot \chi_{I}(x).''')
st.write("The definition of the iterated Hardy-Littlewood maximal operator $M^{k}$." )
st.latex(r'''M^{k}f(x):= \underbrace{(M \circ \dots \circ M)}_{k} f(x)''')
st.write(r"For $f = \chi_{[0,1]}$:" )
st.latex(r"M^k \left[ \chi_{[0,1]}\right](x) = \begin{cases} 1 & 0\leq x\leq 1 \\ \displaystyle \frac{1}{x} \sum_{j=0}^{k-1} \frac{(\log (x))^j}{j!} & x >1 \\ \displaystyle \frac{1}{1-x} \sum_{j=0}^{k-1} \frac{(\log (1-x))^j}{j!} & x < 0 \end{cases}")



