import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import factorial

st.set_page_config(page_title="Iterated maximal operator", layout="wide")

# --- 1. 定数とキャッシュ ---
MAX_K = 15
ABS_MAX_RANGE = 100
NUM_POINTS = 2000

@st.cache_data
def precompute_data():
    x = np.linspace(-ABS_MAX_RANGE, ABS_MAX_RANGE, NUM_POINTS)
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
st.title(r"The behaviour of the iterated maximal function $M^k [ \chi_{[0,1]} ](x)$.")

# --- 2. UI ---
with st.sidebar:
    st.header("Controls")
    # 文字列の不一致を防ぐため変数に格納
    COMP_MODE = "Comparison Mode"
    ALL_MODE = "All Layers Mode"
    mode = st.radio("Mode", [COMP_MODE, ALL_MODE])
    
    if mode == COMP_MODE:
        st.latex(r"M^{k_1} \left[ \chi_{[0,1]}\right] \text{ vs } M^{k_2} \left[ \chi_{[0,1]}\right]")
        k1 = st.slider("$k_1$", 1, MAX_K, 1)
        k2 = st.slider("$k_2$", 1, MAX_K, 5)
    else:
        st.latex(r"M^1, \dots, M^k")
        k_max = st.slider("Max $k$", 1, MAX_K, 10)
    
    view_range = st.slider("Plot Range", 5, ABS_MAX_RANGE, 50)
    show_unit = st.checkbox("Show line $y=1$", value=True)

# --- 3. データ抽出 ---
mask = (X_GRID >= -view_range) & (X_GRID <= view_range + 1)
x_p = X_GRID[mask]
d_p = FULL_DATA[mask, :]

# --- 4. Plotly 描画 ---
fig = go.Figure()

if show_unit:
    fig.add_shape(type="line", x0=-view_range, y0=1, x1=view_range+1, y1=1,
                  line=dict(color="Gray", width=1, dash="dot"))

fig.add_trace(go.Scatter(x=x_p, y=np.where((x_p>0)&(x_p<1), 1.0, 0.0),
                         name=r"$\chi_{[0,1]}$", line=dict(color="black", width=3)))

# モード判定の修正
if mode == COMP_MODE:
    fig.add_trace(go.Scatter(x=x_p, y=d_p[:, k1-1], name=f"M^{k1}", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x_p, y=d_p[:, k2-1], name=f"M^{k2}$", line=dict(width=2)))
else:
    for i in range(k_max):
        show_leg = (i==0 or i==k_max-1 or (i+1)%(max(1, k_max//5))==0)
        fig.add_trace(go.Scatter(x=x_p, y=d_p[:, i], name=f"$M^{i+1}$",
                                 line=dict(width=1), showlegend=show_leg,
                                 opacity=0.6 if not show_leg else 1.0))

fig.update_layout(
    xaxis=dict(title="x", showgrid=True, gridcolor='LightGray', zerolinecolor='black'),
    yaxis=dict(title="y", range=[0, 1.1], showgrid=True, gridcolor='LightGray', zerolinecolor='black'),
    template="plotly_white", height=700, margin=dict(l=20, r=20, t=80, b=20),
    hovermode="x unified", legend=dict(font=dict(size=14))
)

# --- 5. MathJax 強制適用スクリプト ---
#st.markdown("""
#<script type="text/javascript" id="MathJax-script" async
#  src="https://cdn.jsdelivr.net">
#</script>
#""", unsafe_allow_html=True)

# --- 5. 重要：MathJaxを凡例に強制適用するスクリプト ---
# Plotlyの内部要素(svg)に対してもMathJaxを走らせる設定です
st.markdown("""
<script>
  window.MathJax = {
    tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] },
    svg: { fontCache: 'global' }
  };
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net">
</script>
""", unsafe_allow_html=True)

# 描画実行
st.plotly_chart(fig, use_container_width=True, config={'include_mathjax': 'cdn'})

# --- 6. 解説セクション ---
st.write("---")
st.write(r"The definition of the Hardy-Littlewood maximal operator $M$." )
st.latex(r'''Mf(x) :=\sup_{I} \frac{1}{|I|} \int_{I} | f(y) | dy\cdot \chi_{I}(x).''')
st.write("The definition of the iterated Hardy-Littlewood maximal operator $M^{k}$." )
st.latex(r'''M^{k}f(x):= \underbrace{(M \circ \dots \circ M)}_{k} f(x)''')
st.write(r"For $f = \chi_{[0,1]}$:" )
st.latex(r"M^k \left[ \chi_{[0,1]}\right](x) = \begin{cases} 1 & 0\leq x\leq 1 \\ \displaystyle \frac{1}{x} \sum_{j=0}^{k-1} \frac{(\log (x))^j}{j!} & x >1 \\ \displaystyle \frac{1}{1-x} \sum_{j=0}^{k-1} \frac{(\log (1-x))^j}{j!} & x < 0 \end{cases}")

