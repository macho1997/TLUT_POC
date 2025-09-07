# app_tlut_miniguia.py
# Streamlit — Mini-guía pedagógica T-LUT (paso a paso, con bits)
# - Modo MINI (16 muestras) para ver todo clarito y corto
# - Modo FULL (192 muestras) para ver el caso real
# - Tablas con binarios (Q4.4 y Q1.7), construcción d-LUT/e-LUT y reconstrucción
# - Curvas y error

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -------------------------
# Config fija de formatos
# -------------------------
X_FRAC = 4   # Q4.4
Y_FRAC = 7   # Q1.7
X_STEP = 1.0 / (1 << X_FRAC)
X_MIN_FULL, X_MAX_FULL = -6.0, 6.0

# Tema oscuro elegante
TEMPLATE = "plotly_dark"
BG = "#0e1117"
PLOT_BG = "#0e1117"
HL_FILL = "rgba(0,200,255,0.22)"
HL_LINE = "rgba(0,200,255,1.0)"
ALT_A = "rgba(255,255,255,0.06)"
ALT_B = "rgba(255,255,255,0.03)"
BOX_LINE = "rgba(255,255,255,0.35)"

# -------------------------
# Funciones base
# -------------------------
def f_sigmoid(x): return 1.0/(1.0+np.exp(-x))
def f_tanh01(x):  return 0.5*(np.tanh(x)+1.0)  # tanh mapeada a [0,1)

def to_q1_7(y):
    q = np.round(y * (1 << Y_FRAC)).astype(np.int32)
    return np.clip(q, 0, 127).astype(np.uint8)

def from_q1_7(q):
    return q.astype(np.float64) / (1 << Y_FRAC)

def bin_str(val, bits):
    return format(int(val) & ((1<<bits)-1), f"0{bits}b")

# -------------------------
# Datos (MINI y FULL)
# -------------------------
def build_domain(mini_mode: bool):
    if mini_mode:
        # 16 muestras centradas alrededor de 0: de -8/16 a +7/16
        xs = np.arange(-8, 8) * X_STEP  # [-0.5, 0.5) paso 1/16
    else:
        # Dominio real: [-6, 6) paso 1/16 → 192 muestras
        xs = np.arange(X_MIN_FULL, X_MAX_FULL, X_STEP)
    return xs

def eval_function(xs, func_name):
    if func_name == "sigmoid":
        y = f_sigmoid(xs)
    else:
        y = f_tanh01(xs)
    return y

def build_lut_flat(xs, func_name):
    y_real = eval_function(xs, func_name)
    y_q = to_q1_7(y_real)
    return y_real, y_q

def split_blocks(y_q, block_size):
    n_blocks = int(np.ceil(len(y_q)/block_size))
    pad = n_blocks*block_size - len(y_q)
    if pad:
        y_q = np.concatenate([y_q, np.full(pad, y_q[-1], dtype=np.uint8)])
    y_blocks = y_q.reshape(n_blocks, block_size)
    return y_blocks, n_blocks, pad

def tlut_compress(y_q, block_size, e_bits):
    y_blocks, n_blocks, pad = split_blocks(y_q, block_size)
    d_lut = np.min(y_blocks, axis=1).astype(np.uint8)
    delta_max = (1<<e_bits)-1
    deltas = (y_blocks - d_lut[:,None]).clip(0, delta_max).astype(np.uint16)
    return d_lut, deltas, n_blocks, pad

def tlut_reconstruct(d_lut, deltas, pad):
    y_rec = (d_lut[:,None].astype(np.int32) + deltas.astype(np.int32))
    y_rec = np.clip(y_rec, 0, 127).astype(np.uint8).reshape(-1)
    if pad: y_rec = y_rec[:-pad]
    return y_rec

def metrics_lsb(y_rec, y_full):
    err = np.abs(from_q1_7(y_rec) - from_q1_7(y_full))
    lsb = 1.0/(1<<Y_FRAC)
    mae = float(np.mean(err))/lsb
    mx  = float(np.max(err))/lsb
    return mae, mx, err*(1<<Y_FRAC)

# -------------------------
# Bits & tablas pedagógicas
# -------------------------
def q44_from_float(x):
    # dec → Q4.4 signed (para mostrar)
    q = int(np.round(x * (1<<X_FRAC)))
    if q < -128: q = -128
    if q >  127: q =  127
    return np.int8(q)

def make_flat_table(xs, y_real, y_q, show_bits=True):
    # Mostrar idx lógico (0..N-1) sobre el array xs
    df = pd.DataFrame({
        "idx": np.arange(len(xs)),
        "x_real": np.round(xs, 6),
        "f(x) real": np.round(y_real if True else y_real, 6),
        "y_q (Q1.7)": y_q.astype(int)
    })
    if show_bits:
        # x en Q4.4 (signed) y sus bits
        x_q44 = np.array([q44_from_float(x) for x in xs], dtype=np.int8)
        df["x_q4_4 (signed)"] = x_q44.astype(int)
        df["x_q4_4 (bin)"] = [bin_str(v,8) for v in x_q44]
        df["y_q (bin)"] = [bin_str(v,8) for v in y_q]
        df["y_q interpret (Q1.7)"] = [f"{bin_str(v,8)[0]}|{bin_str(v,8)[1:]}" for v in y_q]
    return df

def make_block_mapping_table(xs, y_q, block_size, d_lut, deltas):
    N = len(y_q)
    idx = np.arange(N)
    block_idx = idx // block_size
    offset    = idx % block_size
    d = d_lut[block_idx]
    delta = deltas[block_idx, offset]
    y_rec = np.clip(d.astype(int) + delta.astype(int), 0, 127).astype(int)

    df = pd.DataFrame({
        "idx": idx, "x_real": np.round(xs,6),
        "block_idx": block_idx.astype(int), "offset": offset.astype(int),
        "y_q(LUT)": y_q.astype(int),
        "d_base": d.astype(int),
        "delta": delta.astype(int),
        "y_rec": y_rec
    })
    df["d_base (bin)"] = [bin_str(v,8) for v in df["d_base"]]
    # delta tiene e_bits; vamos a mostrarla con ancho fijo de e_bits (notar que no la sabemos aquí, el caller la añade)
    return df

# -------------------------
# Visuales
# -------------------------
def plot_curve(xs, y_full, y_rec, block_size, func_name, mae, mx):
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
    fig.add_trace(go.Scatter(x=xs, y=from_q1_7(y_full), mode="lines", name="LUT completa"))
    fig.add_trace(go.Scatter(x=xs, y=from_q1_7(y_rec),  mode="lines", name="T-LUT reconstruida"))
    fig.update_layout(
        title=f"{func_name}(x) — Q4.4→Q1.7 | block={block_size} | MAE={mae:.2f} LSB, MAX={mx:.2f} LSB",
        xaxis_title="x (real)", yaxis_title=f"{func_name}(x) (escala Q1.7→[0,1))",
        height=360, legend=dict(orientation="h", y=1.02, x=0)
    )
    return fig

def plot_error(err_lsb):
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
    fig.add_trace(go.Scatter(x=list(range(len(err_lsb))), y=err_lsb, mode="lines", name="error [LSB]"))
    fig.update_layout(title="Error por muestra (LSBs de Q1.7)", xaxis_title="índice", yaxis_title="error [LSB]", height=300)
    return fig

def diagram_blocks_bar(n_blocks, block_size, highlight=None, title="LUT plana dividida en bloques"):
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
    fig.update_xaxes(range=[0, 100], visible=False)
    fig.update_yaxes(range=[0, 100], visible=False)
    shapes, ann = [], []
    x0, x1, y0, y1 = 5, 95, 70, 82
    W = (x1-x0)/n_blocks
    shapes.append(dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                       line=dict(color=BOX_LINE, width=2), fillcolor=PLOT_BG))
    ann.append(dict(x=(x0+x1)/2, y=y1+4, text=title, showarrow=False, font=dict(size=14, color="#e5e9f0")))
    for b in range(n_blocks):
        bx0 = x0 + b*W; bx1 = bx0+W
        fill = ALT_A if (b%2==0) else ALT_B
        line_c = BOX_LINE; line_w = 1
        if highlight is not None and b==highlight:
            fill, line_c, line_w = HL_FILL, HL_LINE, 2
        shapes.append(dict(type="rect", x0=bx0, x1=bx1, y0=y0, y1=y1,
                           line=dict(color=line_c, width=line_w), fillcolor=fill))
        ann.append(dict(x=(bx0+bx1)/2, y=(y0+y1)/2, text=f"blk {b}", showarrow=False, font=dict(size=12, color="#cbd5e1")))
    fig.update_layout(shapes=shapes, annotations=ann, height=220, margin=dict(l=20,r=20,t=40,b=10))
    return fig

def diagram_dlut_elut(d_lut, deltas, block_size, show_numbers=True, highlight=None):
    n_blocks = len(d_lut)
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
    fig.update_xaxes(range=[0, 100], visible=False)
    fig.update_yaxes(range=[0, 100], visible=False)
    shapes, ann = [], []

    # d-LUT panel
    dl_x0, dl_x1, dl_y0, dl_y1 = 5, 30, 10, 60
    H = (dl_y1-dl_y0)/n_blocks
    shapes.append(dict(type="rect", x0=dl_x0, x1=dl_x1, y0=dl_y0, y1=dl_y1,
                       line=dict(color=BOX_LINE, width=2), fillcolor=PLOT_BG))
    ann.append(dict(x=(dl_x0+dl_x1)/2, y=dl_y1+3, text="d-LUT (bases)", showarrow=False, font=dict(size=14, color="#e5e9f0")))
    for b in range(n_blocks):
        y0 = dl_y1 - (b+1)*H; y1 = y0+H
        fill = ALT_A if (b%2==0) else ALT_B; line_c=BOX_LINE; line_w=1
        if highlight is not None and b==highlight:
            fill, line_c, line_w = HL_FILL, HL_LINE, 2
        shapes.append(dict(type="rect", x0=dl_x0, x1=dl_x1, y0=y0, y1=y1,
                           line=dict(color=line_c, width=line_w), fillcolor=fill))
        text = f"d[{b}]"
        if show_numbers: text += f"={int(d_lut[b])} ({bin_str(d_lut[b],8)})"
        ann.append(dict(x=(dl_x0+dl_x1)/2, y=(y0+y1)/2, text=text, showarrow=False, font=dict(size=11, color="#cbd5e1")))

    # e-LUT panel (rejilla)
    el_x0, el_x1, el_y0, el_y1 = 40, 95, 10, 60
    cw = (el_x1-el_x0)/block_size
    ch = (el_y1-el_y0)/n_blocks
    shapes.append(dict(type="rect", x0=el_x0, x1=el_x1, y0=el_y0, y1=el_y1,
                       line=dict(color=BOX_LINE, width=2), fillcolor=PLOT_BG))
    ann.append(dict(x=(el_x0+el_x1)/2, y=el_y1+3, text="e-LUT (deltas)", showarrow=False, font=dict(size=14, color="#e5e9f0")))

    for b in range(n_blocks):
        y0 = el_y1 - (b+1)*ch; y1 = y0+ch
        shapes.append(dict(type="rect", x0=el_x0, x1=el_x1, y0=y0, y1=y1,
                           line=dict(color="rgba(0,0,0,0)"), fillcolor=ALT_B))
        for off in range(block_size):
            x0 = el_x0 + off*cw; x1 = x0+cw
            line_c = BOX_LINE; line_w = 1; fill = PLOT_BG
            if highlight is not None and b==highlight:
                fill = HL_FILL; line_c = HL_LINE; line_w = 2 if off in (0,block_size-1) else 1
            shapes.append(dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                               line=dict(color=line_c, width=line_w), fillcolor=fill))
            if show_numbers and highlight is not None and b==highlight:
                ann.append(dict(x=(x0+x1)/2, y=(y0+y1)/2,
                                text=str(int(deltas[b,off])), showarrow=False, font=dict(size=11, color="#cbd5e1")))
    fig.update_layout(shapes=shapes, annotations=ann, height=520, margin=dict(l=20,r=20,t=50,b=20))
    return fig

# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="Mini-guía T-LUT (paso a paso)", layout="wide")
st.title("Mini-guía T-LUT (paso a paso, con bits)")

with st.sidebar:
    st.header("Parámetros")
    func_name = st.selectbox("Función", ["sigmoid", "tanh"], index=0)
    mini_mode = st.checkbox("Modo MINI (16 muestras)", value=True)
    block_size = st.selectbox("block_size", [8,16,24,32] if not mini_mode else [8,16], index=0)
    e_bits = st.slider("e_bits (bits por delta)", 1, 8, 6, step=1)
    show_numbers = st.checkbox("Mostrar números en diagramas", value=True)

# Dominio y LUT plana
xs = build_domain(mini_mode)
y_real, y_q = build_lut_flat(xs, func_name)

# Compresión T-LUT
d_lut, deltas, n_blocks, pad = tlut_compress(y_q, block_size, e_bits)
y_rec = tlut_reconstruct(d_lut, deltas, pad)
mae, mx, err_lsb = metrics_lsb(y_rec, y_q)

# Paso a paso
st.subheader("Paso a paso")
step = st.radio(
    "Seleccioná el paso",
    ["1) LUT plana (con bits)", "2) División en bloques", "3) d-LUT (bases)", "4) e-LUT (deltas)", "5) Reconstrucción y error"],
    index=0, horizontal=True
)

# 1) LUT plana con bits
if step.startswith("1"):
    st.caption("LUT plana cuantizada a Q1.7. También mostramos x en Q4.4 (signed).")
    flat_df = make_flat_table(xs, y_real, y_q, show_bits=True)
    st.dataframe(flat_df, use_container_width=True, hide_index=True)

# 2) División en bloques + barra visual
elif step.startswith("2"):
    st.caption("Agrupamos consecutivos en bloques de tamaño block_size.")
    block_sel = st.slider("Bloque a resaltar", 0, n_blocks-1, 0, step=1)
    st.plotly_chart(diagram_blocks_bar(n_blocks, block_size, highlight=block_sel), use_container_width=True)
    # tabla corta de índices y pertenencia a bloques
    idx = np.arange(len(y_q))
    block_idx = idx // block_size
    offset = idx % block_size
    map_df = pd.DataFrame({"idx": idx, "block_idx": block_idx.astype(int), "offset": offset.astype(int), "y_q(Q1.7)": y_q.astype(int)})
    st.dataframe(map_df, use_container_width=True, hide_index=True)

# 3) d-LUT
elif step.startswith("3"):
    st.caption("Para cada bloque, d_base = mínimo del bloque (Q1.7).")
    block_sel = st.slider("Bloque a resaltar", 0, n_blocks-1, 0, step=1)
    st.plotly_chart(diagram_dlut_elut(d_lut, deltas, block_size, show_numbers=False, highlight=block_sel), use_container_width=True)
    dlut_df = pd.DataFrame({"block_idx": np.arange(n_blocks, dtype=int), "d_base (Q1.7)": d_lut.astype(int), "d_base (bin)": [bin_str(v,8) for v in d_lut]})
    st.dataframe(dlut_df, use_container_width=True, hide_index=True)

# 4) e-LUT
elif step.startswith("4"):
    st.caption("Deltas por bloque/offset: delta = y_q - d_base (clamp a 0..2^e_bits-1). Mostramos la fila del bloque seleccionado.")
    block_sel = st.slider("Bloque a resaltar", 0, n_blocks-1, 0, step=1)
    st.plotly_chart(diagram_dlut_elut(d_lut, deltas, block_size, show_numbers=True, highlight=block_sel), use_container_width=True)
    # tabla del bloque seleccionado con binarios de delta a e_bits
    start = block_sel * block_size
    end = min(start + block_size, len(y_q))
    block_view = pd.DataFrame({
        "idx": np.arange(start, end),
        "x_real": np.round(xs[start:end], 6),
        "y_q(LUT)": y_q[start:end].astype(int),
        "d_base": int(d_lut[block_sel]),
        "delta": deltas[block_sel, :end-start].astype(int),
        "y_rec": (np.clip(d_lut[block_sel] + deltas[block_sel, :end-start], 0, 127)).astype(int),
    })
    block_view["d_base (bin)"] = bin_str(d_lut[block_sel], 8)
    block_view["delta (bin)"]  = [bin_str(v, e_bits) for v in block_view["delta"]]
    st.dataframe(block_view, use_container_width=True, hide_index=True)

# 5) Reconstrucción y error
else:
    st.caption("En tiempo de ejecución: y = d_base + delta (saturado a 8 bits). Comparamos curvas y mostramos error.")
    st.plotly_chart(plot_curve(xs, y_q, y_rec, block_size, func_name, mae, mx), use_container_width=True)
    st.plotly_chart(plot_error(err_lsb), use_container_width=True)

    # Tabla de mapeo completa con binarios clave
    map_df = make_block_mapping_table(xs, y_q, block_size, d_lut, deltas)
    # agregar delta bin y y_rec bin
    map_df["delta (bin)"] = [bin_str(v, e_bits) for v in map_df["delta"]]
    map_df["y_rec (bin)"] = [bin_str(v, 8) for v in map_df["y_rec"]]
    st.dataframe(map_df, use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    """
**Resumen de flujo (software/RTL):**  
1) Interpretar `x_in` como Q4.4 (signed).  
2) (FULL) Saturar dominio: x<=-6 → y=0; x>=+6 → y=127.  
3) `idx = x_q4_4 - (-6*16) = x_q4_4 + 96` (en FULL).  
4) `block_idx = idx // block_size`, `offset = idx % block_size`.  
5) `d = d_base[block_idx]`, `δ = delta[block_idx, offset]`.  
6) `y = sat8(d + δ)`  (Q1.7).  
"""
)
