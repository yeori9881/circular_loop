import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
st.set_page_config(page_title="원형 코일 2D 자기장 시뮬레이터", layout="wide")

st.title("원형 코일 2D 자기장 시뮬레이터 — 개선형)")
st.markdown("""
정밀도를 올리기 위해 벡터화된 수치적분을 사용합니다.  
`샘플링 방식`, `분할수(n)`, `근거리 컷오프(eps)`를 조절해 수렴을 확인하세요.
""")

# -----------------------
# Sidebar: 입력 변수
# -----------------------
st.sidebar.header("변수 설정")
I = st.sidebar.number_input("전류 I (A)", min_value=0.001, max_value=10.0, value=1.0, step=0.01)
R = st.sidebar.number_input("코일 반지름 R (m)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
N = st.sidebar.number_input("코일 감은 수 N (회)", min_value=1, max_value=100, value=1, step=1)
x = st.sidebar.number_input("X 좌표 (m)", min_value=-2.0, max_value=2.0, value=0.99, step=0.01)
y = st.sidebar.number_input("Y 좌표 (m)", min_value=-2.0, max_value=2.0, value=0.0, step=0.01)
z = st.sidebar.number_input("Z 좌표 (m)", min_value=-2.0, max_value=2.0, value=0.0, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("수치 설정 (정밀도)")
n_elements = st.sidebar.number_input("분할수 n (권장: 2000~50000)", min_value=100, max_value=200000, value=20000, step=100)
sampling = st.sidebar.selectbox("샘플링 방식", options=["midpoint", "endpoint"], index=0)
eps = st.sidebar.number_input("근거리 컷오프 eps (m)", min_value=0.0, max_value=1e-2, value=0.0, step=1e-6, format="%.6f")
use_finite_radius = st.sidebar.checkbox("유한 도선 반경 근사 사용", value=False)
a = 0.0
if use_finite_radius:
    a = st.sidebar.number_input("도선 반지름 a (m)", min_value=1e-4, max_value=0.5, value=1e-3, step=1e-4)

st.sidebar.markdown("---")
st.sidebar.write("설정을 마친 후 '계산' 버튼을 눌러주세요.")

# -----------------------
# 물리 상수
# -----------------------
mu0 = 4 * np.pi * 1e-7

# -----------------------
# Biot-Savart Z 계산 함수
# -----------------------
def Bz_vectorized(rho_x, rho_y, rho_z, I, R, N=1, n=20000, sampling="midpoint", eps=0.0):
    if sampling == "endpoint":
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        dtheta = 2*np.pi / n
    else:
        dtheta = 2*np.pi / n
        theta = np.linspace(0, 2*np.pi, n, endpoint=False) + 0.5*dtheta

    # 소자 위치 및 dl
    rx = R * np.cos(theta)
    ry = R * np.sin(theta)
    dlx = -R * np.sin(theta) * dtheta
    dly = R * np.cos(theta) * dtheta
    dlz = np.zeros_like(dlx)

    # 관찰점까지 벡터
    rx_to_obs = rho_x - rx
    ry_to_obs = rho_y - ry
    rz_to_obs = np.full_like(rx_to_obs, rho_z)
    r_mag = np.sqrt(rx_to_obs**2 + ry_to_obs**2 + rz_to_obs**2)
    if eps > 0.0:
        r_mag = np.maximum(r_mag, eps)

    # cross(dl, r) z-component
    cross_z = dlx*ry_to_obs - dly*rx_to_obs
    factor = mu0 * I / (4*np.pi)
    dBz = factor * (cross_z / r_mag**3)
    Bz_total = np.sum(dBz) * N

    # 일부 샘플 단계
    sample_indices = np.linspace(0, n-1, 10, dtype=int)
    calc_steps = []
    for i in sample_indices:
        calc_steps.append({
            "i": int(i),
            "dl": (float(dlx[i]), float(dly[i]), 0.0),
            "r_vec": (float(rx_to_obs[i]), float(ry_to_obs[i]), float(rz_to_obs[i])),
            "r_mag": float(r_mag[i]),
            "dBz": float(dBz[i]),
        })

    # 시각화용 배열
    dl_array = np.column_stack((rx, ry, np.zeros_like(rx)))
    dB_array = dBz
    return Bz_total, calc_steps, dl_array, dB_array

# -----------------------
# 계산 버튼
# -----------------------
if st.button("계산"):
    eps_used = max(eps, a) if use_finite_radius else eps
    B_n, steps_n, dl_vis, dB_vis = Bz_vectorized(x, y, z, I, R, N, n=int(n_elements), sampling=sampling, eps=eps_used)
    B_2n, steps_2n, _, _ = Bz_vectorized(x, y, z, I, R, N, n=int(min(n_elements*2,200000)), sampling=sampling, eps=eps_used)

    B_n_pi = B_n / (np.pi*1e-7)
    rel_diff = abs(B_2n - B_n)/(abs(B_2n)+1e-30)

    # -----------------------
    # 한 줄 요약 출력
    # -----------------------
    st.markdown("### 결과")
    st.write(f"선택 위치 (X,Y,Z) = ({x:.4f}, {y:.4f}, {z:.4f}) m "
             f"Z축 방향 자기장 Bz ≈ {B_n_pi:.3f} π × 10^-7 T "
             f"(상대차이 2n-n: {rel_diff:.3e})")

    # -----------------------
    # 계산 과정(expander)
    # -----------------------
    with st.expander("계산 과정(샘플 일부)"):
        for s in steps_n:
            st.write(f"i={s['i']} | dl={s['dl']} | r_vec={s['r_vec']} | |r|={s['r_mag']:.6e} | dBz={s['dBz']:.6e} T")

    # -----------------------
    # 시각화
    # -----------------------
    show_arrows = st.checkbox("dl 소자 화살표 표시", value=True)
    fig, ax = plt.subplots(figsize=(6,6))
    circle = plt.Circle((0,0), R, fill=False, color='orange', linewidth=2, label='코일 (R)')
    ax.add_patch(circle)
    ax.plot(x, y, 'ro', markersize=8, label='관찰점')

    if show_arrows:
        vis_n = min(500, int(n_elements))
        _, _, dl_vis_small, dB_vis_small = Bz_vectorized(x, y, z, I, R, N, n=vis_n, sampling=sampling, eps=eps_used)
        max_abs = np.max(np.abs(dB_vis_small)) if dB_vis_small.size>0 else 1.0
        scale = 0.15 / max_abs if max_abs != 0 else 1.0
        for dl_vec, db in zip(dl_vis_small, dB_vis_small):
            px, py = dl_vec[0], dl_vec[1]
            ax.arrow(px, py, 0.0, db*scale, head_width=0.02, head_length=0.02, fc='blue', ec='blue', alpha=0.8)

    ax.set_xlim(-1.5*R, 1.5*R)
    ax.set_ylim(-1.5*R, 1.5*R)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title("코일과 관찰점 (dl 화살표 = dBz 방향)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig, use_container_width=True)

    st.info("팁: 관찰점이 도선에 매우 가까우면 n을 크게 늘리고 midpoint 샘플링을 선택하세요. 필요 시 '유한 도선 반경 근사' 사용.")
