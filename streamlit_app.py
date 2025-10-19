import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# non-interactive backend to avoid toolbar/boxes
matplotlib.use("Agg")

st.set_page_config(page_title="원형 코일 2D 자기장 시뮬레이터 (정밀 모드)", layout="wide")

st.title("🧲 원형 코일 2D 자기장 시뮬레이터 — 개선형 (정확도 상승)")
st.markdown("""
정밀도를 올리기 위해 벡터화된 수치적분을 사용합니다.  
`샘플링 방식`, `분할수(n)`, `근거리 컷오프(eps)`를 조절해 수렴을 확인하세요.
""")

# -----------------------
# Sidebar: 입력 변수
# -----------------------
st.sidebar.header("변수 설정")

I = st.sidebar.number_input("전류 I (A)", min_value=0.001, max_value=10.0, value=1.00, step=0.01, format="%.2f")
R = st.sidebar.number_input("코일 반지름 R (m)", min_value=0.01, max_value=5.0, value=1.00, step=0.01, format="%.2f")
N = st.sidebar.number_input("코일 감은 수 N (회)", min_value=1, max_value=100, value=1, step=1, format="%d")
x = st.sidebar.number_input("X 좌표 (m)", min_value=-2.0, max_value=2.0, value=0.99, step=0.01, format="%.2f")
y = st.sidebar.number_input("Y 좌표 (m)", min_value=-2.0, max_value=2.0, value=0.00, step=0.01, format="%.2f")
z = st.sidebar.number_input("Z 좌표 (m)", min_value=-2.0, max_value=2.0, value=0.00, step=0.01, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.header("수치 설정 (정밀도)")
n_elements = st.sidebar.number_input("분할수 n (권장: 2000~50000)", min_value=100, max_value=200000, value=20000, step=100, format="%d")
sampling = st.sidebar.selectbox("샘플링 방식", options=["midpoint", "endpoint"], index=0,
                                help="midpoint: 각 dl의 중심에서 샘플링 (더 안정적), endpoint: 구간 시작점 샘플링")
eps = st.sidebar.number_input("근거리 컷오프 eps (m)", min_value=0.0, max_value=1e-2, value=0.0, step=1e-6, format="%.6f",
                              help="r_mag를 최소 eps로 제한하여 수치적 특이점을 완화")
use_finite_radius = st.sidebar.checkbox("유한 도선 반경 근사 사용 (eps = 도선 반지름 a)", value=False)
if use_finite_radius:
    a = st.sidebar.number_input("도선 반지름 a (m)", min_value=1e-4, max_value=0.5, value=1e-3, step=1e-4, format="%.4f")
else:
    a = 0.0

st.sidebar.markdown("---")
st.sidebar.write("계산된 값의 신뢰도를 보려면 n을 증가시키고 '재계산'을 눌러 비교하세요.")

# -----------------------
# 물리 상수
# -----------------------
mu0 = 4 * np.pi * 1e-7  # H/m

# -----------------------
# 벡터화된 Biot-Savart (빠른 구현)
# -----------------------
def Bz_vectorized(rho_x, rho_y, rho_z, I, R, N=1, n=20000, sampling="midpoint", eps=0.0):
    """
    벡터화된 Biot-Savart 계산 (Z 성분)
    rho_x, rho_y, rho_z: 관찰점 좌표
    sampling: 'midpoint' 또는 'endpoint'
    eps: 최소 거리(정규화)
    반환: Bz (T) 및 (몇 개의) 샘플 단계 정보
    """
    # theta 샘플
    if sampling == "endpoint":
        theta = np.linspace(0.0, 2*np.pi, n, endpoint=False)
        dtheta = 2*np.pi / n
    else:  # midpoint
        dtheta = 2*np.pi / n
        theta = (np.linspace(0.0, 2*np.pi, n, endpoint=False) + 0.5 * dtheta)

    # 소자 위치와 dl (벡터화)
    rx = R * np.cos(theta)   # x 위치
    ry = R * np.sin(theta)   # y 위치
    dlx = -R * np.sin(theta) * dtheta
    dly = R * np.cos(theta) * dtheta
    dlz = np.zeros_like(dlx)

    # r 벡터 components (arrays)
    rx_to_obs = rho_x - rx  # vector from element to observation, x-comp
    ry_to_obs = rho_y - ry  # y-comp
    rz_to_obs = np.full_like(rx_to_obs, rho_z)  # z-comp (scalar repeated)

    # 거리
    r_mag = np.sqrt(rx_to_obs**2 + ry_to_obs**2 + rz_to_obs**2)
    if eps > 0.0:
        r_mag = np.maximum(r_mag, eps)

    # cross(dl, r) z-component = dlx*ry_to_obs - dly*rx_to_obs
    cross_z = dlx * ry_to_obs - dly * rx_to_obs

    # dBz array
    factor = mu0 * I / (4.0 * np.pi)
    dBz = factor * (cross_z / (r_mag**3))

    Bz_total = np.sum(dBz) * N

    # sample steps (10 points) for inspection
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

    return Bz_total, calc_steps

# -----------------------
# 계산 실행 (현재 n과 2n 비교)
# -----------------------
# apply finite radius if requested
if use_finite_radius and a > 0.0:
    eps_used = max(eps, a)
else:
    eps_used = eps

# compute with n and 2n to estimate convergence
B_n, steps_n = Bz_vectorized(x, y, z, I, R, N, n=int(n_elements), sampling=sampling, eps=eps_used)
B_2n, steps_2n = Bz_vectorized(x, y, z, I, R, N, n=int(min(n_elements*2, 200000)), sampling=sampling, eps=eps_used)

# convert to π×10^-7 units for display
pi_unit = np.pi * 1e-7
B_n_pi = B_n / pi_unit
B_2n_pi = B_2n / pi_unit
rel_diff = abs(B_2n - B_n) / (abs(B_2n) + 1e-30)

# -----------------------
# 출력: 숫자 및 π표현
# -----------------------
st.markdown("### 측정 결과 (정밀 계산)")
st.write(f"선택 위치 (X,Y,Z) = ({x:.4f}, {y:.4f}, {z:.4f}) m")
st.write(f"전류 I = {I:.4f} A, 반지름 R = {R:.4f} m, 감은수 N = {N:d}")
st.write(f"샘플링 = {sampling}, n = {int(n_elements)}, eps_used = {eps_used:.6e} m")

st.markdown("---")
st.write("수치 결과 (n 기준):")
st.write(f"  Bz = {B_n:.12e} T")
st.write(f"  (π × 10^-7 단위) Bz = {B_n_pi:.6f} × π × 10^-7 T")
st.write("")
st.write("수렴 확인 (2n 계산):")
st.write(f"  Bz(2n) = {B_2n:.12e} T")
st.write(f"  (π × 10^-7 단위) Bz(2n) = {B_2n_pi:.6f} × π × 10^-7 T")
st.write(f"  상대차이 |B(2n)-B(n)|/|B(2n)| = {rel_diff:.3e}")

st.markdown("---")
st.write("원하면 `n`을 키우거나 `샘플링=midpoint` 및 작은 `eps`로 재계산하여 값이 수렴하는지 확인하세요.")

# -----------------------
# 계산 과정(expander) — 일부 샘플링 단계 표시
# -----------------------
with st.expander("계산 과정(샘플 일부) — dl, r, r_mag, dBz"):
    st.markdown("아래는 n 샘플 중 일부 단계에서의 값(예시)")
    for s in steps_n:
        st.write(f"i={s['i']} | dl={s['dl']} | r_vec={s['r_vec']} | |r|={s['r_mag']:.6e} | dBz={s['dBz']:.6e} T")

# -----------------------
# 시각화: 코일, 관찰점, dl별 dBz 화살표 (화살표 ON/OFF)
# -----------------------
show_arrows = st.checkbox("dl 소자 화살표 표시", value=True)
fig, ax = plt.subplots(figsize=(6,6))
circle = plt.Circle((0,0), R, fill=False, color='orange', linewidth=2, label='코일 (R)')
ax.add_patch(circle)
ax.plot(x, y, 'ro', markersize=8, label='관찰점')

if show_arrows:
    # 시각화를 위해 dl sample 수를 줄임 (겹침 방지)
    vis_n = min(500, int(n_elements))
    _, _, dl_positions_vis, dB_vectors_vis = Bz_vectorized(x, y, z, I, R, N, n=vis_n, sampling=sampling, eps=eps_used)
    # dB_vectors_vis는 dBz (T) 배열 length vis_n
    # 표시 스케일 조절
    dB_arr = np.array([d['dBz'] for d in dl_positions_vis]) if False else None  # placeholder (not used)

    # recompute vectorized arrays for visualization (simple)
    theta_vis = (np.linspace(0.0, 2*np.pi, vis_n, endpoint=False) + (0.5 * (2*np.pi/vis_n) if sampling=='midpoint' else 0.0))
    rx_vis = R * np.cos(theta_vis)
    ry_vis = R * np.sin(theta_vis)
    dtheta_vis = 2*np.pi/vis_n
    dlx_vis = -R * np.sin(theta_vis) * dtheta_vis
    dly_vis = R * np.cos(theta_vis) * dtheta_vis
    rx_to_obs_vis = x - rx_vis
    ry_to_obs_vis = y - ry_vis
    rz_to_obs_vis = np.full_like(rx_to_obs_vis, z)
    rmag_vis = np.sqrt(rx_to_obs_vis**2 + ry_to_obs_vis**2 + rz_to_obs_vis**2)
    rmag_vis = np.maximum(rmag_vis, eps_used)
    cross_z_vis = dlx_vis * ry_to_obs_vis - dly_vis * rx_to_obs_vis
    factor = mu0 * I / (4.0 * np.pi)
    dBz_vis = factor * (cross_z_vis / (rmag_vis**3)) * N

    # draw arrows (scaled)
    max_abs = np.max(np.abs(dBz_vis)) if dBz_vis.size>0 else 1.0
    if max_abs == 0:
        scale = 1.0
    else:
        scale = 0.15 / max_abs  # heuristic scaling for visibility
    for px, py, db in zip(rx_vis, ry_vis, dBz_vis):
        ax.arrow(px, py, 0.0, db*scale, head_width=0.02, head_length=0.02, fc='blue', ec='blue', alpha=0.8)

ax.set_xlim(-1.5*R, 1.5*R)
ax.set_ylim(-1.5*R, 1.5*R)
ax.set_aspect('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title("코일과 관찰점 (dl 소자 화살표는 dBz 방향, 스케일 조정됨)")
ax.legend()
ax.grid(True)
st.pyplot(fig, use_container_width=True)

# -----------------------
# 간단 권장 메시지
# -----------------------
st.info("팁: 관찰점이 도선에 매우 가까우면 n을 크게(수만 단위) 늘리고 midpoint 샘플링을 선택하세요. 유한 도선 반경이 필요한 경우 '유한 도선 반경 근사'를 켜고 실제 도선 반지름을 입력하세요.")
