import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="원형 코일 2D 자기장 시뮬레이터", layout="wide")

st.title("🧲 원형 코일 2D 자기장 시뮬레이터")
st.markdown("""
마우스로 화면 위 위치를 선택하면 해당 지점에서의 자기장 세기를 계산합니다.  
전류, 코일 반지름, 감은 수를 조절할 수 있습니다.
""")

# --- Sidebar: 변수 설정 (슬라이더 + 직접 입력) ---
st.sidebar.header("⚙️ 변수 설정")

I = st.sidebar.number_input(
    "전류 I (A)", min_value=0.1, max_value=10.0, value=2.0, step=0.1, format="%.1f"
)
R = st.sidebar.number_input(
    "코일 반지름 R (m)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, format="%.1f"
)
N = st.sidebar.number_input(
    "코일 감은 수 N (회)", min_value=1, max_value=20, value=5, step=1, format="%d"
)

x = st.sidebar.number_input(
    "X 좌표 (m)", min_value=-2.0, max_value=2.0, value=0.5, step=0.1, format="%.1f"
)
y = st.sidebar.number_input(
    "Y 좌표 (m)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, format="%.1f"
)
z = st.sidebar.number_input(
    "Z 좌표 (m)", min_value=-1.0, max_value=1.0, value=0.0, step=0.1, format="%.1f"
)

mu0 = 4 * np.pi * 1e-7  # 진공 투자율

# --- Biot-Savart 법칙 기반 Bz 계산 함수 ---
def Bz_point(x, y, z, I, R, N=1, n_elements=200):
    """
    XY 평면 원형 코일 중심(0,0) 기준, Z축 방향 자기장 계산
    Biot-Savart 법칙을 수치적으로 근사
    """
    theta = np.linspace(0, 2*np.pi, n_elements)
    rx = R * np.cos(theta)
    ry = R * np.sin(theta)
    dlx = -R * np.sin(theta) * (2*np.pi/n_elements)
    dly = R * np.cos(theta) * (2*np.pi/n_elements)

    Bz_total = 0.0
    for i in range(n_elements):
        r_vec = np.array([x - rx[i], y - ry[i], z])
        dl_vec = np.array([dlx[i], dly[i], 0.0])
        r_mag = np.linalg.norm(r_vec)
        if r_mag == 0:
            continue
        dB = (mu0 * I / (4*np.pi)) * np.cross(dl_vec, r_vec) / (r_mag**3)
        Bz_total += dB[2]
    return Bz_total * N

# --- 자기장 계산 ---
B_here = Bz_point(x, y, z, I, R, N)

# --- 시각화 ---
fig, ax = plt.subplots(figsize=(6,6))
circle = plt.Circle((0,0), R, fill=False, color='orange', linewidth=2, label='코일')
ax.add_patch(circle)
ax.plot(x, y, 'ro', markersize=8, label=f'측정 위치 ({x:.1f}, {y:.1f}) m')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title("XY 평면에서 원형 코일과 측정 위치")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# --- 결과 출력 ---
st.markdown(f"### 📊 측정 결과")
st.markdown(f"**선택 위치 (X,Y,Z) = ({x:.1f}, {y:.1f}, {z:.1f}) m**")
st.markdown(f"**Z축 방향 자기장 Bz = {B_here:.3e} T**")
st.caption("Biot-Savart 법칙을 수치적분으로 계산한 값")
