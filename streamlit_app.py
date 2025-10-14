import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="원형 코일 2D 자기장 시뮬레이터", layout="wide")

st.title("🧲 원형 코일 2D 자기장 시뮬레이터")
st.markdown("""
마우스로 화면 위 위치를 선택하면 해당 지점에서의 자기장 세기를 계산합니다.  
전류, 코일 반지름, 감은 수를 조절할 수 있습니다.
""")

# --- Sidebar: 변수 설정 ---
st.sidebar.header("⚙️ 변수 설정")
I = st.sidebar.slider("전류 I (A)", 0.1, 10.0, 2.0, 0.1)
R = st.sidebar.slider("코일 반지름 R (m)", 0.1, 2.0, 0.5, 0.1)
N = st.sidebar.slider("코일 감은 수 N (회)", 1, 20, 5)

mu0 = 4 * np.pi * 1e-7  # 진공 투자율

# --- 마우스로 선택할 위치 ---
st.markdown("### 📍 측정할 위치 선택")
x = st.slider("X 좌표 (m)", -2.0, 2.0, 0.5, 0.01)
y = st.slider("Y 좌표 (m)", -2.0, 2.0, 0.0, 0.01)
z = st.slider("Z 좌표 (m)", -1.0, 1.0, 0.0, 0.01)

# --- Biot-Savart 법칙 기반 Bz 계산 함수 ---
def Bz_point(x, y, z, I, R, N=1, n_elements=200):
    """
    XY 평면 원형 코일 중심(0,0) 기준, Z축 방향 자기장 계산
    Biot-Savart 법칙을 수치적으로 근사
    """
    theta = np.linspace(0, 2*np.pi, n_elements)
    # 코일 소자 위치
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
        Bz_total += dB[2]  # Z축 방향만 취함
    return Bz_total * N

# --- 자기장 계산 ---
B_here = Bz_point(x, y, z, I, R, N)

# --- 시각화 ---
fig, ax = plt.subplots(figsize=(6,6))
# 코일 표시
circle = plt.Circle((0,0), R, fill=False, color='orange', linewidth=2, label='코일')
ax.add_patch(circle)
# 선택 지점 표시
ax.plot(x, y, 'ro', markersize=8, label=f'측정 위치 ({x:.2f}, {y:.2f}) m')
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
st.markdown(f"**선택 위치 (X,Y,Z) = ({x:.2f}, {y:.2f}, {z:.2f}) m**")
st.markdown(f"**Z축 방향 자기장 Bz = {B_here:.3e} T**")
st.caption("Biot-Savart 법칙을 수치적분으로 계산한 값")
