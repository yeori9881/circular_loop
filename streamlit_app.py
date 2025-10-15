import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="원형 코일 2D 자기장 시뮬레이터", layout="wide")

st.title("🧲 원형 코일 2D 자기장 시뮬레이터")
st.markdown("""
마우스로 화면 위 위치를 선택하면 해당 지점에서의 자기장 세기를 계산합니다.  
전류, 코일 반지름, 감은 수 조절 가능, dl 소자 화살표 ON/OFF, 계산 과정 확인 가능
""")

# --- Sidebar: 변수 설정 ---
st.sidebar.header("⚙️ 변수 설정")

I = st.sidebar.number_input("전류 I (A)", min_value=0.1, max_value=10.0, value=2.0, step=0.1, format="%.1f")
R = st.sidebar.number_input("코일 반지름 R (m)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, format="%.1f")
N = st.sidebar.number_input("코일 감은 수 N (회)", min_value=1, max_value=20, value=5, step=1, format="%d")
x = st.sidebar.number_input("X 좌표 (m)", min_value=-2.0, max_value=2.0, value=0.5, step=0.1, format="%.1f")
y = st.sidebar.number_input("Y 좌표 (m)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, format="%.1f")
z = st.sidebar.number_input("Z 좌표 (m)", min_value=-1.0, max_value=1.0, value=0.0, step=0.1, format="%.1f")

# μ₀/4 계산 상수
mu0_div_4 = 1e-7  # μ₀/4 계산
pi_symbol = "π"   # 계산 과정에서는 문자로 표현

# --- Biot-Savart 법칙 함수 ---
def Bz_point_verbose(x, y, z, I, R, N=1, n_elements=200):
    theta = np.linspace(0, 2*np.pi, n_elements, endpoint=False)
    rx = R * np.cos(theta)
    ry = R * np.sin(theta)
    dlx = -R * np.sin(theta) * (2*np.pi/n_elements)
    dly = R * np.cos(theta) * (2*np.pi/n_elements)

    Bz_total = 0.0
    calc_steps = []
    dl_positions = []
    dB_vectors = []

    for i in range(n_elements):
        r_vec = np.array([x - rx[i], y - ry[i], z])
        dl_vec = np.array([dlx[i], dly[i], 0.0])
        r_mag = np.linalg.norm(r_vec)
        if r_mag == 0:
            continue
        dB = (mu0_div_4 * I) * np.cross(dl_vec, r_vec) / (r_mag**3)
        Bz_total += dB[2]

        if i % max(1, n_elements // 10) == 0:
            step_info = {
                "i": i,
                "dl_vector": dl_vec,
                "r_vector": r_vec,
                "r_mag": r_mag,
                "dB_vector": dB,
                "dBz": dB[2]
            }
            calc_steps.append(step_info)

        dl_positions.append((rx[i], ry[i]))
        dB_vectors.append(dB[2])

    return Bz_total * N, calc_steps, dl_positions, dB_vectors

# --- 계산 ---
B_here, calc_steps, dl_positions, dB_vectors = Bz_point_verbose(x, y, z, I, R, N)

# --- 결과 ---
st.markdown(f"### 📊 측정 결과")
st.markdown(f"**선택 위치 (X,Y,Z) = ({x:.1f}, {y:.1f}, {z:.1f}) m**")
st.markdown(f"**Z축 방향 자기장 Bz = {B_here:.3e} T**")
st.caption(f"계산식 예시: Bz = Σ ({mu0_div_4}*I/{pi_symbol}) * (dl × r)/|r|³")

# --- 계산 과정 보기 ---
with st.expander("🔍 계산 과정 보기"):
    st.markdown(f"**사용 공식:** Bz = Σ (μ₀ I / 4{pi_symbol}) * (dl × r) / |r|³  (Z축 방향만)")
    st.markdown("**각 dl 소자가 선택 위치에서 만드는 Bz 계산 과정:**")
    for step in calc_steps:
        st.markdown(
            f"i={step['i']} | dl={step['dl_vector']} | r={step['r_vector']} | "
            f"|r|={step['r_mag']:.3f} | dB={step['dB_vector']} | dBz={step['dBz']:.3e}"
        )

# --- 시각화 ---
show_arrows = st.checkbox("💠 dl 소자 화살표 표시", value=True)
fig, ax = plt.subplots(figsize=(6,6))
circle = plt.Circle((0,0), R, fill=False, color='orange', linewidth=2, label='코일')
ax.add_patch(circle)
ax.plot(x, y, 'ro', markersize=8, label=f'측정 위치 ({x:.1f},{y:.1f}) m')

if show_arrows:
    for (px, py), dBz in zip(dl_positions, dB_vectors):
        scale = 1e8
        ax.arrow(px, py, 0, dBz*scale, head_width=0.02, head_length=0.02, fc='blue', ec='blue')

ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_aspect('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title("XY 평면: 원형 코일과 dl 소자가 만드는 Bz 화살표")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- 공식 이미지 표시 ---
with st.expander("📝 관련 공식 및 개념 설명"):
    st.markdown("**1️⃣ 원형 코일 중심 Z축 자기장 공식**")
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/88/Solenoid_B_field.svg", caption="원형 코일 자기장 개념")
    st.markdown(
        "Bz = μ₀ I N R² / (2 (R² + z²)^(3/2))\n\n"
        "- μ₀: 진공 투자율 (4π×10⁻⁷ H/m)\n"
        "- I: 전류 (A)\n"
        "- N: 코일 감은 수\n"
        "- R: 코일 반지름 (m)\n"
        "- z: 중심에서 떨어진 거리 (m)"
    )

    st.markdown("**2️⃣ Biot-Savart 법칙**")
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/5d/Biot-Savart_law.svg", caption="Biot-Savart 법칙")
    st.markdown(
        "dB = (μ₀ I / 4π) * (dl × r) / |r|³\n\n"
        "- dl: 미소 전류 요소 벡터\n"
        "- r: 관찰점까지 위치 벡터\n"
        "- |r|: 거리"
    )
