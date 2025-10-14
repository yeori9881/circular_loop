import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="원형 코일 자기장 시뮬레이터", layout="wide")

st.title("🧲 원형 코일 자기장 시뮬레이터")
st.markdown("코일 중심이 아닌 지점에서의 자기장 세기를 실험해보세요!")

# --- Sidebar: 변수 설정 ---
st.sidebar.header("⚙️ 실험 변수 조절")
I = st.sidebar.slider("전류 I (A)", 0.1, 10.0, 2.0, 0.1)
R = st.sidebar.slider("코일 반지름 R (m)", 0.1, 2.0, 0.5, 0.1)
N = st.sidebar.slider("코일 감은 수 N (회)", 1, 20, 5)
z = st.sidebar.slider("관찰점의 높이 z (m)", -2.0, 2.0, 0.0, 0.1)

mu0 = 4 * np.pi * 1e-7  # 투자율

# --- 계산 ---
Bz = mu0 * N * I * R**2 / (2 * (R**2 + z**2)**(1.5))

# --- 그래프 ---
z_values = np.linspace(-2, 2, 400)
B_values = mu0 * N * I * R**2 / (2 * (R**2 + z_values**2)**(1.5))

fig, ax = plt.subplots()
ax.plot(z_values, B_values, color='royalblue')
ax.axvline(z, color='red', linestyle='--', label=f"현재 위치 z = {z:.2f} m")
ax.set_xlabel("z (m)")
ax.set_ylabel("자기장 세기 B (T)")
ax.set_title("코일 축 방향의 자기장 분포")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# --- 결과 ---
st.markdown(f"### 📍 현재 z={z:.2f} m에서의 자기장 세기:")
st.markdown(f"**B = {Bz:.3e} T**")
st.caption("공식: B = (μ₀ N I R²) / (2(R² + z²)^(3/2))")
