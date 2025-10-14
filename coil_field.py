import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 기본 설정 ---
st.set_page_config(page_title="원형 코일 자기장 시뮬레이터", layout="wide")

st.title("🧲 원형 코일 자기장 시뮬레이터")
st.markdown("""
코일 중심이 아닌 지점에서의 자기장 세기를 계산하고 시각화합니다.  
전류, 코일 반지름, 감은 수, 관찰 위치를 바꿔보세요!
""")

# --- 사이드바: 변수 입력 ---
st.sidebar.header("⚙️ 변수 설정")
I = st.sidebar.slider("전류 I (A)", 0.1, 10.0, 2.0, 0.1)
R = st.sidebar.slider("코일 반지름 R (m)", 0.1, 2.0, 0.5, 0.1)
N = st.sidebar.slider("코일 감은 수 N (회)", 1, 20, 5)
z = st.sidebar.slider("관찰점 위치 z (m)", -2.0, 2.0, 0.0, 0.1)

# --- 물리 상수 ---
mu0 = 4 * np.pi * 1e-7  # 진공 투자율 (T·m/A)

# --- 계산식 ---
Bz = mu0 * N * I * R**2 / (2 * (R**2 + z**2)**(1.5))

# --- 축 방향 자기장 분포 ---
z_values = np.linspace(-2, 2, 400)
B_values = mu0 * N * I * R**2 / (2 * (R**2 + z_values**2)**(1.5))

fig, ax = plt.subplots()
ax.plot(z_values, B_values, color='royalblue', label="자기장 분포")
ax.axvline(z, color='red', linestyle='--', label=f"현재 위치 z = {z:.2f} m")
ax.set_xlabel("z (m)")
ax.set_ylabel("자기장 세기 B (T)")
ax.set_title("코일 축 방향의 자기장 분포")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# --- 결과 출력 ---
st.markdown(f"### 📍 현재 위치 z={z:.2f} m에서의 자기장 세기:")
st.markdown(f"**B = {Bz:.3e} T**")
st.caption("공식:  B = (μ₀ N I R²) / [2(R² + z²)^(3/2)]")
