import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. 미분방정식 정의 (Van der Pol 오실레이터: dx/dt = v, dv/dt = mu*(1-x^2)*v - x)
def vdp_system(t, y, mu=1.0):
    x, v = y
    dxdt = v
    dvdt = mu * (1.0 - x**2) * v - x
    return [dxdt, dvdt]

# 2. 시뮬레이션 설정
t_span = (0.0, 20.0)          # 시작 시간과 종료 시간
t_eval = np.linspace(0, 20, 500) # 결과를 평가할 시간 간격
y0 = [2.0, 0.0]               # 초기 상태 (x=2.0, v=0.0)

# 3. 룽게쿠타법(RK45)으로 ODE 풀이 수행
sol = solve_ivp(vdp_system, t_span, y0, method='RK45', t_eval=t_eval)

# 4. 결과 시각화
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(sol.t, sol.y[0], label='x(t)', color='blue')
plt.plot(sol.t, sol.y[1], label='v(t)', color='orange', linestyle='--')
plt.title("Van der Pol Oscillator (Time Series)")
plt.xlabel("Time t")
plt.ylabel("States")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(sol.y[0], sol.y[1], color='purple')
plt.title("Phase Portrait")
plt.xlabel("x")
plt.ylabel("v (dx/dt)")
plt.grid(True)

plt.tight_layout()
plt.show()