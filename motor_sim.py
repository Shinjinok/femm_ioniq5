import matplotlib.pyplot as plt

L = 0.00001  # 코일의 인덕턴스
R = 5.0    # 코일의 저항
E = 10.0   # 코일의 가해지는 전압
dt = 0.000001  # 시간의 변화량
i = 0.0    # 전류

time_list = []
current_list = []
v_R_list = []
v_L_list = []

t = 0.0
while t < 0.002:
    di = (E - i * R) / L * dt
    i += di

    v_R = i * R
    v_L = E - i * R

    time_list.append(t)
    current_list.append(i)
    v_R_list.append(v_R)
    v_L_list.append(v_L)

    t += dt

# 그래프 그리기
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time_list, current_list, label='Current (i)', color='b')
plt.ylabel('Current [A]')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_list, v_R_list, label='v_R (Resistor)', color='r')
plt.plot(time_list, v_L_list, label='v_L (Inductor)', color='g')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()