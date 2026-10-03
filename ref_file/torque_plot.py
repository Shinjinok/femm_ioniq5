import matplotlib.pyplot as plt

# 첫 번째 그래프 데이터
angle = [
    0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310,
    320, 330, 340, 350
]
torque_1 = [
    260, 330, 390, 430, 450, 450, 410, 315, 175, 0, -175, -315, -410, -450,
    -450, -430, -385, -325, -260, -190, -125, -75, -30, 15, 35, 40, 25, 0, -20,
    -35, -35, -10, 30, 75, 130, 195
]

# 두 번째 그래프 데이터
torque_2 = [
    0, 52, 115, 170, 208, 225, 215, 170, 92, 0, -92, -170, -215, -225, -208,
    -170, -115, -52, 0, 52, 115, 170, 208, 225, 215, 170, 92, 0, -92, -170,
    -215, -225, -208, -170, -115, -52
]

# 첫 번째 - 두 번째 데이터 차이 계산
torque_diff = [t1 - t2 for t1, t2 in zip(torque_1, torque_2)]

# 그래프 설정
plt.figure(figsize=(12, 6))

# 데이터 플롯
plt.plot(
    angle,
    torque_1,
    marker='o',
    color='darkorange',
    linewidth=2,
    label='Total Torque'
)
plt.plot(
    angle,
    torque_2,
    marker='s',
    color='dodgerblue',
    linewidth=2,
    label='Reluctance Torque'
)
plt.plot(
    angle,
    torque_diff,
    marker='^',
    color='forestgreen',
    linewidth=2,
    label='Magnetic Torque'
)

# 축 라벨 및 제목
plt.title('Electrical Angle vs Electromagnetic Torque (With Difference)', fontsize=14)
plt.xlabel('Electrical Angle [deg]', fontsize=12)
plt.ylabel('Torque [Nm]', fontsize=12)

# 축 범위 및 그리드
plt.xlim(0, 350)
plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
plt.grid(True, linestyle='--', alpha=0.6)

# 범례 및 레이아웃
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()