import matplotlib.pyplot as plt
import numpy as np

# 1. 0부터 2파이(약 6.28)까지 0.1 간격으로 x값 생성
x = np.arange(0, 2 * np.pi, 0.1)

# 2. 코사인 값 계산
y = -np.cos(2*x-2*np.pi/3)  # y = -cos(2x + 2π/3)

# 3. 그래프 크기 및 설정
plt.figure(figsize=(6, 3), dpi=100)
plt.plot(x, y, label='y = cos(x)', color='tab:blue', linewidth=2)

# 4. 그래프 디자인 및 라벨 추가
plt.title('Cosine Wave', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('X (radians)', fontsize=12)
plt.ylabel('Y', fontsize=12)

# x축 눈금을 파이(π) 단위로 보기 쉽게 설정 (선택사항)
plt.xticks(
    [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
    ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'],
)

plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # x축 기준선
plt.legend(fontsize=11)
plt.tight_layout()

# 5. 저장 및 출력
plt.savefig('cosine_wave.png', dpi=100)
plt.show()