import numpy as np
from scipy.interpolate import interp1d

# 1. 원본 데이터 정의 (x와 y)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 5, 4, 8])

# 2. interp1d 객체 생성 (기본값은 'linear')
f = interp1d(x, y)

# 3. 원본 데이터 범위 내의 새로운 x 좌표에서 보간 값 계산
xnew = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
ynew = f(xnew)

print("새로운 x:", xnew)
print("보간된 y:", ynew)