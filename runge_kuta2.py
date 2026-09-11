import numpy as np

def rk45_adaptive_step(f, t, y, dt, tol=1e-6):
    """
    단일 단계에 대한 RK45 적응형 스텝 계산 (Fehlberg 계수 사용 예시)
    """
    # Butcher Tableau 계수 (Fehlberg 4(5))
    k1 = f(t, y)
    k2 = f(t + 1/4*dt, y + dt*(1/4*k1))
    k3 = f(t + 3/8*dt, y + dt*(3/32*k1 + 9/32*k2))
    k4 = f(t + 12/13*dt, y + dt*(1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3))
    k5 = f(t + dt, y + dt*(439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4))
    k6 = f(t + 1/2*dt, y + dt*(-8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5))
    
    # 5차 해와 4차 해의 조합
    y_5th = y + dt * (16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)
    y_4th = y + dt * (25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)
    
    # 오차 추정 (5차와 4차의 차이)
    error = np.abs(y_5th - y_4th)
    
    return y_5th, error

# 사용 예시: dy/dt = -2y + t
def f(t, y):
    return -2.0 * y + t

t, y = 0.0, 1.0
dt = 0.1
y_next, err = rk45_adaptive_step(f, t, y, dt)

print(f"다음 위치 값(y): {y_next:.5f}, 추정 오차: {err:.7f}")