import numpy as np
from scipy.linalg import expm
import control as ct
import numpy as np
def continuous_to_discrete_zoh(A, B, C, D, Ts):
    """
    연속 시간 상태방정식 (A, B, C, D)을 ZOH 방식으로 이산화합니다.
    
    Parameters:
        A, B, C, D : numpy.ndarray (연속 시스템 행렬)
        Ts : float (샘플링 주기)
        
    Returns:
        Ad, Bd, Cd, Dd : numpy.ndarray (이산 시스템 행렬)
    """
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    
    # 1. 확장 행렬(Augmented Matrix) 구성
    # [ A   B ] * Ts
    # [ 0   0 ]
    m = np.block([
        [A * Ts, B * Ts],
        [np.zeros((n_inputs, n_states)), np.zeros((n_inputs, n_inputs))]
    ])
    print("\nm =\n", m)
    # 2. 행렬 지수함수(Matrix Exponential) 계산
    m_exp = expm(m)
    mA_exp = expm(A * Ts)
    B2 = np.linalg.inv(A)@(mA_exp-np.eye(2,2))@B
    print("\nB2 =\n", B2)
    print("\nm_exp =\n", m_exp)
    print("\nmA_exp =\n", mA_exp)
 
    # 3. 결과 행렬 추출
    Ad = m_exp[:n_states, :n_states]
    Bd = m_exp[:n_states, n_states:]
    Cd = C.copy()
    Dd = D.copy()
    
    return Ad, Bd, Cd, Dd

# --- [예제 실행] ---
# 연속 시스템 행렬 정의 (예: 간단한 2차 시스템)
A = np.array([[0.0, 1.0], 
              [-2.0, -3.0]])
B = np.array([[0.0], 
              [1.0]])
C = np.array([[1.0, 0.0]])
D = np.array([[0.0]])

Ts = 0.1  # 샘플링 주기 (0.1초)

# ZOH 이산화 수행
Ad, Bd, Cd, Dd = continuous_to_discrete_zoh(A, B, C, D, Ts)

print("--- 이산 시스템 행렬 (ZOH) ---")
print("Ad =\n", Ad)
print("\nBd =\n", Bd)
print("\nCd =\n", Cd)
print("\nDd =\n", Dd)

sys_ss_c = ct.ss(A, B, C, D)  # 연속 상태공간 모델
sys_ss_d = ct.ss(Ad, Bd, Cd, Dd, Ts)    
print("--- 이산시간 전달함수 ---")
print(sys_ss_d)