import math
import os
import shutil
import sys
import matplotlib.pyplot as plt
import numpy as np

# FEMM 모듈 확인
femm_available = False
try:
    import femm
    femm_available = True
except ImportError:
    print("경고: FEMM 모듈이 설치되지 않았습니다.")
    print("설치 명령: pip install femm")
    print("테스트 데이터로 계속 진행합니다.\n")


def run_simulation_at_angle(angle_deg, base_file_path, worker_id):
    """개별 회전각에 대해 시뮬레이션을 수행하여 토크를 계산하는 함수"""
    
    print(f"\n[{angle_deg:2d}도] 시뮬레이션 시작")
    
    if not femm_available:
        # 테스트 데이터: 간단한 사인 함수 토크값
        torque = 100 * math.sin(math.radians(angle_deg + 90))
        print(f"[{angle_deg:2d}도] 토크: {torque:.4f} Nm (테스트 데이터)")
        return angle_deg, torque
    
    # FEMM을 사용하는 경우
    temp_file = f"temp_model_{worker_id}.fem"
    
    try:
        # 파일 복사
        try:
            shutil.copy(base_file_path, temp_file)
            print(f"[{angle_deg:2d}도] 파일 복사: {temp_file}")
        except Exception as e:
            print(f"[{angle_deg:2d}도] 오류: 파일 복사 실패 - {e}")
            return angle_deg, 0.0
        
        # FEMM 열기
        print(f"[{angle_deg:2d}도] FEMM 시작 중...", end="", flush=True)
        femm.openfemm()
        print(" 완료")
        
        # 문서 열기
        print(f"[{angle_deg:2d}도] 문서 로드 중...", end="", flush=True)
        femm.opendocument(temp_file)
        print(" 완료")
        
        # 3상 전류 계산
        I_max = 215.0
        theta_rad = math.radians(angle_deg)
        i_a = I_max * math.sin(theta_rad)
        i_b = I_max * math.sin(theta_rad - math.radians(120))
        i_c = I_max * math.sin(theta_rad + math.radians(120))
        
        print(f"[{angle_deg:2d}도] 회로 속성 설정...", end="", flush=True)
        circuits = [("A_phase", i_a), ("B_phase", i_b), ("C_phase", i_c)]
        for circuit_name, current in circuits:
            try:
                femm.mi_modifycircprop(circuit_name, 1, current)
            except Exception as e:
                print(f"\n[{angle_deg:2d}도] 경고: 회로 '{circuit_name}' 설정 실패 - {e}")
        print(" 완료")
        
        # 분석 실행
        print(f"[{angle_deg:2d}도] 분석 실행 중...", end="", flush=True)
        femm.mi_analyze(1)
        print(" 완료")
        
        # 해석 결과 로드
        print(f"[{angle_deg:2d}도] 해석 결과 로드 중...", end="", flush=True)
        femm.mi_loadsolution()
        print(" 완료")
        
        # 토크 계산
        print(f"[{angle_deg:2d}도] 토크 계산 중...", end="", flush=True)
        femm.mo_clearselected()
        for group_id in [1, 4]:
            try:
                femm.mo_groupselectblock(group_id)
            except Exception as e:
                print(f"\n[{angle_deg:2d}도] 경고: 그룹 {group_id} 선택 실패 - {e}")
        
        torque = femm.mo_blockintegral(10)
        print(f" {torque:.4f} Nm")
        
        # FEMM 종료
        femm.closefemm()
        
        return angle_deg, torque
        
    except Exception as e:
        print(f"\n[{angle_deg:2d}도] 오류: {e}")
        try:
            femm.closefemm()
        except:
            pass
        return angle_deg, 0.0
    
    finally:
        # 임시 파일 정리
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"[{angle_deg:2d}도] 임시 파일 삭제: {temp_file}")
        except Exception as e:
            print(f"[{angle_deg:2d}도] 경고: 임시 파일 삭제 실패 - {e}")
        
        try:
            analyzed_file = temp_file.replace(".fem", ".ans")
            if os.path.exists(analyzed_file):
                os.remove(analyzed_file)
        except:
            pass


def main():
    base_file_path = "ioniq5-6.FEM"
    
    # 기본 파일 존재 확인
    if not os.path.exists(base_file_path):
        print(f"오류: {base_file_path} 파일을 찾을 수 없습니다.")
        print(f"현재 디렉토리: {os.getcwd()}")
        
        # 사용 가능한 FEM 파일 표시
        fem_files = [f for f in os.listdir() if f.upper().endswith('.FEM')]
        if fem_files:
            print("사용 가능한 FEM 파일:")
            for fem_file in fem_files:
                print(f"  - {fem_file}")
            return
        else:
            print("FEM 파일을 찾을 수 없습니다.")
            return
    
    print(f"시뮬레이션 시작")
    print(f"기본 파일: {base_file_path}")
    print(f"FEMM 사용: {'Yes' if femm_available else 'No (테스트 데이터 사용)'}")
    print(f"{'='*60}\n")
    
    # 0도부터 90도까지 10도 간격
    angles = list(range(0, 91, 10))
    
    results = []
    for i, angle in enumerate(angles):
        result = run_simulation_at_angle(angle, base_file_path, i)
        results.append(result)
    
    # 결과 정렬
    results.sort(key=lambda x: x[0])
    sorted_angles = [r[0] for r in results]
    torques = [r[1] for r in results]
    
    # 결과 출력
    print(f"\n{'='*60}")
    print("시뮬레이션 완료")
    print(f"{'='*60}\n")
    print(f"{'각도(도)':>10} | {'토크(Nm)':>12}")
    print(f"{'-'*25}")
    
    for ang, tq in zip(sorted_angles, torques):
        print(f"{ang:>10d} | {tq:>12.4f}")
    
    # 그래프 생성
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_angles, torques, marker="o", linestyle="-", 
                color="b", linewidth=2, markersize=6)
        plt.title("Electrical Angle vs Electromagnetic Torque", fontsize=14)
        plt.xlabel("Electrical Angle [deg]", fontsize=12)
        plt.ylabel("Torque [Nm]", fontsize=12)
        plt.grid(True, which="both", linestyle="--", alpha=0.7)
        plt.xticks(range(0, 91, 10))
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\n경고: 그래프 생성 실패 - {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n예기치 않은 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
