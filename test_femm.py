import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 필수 모듈 확인

try:
    import femm
except ImportError:
    print("오류: 'pyfemm' 모듈이 설치되지 않았습니다.")
    print("설치 명령: pip install pyfemm")
    sys.exit(1)

femm_path = r"C:\femm42\bin"     # 사용자 지정 FEMM 경로
femm.openfemm()
temp_file = f"temp_model_worker_0_90deg.ans"
femm.opendocument(temp_file)
femm.mo_clearblock()
femm.main_resize(1200, 1000)
femm.mo_zoomnatural()
femm.mo_showdensityplot(1, 0, 0, 0, "bmag")
femm.mo_reload()
femm.mo_savebitmap(f"torque_result.png")
print(f"[이미지 저장완료] torque_result.png ")
output_filename = "torque_result.png"
try:
    img = Image.open(output_filename)
    draw = ImageDraw.Draw(img)

    # 폰트 설정 (기본 폰트 사용 또는 시스템 폰트 경로 지정 가능)
    # Windows 기본 폰트 예시: "malgun.ttf" (맑은 고딕), 크기 20
    try:
        font = ImageFont.truetype("malgun.ttf", 20)
    except IOError:
        font = ImageFont.load_default()  # 기본 폰트 실패 시 시스템 기본값

    # 왼쪽 상단 여백 설정 (x=20, y=20)
    text_position = (20, 20)
    text_color = (0, 0, 0)  # 글자 색상 (검은색). 배경에 따라 (255, 255, 255) 흰색으로 변경 가능

    # 텍스트 배경을 살짝 보이게 하거나 깔끔하게 그리기 위해 글자 렌더링
    draw.text(text_position, temp_file, fill=text_color, font=font)

    # 수정된 이미지 덮어쓰기 저장
    img.save(output_filename)
    print(f"[이미지 및 텍스트 합성 완료] {output_filename}")

except Exception as e:
    print(
        f"[이미지 저장 완료, 텍스트 합성 실패]: {output_filename} (오류: {e})"
    )