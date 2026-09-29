import os
import glob
from PIL import Image

def create_flux_gif(image_folder=".", output_gif_name="flux_animation.gif", duration_ms=200):
    # 폴더 내의 지정된 패턴 이미지 파일들 검색
    # (예: flux_dist_theta_*.jpg 파일들을 모두 불러옴)
    search_path = os.path.join(image_folder, "flux_dist_theta_*")
    image_files = glob.glob(search_path)
    
    if not image_files:
        print("조건에 맞는 이미지 파일을 찾을 수 없습니다.")
        return

    # 파일 이름에 포함된 각도(theta) 숫자 기준으로 정확하게 오름차순 정렬
    def extract_angle(filepath):
        filename = os.path.basename(filepath)
        # 예: 'flux_dist_theta_224deg.jpg' 에서 숫자만 추출
        try:
            num_str = filename.split('_theta_')[1].split('deg')[0]
            return int(num_str)
        except:
            return 0

    image_files = sorted(image_files, key=extract_angle)
    
    print(f"총 {len(image_files)}개의 이미지를 순서대로 불러옵니다.")
    for f in image_files:
        print(f" - {os.path.basename(f)}")

    # 이미지 열기 및 리스트에 담기
    frames = [Image.open(img_path) for img_path in image_files]

    # 첫 번째 프레임을 기준으로 GIF 저장 (duration: 프레임당 유지 시간(ms), loop: 0은 무한 반복)
    frames[0].save(
        output_gif_name,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration_ms,  # 프레임 간격 (200ms)
        loop=0                 # 무한 반복
    )

    print(f"\n[GIF 생성 완료] '{output_gif_name}' 파일이 성공적으로 저장되었습니다!")

if __name__ == "__main__":
    
    # 현재 디렉토리에 있는 이미지를 대상으로 200ms 간격의 GIF 생성
    create_flux_gif(image_folder="./ans_images_300A", output_gif_name="motor_flux_animation300A.gif", duration_ms=200)