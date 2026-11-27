import glob
import os
import re
from PIL import Image


def create_flux_gif(
    image_folder="flux_linkage_heatmaps",
    output_gif_name="motor_flux_map_animation.gif",
    duration_ms=200,
):
    search_path = os.path.join(image_folder, "flux_theta*.png")
    image_files = glob.glob(search_path)

    if not image_files:
        print(
            f"'{os.path.abspath(image_folder)}' 경로에서 조건에 맞는 이미지 파일을 찾을 수 없습니다."
        )
        return

    # 파일명에서 숫자만 추출해 오름차순 정렬 (예: flux_theta_002.png -> 2)
    def extract_angle(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r"_theta_(\d+)", filename)
        return int(match.group(1)) if match else 0

    image_files = sorted(image_files, key=extract_angle)

    print(f"총 {len(image_files)}개의 이미지를 순서대로 불러옵니다.")
    for f in image_files:
        print(f" - {os.path.basename(f)}")

    frames = [Image.open(img_path) for img_path in image_files]

    frames[0].save(
        output_gif_name,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration_ms,
        loop=0,
    )

    print(f"\n[GIF 생성 완료] '{output_gif_name}' 파일이 성공적으로 저장되었습니다!")


if __name__ == "__main__":
    create_flux_gif(
        image_folder="flux_linkage_heatmaps",
        output_gif_name="motor_flux_map_animation.gif",
        duration_ms=200,
    )