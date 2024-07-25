import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# GPU 사용 가능 여부 확인
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

MODEL = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
)

FILE_PATH = "/Users/ki/Desktop/KEB/Real-ESRGAN/RealESRGAN_x4plus.pth"
upsampler = RealESRGANer(
    scale=4,
    model_path=FILE_PATH,
    dni_weight=0.5,
    model=MODEL,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True,  # GPU 사용 시 half precision
    gpu_id=0 if device == "cuda" else None,
    device=device,
)


def enhance_image(input_path, output_path, upsampler, outscale=4):
    """Enhance the image using Real-ESRGAN model."""
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)

    try:
        output, _ = upsampler.enhance(img_np, outscale=outscale)
        output_img = Image.fromarray(output)
    except RuntimeError as error:
        print(f"Error processing {input_path}: {error}")
        print(
            "If you encounter CUDA out of memory, try to set --tile with a smaller number."
        )
        return

    # 원본 파일 이름과 확장자 유지
    output_name = os.path.basename(input_path)
    save_path = os.path.join(output_path, output_name)

    # PIL을 사용하여 이미지 저장
    output_img.save(save_path)
    print(f"Saved enhanced image to: {save_path}")


input_dir = "/Users/ki/Desktop/KEB/train"
output_dir = "/Users/ki/Desktop/KEB/r-esrgan_images"

# 출력 디렉토리가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 모든 이미지 파일 경로 가져오기
image_paths = glob.glob(os.path.join(input_dir, "*.*"))
print(f"Found {len(image_paths)} images to process")

# tqdm으로 진행 상황 표시
for input_image_path in tqdm(image_paths, desc="Enhancing images", unit="image"):
    enhance_image(input_image_path, output_dir, upsampler)

print("Image enhancement completed.")
