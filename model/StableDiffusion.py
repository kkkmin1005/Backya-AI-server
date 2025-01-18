import openai
import requests
from PIL import Image
from io import BytesIO
import os
import config

# OpenAI API 키 설정
openai.api_key = config.OPENAI_API_KEY

def GenerateAndSaveImg(prompt, output_dir="output", file_name="generated_image.png"):
    """
    OpenAI DALL·E API를 사용하여 텍스트 프롬프트로 이미지를 생성하고,
    지정된 폴더에 저장한 뒤 경로를 반환.
    """
    try:
        print(f"Generating image for prompt: {prompt}")

        # OpenAI DALL·E API 요청
        response = openai.Image.create(
            prompt=prompt,
            n=1,  # 생성할 이미지 개수
            size="512x512",  # 이미지 크기
        )

        # API 응답에서 이미지 URL 추출
        if "data" in response and len(response["data"]) > 0:
            image_url = response["data"][0]["url"]
            print("Image URL:", image_url)

            # 이미지 다운로드
            img_response = requests.get(image_url)
            if img_response.status_code == 200:
                # output 폴더 생성
                os.makedirs(output_dir, exist_ok=True)

                # 이미지 저장 경로
                file_path = os.path.join(output_dir, file_name)

                # 이미지 저장
                image = Image.open(BytesIO(img_response.content))
                image.save(file_path)
                print(f"Image saved at: {file_path}")
                return file_path
            else:
                raise Exception(f"Failed to download image from URL: {image_url}")
        else:
            raise Exception("No image URL returned from OpenAI API")

    except Exception as e:
        print(f"Error during image generation: {e}")
        raise