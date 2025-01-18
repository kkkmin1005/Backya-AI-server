from flask import Flask, request, jsonify
import os
from io import BytesIO
from PIL import Image
from model.StableDiffusion import GenerateAndSaveImg  # 새로 작성한 함수
from model.StyleTransfer import Cartoonify
import boto3
from datetime import datetime  # 시간 기반 타임스탬프 추가
import uuid  # UUID 사용
import openai  # OpenAI API 사용
import config

# AWS S3 설정
AWS_ACCESS_KEY = config.AWS_ACCESS_KEY
AWS_SECRET_KEY = config.AWS_SECRET_KEY
S3_BUCKET_NAME = config.S3_BUCKET_NAME

# OpenAI API 설정
openai.api_key = config.OPENAI_API_KEY

# S3 클라이언트 생성
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

app = Flask(__name__)

print("Flask app initialized.")  # 로그 추가


def summarize_prompt(prompt):
    """GPT를 사용해 프롬프트를 영어로 1줄 요약."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 최신 모델로 변경
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes prompts into a single concise English sentence."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.7,
        )
        summary = response["choices"][0]["message"]["content"].strip()
        print(f"Prompt summarized to: {summary}")  # 로그 추가
        return summary
    except Exception as e:
        print("Error summarizing prompt:", e)  # 에러 로그 추가
        raise Exception("Failed to summarize the prompt using GPT.")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        # 사용자 입력 받기
        data = request.get_json()
        prompt = data.get("prompt")

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        print("Received prompt:", prompt)  # 로그 추가

        # 프롬프트 요약
        summarized_prompt = summarize_prompt(prompt)
        final_prompt = f"{summarized_prompt}. Generate this as a cartoon-style image."

        # 이미지 생성 및 저장
        saved_path = GenerateAndSaveImg(final_prompt, output_dir="output", file_name="generated_image.png")
        print(f"Image saved at: {saved_path}")  # 로그 추가

        # 저장된 이미지를 열기
        with Image.open(saved_path) as img:
            print("Image loaded for cartoon effect.")  # 로그 추가

            # 카툰화 처리
            cartoon_image = Cartoonify(img)
            print("Cartoon effect applied.")  # 로그 추가

        # S3에 저장할 고유 파일 이름 생성
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # 현재 시간
        unique_id = str(uuid.uuid4())  # UUID 생성
        s3_key = f"cartoonified_images/{timestamp}_{unique_id}.png"  # S3에 저장될 키
        print(f"Generated S3 key: {s3_key}")  # 로그 추가

        # 카툰화된 이미지를 S3에 저장
        img_io = BytesIO()
        cartoon_image.save(img_io, "PNG")
        img_io.seek(0)

        # S3 업로드
        s3_client.upload_fileobj(img_io, S3_BUCKET_NAME, s3_key, ExtraArgs={"ContentType": "image/png"})
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
        print(f"Image uploaded to S3: {s3_url}")  # 로그 추가

        return jsonify({"message": "Image successfully uploaded to S3", "s3_url": s3_url})

    except Exception as e:
        print("Error occurred:", e)  # 에러 로그 추가
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask server...")  # 로그 추가
    os.makedirs("output", exist_ok=True)  # output 폴더 생성
    app.run(host="0.0.0.0", port=5000, debug=True)
