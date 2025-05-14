"""
작성자 : kp
작성일 : 2025-05-13
목적 : PDF에서 이미지 추출 후 OCR로 텍스트 강제 추출 (문단 단위 청킹)
내용 : OCR 실패 시 예외 발생 → 시스템 중단, 문단 단위로 텍스트 분할하여 반환
"""

import os
from pdf2image import convert_from_path
import pytesseract
import re

def chunk_text_to_paragraphs(text, chunk_size=3):
    """
    OCR로 추출된 문단 리스트를 N개씩 병합하여 하나의 청크로 반환
    """
    # 기본 문단 분리
    raw_paragraphs = re.split(r'\n\s*\n|(?<=[.!?])\n', text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # N개씩 묶어서 새로운 청크 구성
    chunks = []
    for i in range(0, len(paragraphs), chunk_size):
        merged = "\n".join(paragraphs[i:i + chunk_size])
        chunks.append(merged)
    return chunks


def extract_text_and_images_from_pdf(file_path, image_output_dir="extracted_images"):
    """
    PDF 파일에서 이미지를 추출하고, OCR을 통해 텍스트를 문단 단위로 추출합니다.
    OCR 실패 시 RuntimeError 발생. 성공 시 (문단 청크 리스트, 이미지 리스트) 반환.
    """
    print(f"[INFO] PDF 처리 시작: {os.path.basename(file_path)}")
    full_text = ""
    images = []

    try:
        os.makedirs(image_output_dir, exist_ok=True)
        pil_images = convert_from_path(file_path)
        print(f"[INFO] 총 {len(pil_images)} 페이지 이미지 변환 완료")

        for idx, img in enumerate(pil_images):
            image_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_page{idx+1}.png"
            image_path = os.path.join(image_output_dir, image_name)
            img.save(image_path, "PNG")
            images.append(image_path)

            ocr_text = pytesseract.image_to_string(img, lang="kor+eng", config='--psm 6 --oem 3')
            if ocr_text.strip():
                print(f"[INFO] Page {idx+1} 텍스트 추출 성공")
                full_text += "\n" + ocr_text.strip()
            else:
                print(f"[WARNING] Page {idx+1}에서 텍스트 없음")

        if not full_text.strip():
            raise RuntimeError(f"OCR 실패: '{os.path.basename(file_path)}'에서 텍스트를 추출하지 못함.")

        paragraphs = chunk_text_to_paragraphs(full_text)
        print(f"[INFO] 문단 {len(paragraphs)}개 추출 완료")
        print(f"[INFO] 이미지 {len(images)}개 저장 완료: {image_output_dir}")

    except Exception as e:
        raise RuntimeError(f"[ERROR] PDF 처리 중 오류 발생: {e}")

    return paragraphs, images
