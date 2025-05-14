"""
작성자 : kp
작성일 : 2025-05-13
목적 : PDF, 음성, 영상 파일에서 텍스트 추출 후 Chroma DB에 업로드 및 유사도 검색
내용 : OCR 및 STT 텍스트를 '내용' 필드로 벡터화하여 저장 후, 코사인 유사도 기반 검색 수행
"""

import os
from pdf_processor import extract_text_and_images_from_pdf
from voice_processor import extract_text_from_voice, extract_info_from_voice
from video_processor import extract_text_from_video, extract_info_from_video
import re
import chromadb
from chromadb.utils import embedding_functions

# Chroma 설정
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="multimodal-documents",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

def process_file_and_upload(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)

    try:
        if ext == ".pdf":
            print(f"[INFO] PDF 처리 시작: {file_name}")
            paragraphs, images = extract_text_and_images_from_pdf(file_path)

            for i, para in enumerate(paragraphs):
                clean_text = para.strip()

                # ① 길이 필터링
                if len(clean_text) < 30:
                    continue

                # ② 의미 없는 반복 문자 필터링
                if re.match(r"^(k|ㅎ|\.|\s){3,}$", clean_text, re.IGNORECASE):
                    continue

                para_id = f"{file_path}#chunk_{i+1}"
                meta = {
                    "유형": "pdf",
                    "파일명": file_name,
                    "청크": i + 1
                }

                collection.add(
                    documents=[clean_text],
                    metadatas=[meta],
                    ids=[para_id]
                )
                print("clean_text는 아래와 같습니다. 이는 아래 내역에서 확인가능합니다.")
                print(clean_text)
                print(f"[UPLOAD] '{file_name}' → 청크 {i+1} 업로드 완료")

        elif ext == ".mp3":
            print(f"[INFO] 음성 처리 시작: {file_name}")
            text = extract_text_from_voice(file_path)
            meta = {
                "유형": "voice",
                "파일명": file_name,
                "내용": extract_info_from_voice(file_path)
            }

        elif ext == ".mp4":
            print(f"[INFO] 영상 처리 시작: {file_name}")
            text = extract_text_from_video(file_path)
            meta = {
                "유형": "video",
                "파일명": file_name,
                "내용": extract_info_from_video(file_path)
            }

        else:
            print(f"[SKIP] 지원하지 않는 파일 형식: {file_name}")
            return

        if not text or not text.strip():
            raise RuntimeError(f"[ERROR] 텍스트 추출 실패: {file_name}")

        collection.add(
            documents=[text],
            metadatas=[meta],
            ids=[file_path]
        )
        print(f"[UPLOAD] '{file_name}' → 벡터 DB에 업로드 완료")

    except Exception as e:
        print(f"[FAIL] 파일 처리 실패: {file_name} - {e}")

def collect_files_from_directories(base_dirs):
    """지정된 디렉토리 내 모든 유효 파일 수집"""
    valid_exts = {".pdf", ".mp3", ".mp4"}
    files = []

    for base_dir in base_dirs:
        for root, _, filenames in os.walk(base_dir):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in valid_exts:
                    files.append(os.path.join(root, fname))
    return files

def search_similar_documents(query, top_k=5):
    """사용자 쿼리로 Chroma에서 유사 문서 검색"""
    print(f"\n🔍 검색어: '{query}' → 유사 문서 {top_k}건\n")

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    if not results['documents'] or not results['documents'][0]:
        print("❌ 유사한 문서를 찾지 못했습니다.\n")
        return

    for i, doc in enumerate(results['documents'][0]):
        distance = results['distances'][0][i]
        similarity = distance  # ✅ 유사도 음수 방지

        metadata = results['metadatas'][0][i]
        doc_id = results['ids'][0][i]

        print(f"[{i+1}] ID: {doc_id}")
        print(f"    유형: {metadata.get('유형')}, 파일명: {metadata.get('파일명')}")
        print(f"    유사도: {similarity:.4f} (높을수록 유사)")
        print(f"    내용: {doc}...\n")


if __name__ == "__main__":
    # 1단계: 파일 수집 및 업로드
    target_dirs = ["./pdf", "./voice", "./video"]
    all_files = collect_files_from_directories(target_dirs)

    print(f"\n총 {len(all_files)}개 파일 처리 시작...\n")
    for f in all_files:
        process_file_and_upload(f)

    # 2단계: 검색 루프
    print("\n==============================")
    print("✅ 모든 문서 업로드 완료. 검색 모드로 진입합니다.")
    print("검색어를 입력하세요 (종료하려면 'exit'):")
    print("==============================\n")

    while True:
        query = input("🔎 검색어: ")
        if query.lower() == "exit":
            print("👋 종료합니다.")
            break
        search_similar_documents(query)
