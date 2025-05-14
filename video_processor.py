"""
작성자 : kp
작성일 : 2025-05-13
목적 : 영상 파일에서 오디오 추출 후 STT 텍스트 추출 및 저장
내용 : ffmpeg로 mp4 → wav 추출 후 faster_whisper로 STT 처리, voice_text.json 저장
"""

import os
import json
import tempfile
import subprocess
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("large-v3-turbo", compute_type="float32")
voice_text_json_path = "voice_text.json"

def save_transcript_to_json(file_name, transcript, json_path=voice_text_json_path):
    """STT 텍스트를 JSON 파일에 저장"""
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            db = json.load(f)
    else:
        db = {}

    db[file_name] = transcript

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def extract_text_from_video(file_path, language="ko"):
    """영상(mp4)에서 오디오 추출 후 텍스트 추출 및 저장"""
    file_name = os.path.basename(file_path)
    print(f"영상 파일에서 텍스트 추출 시도: {file_name}")

    try:
        # mp4 → wav 임시파일로 추출
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            audio_path = tmpfile.name

        cmd = [
            "ffmpeg", "-y",
            "-i", file_path,
            "-vn",  # 영상 제외
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # faster-whisper STT 수행
        segments, _ = model.transcribe(audio_path, language=language, beam_size=5, condition_on_previous_text=False)
        texts = [f"[{seg.start:.2f} → {seg.end:.2f}] {seg.text}" for seg in segments]

        if os.path.exists(audio_path):
            os.remove(audio_path)

        if texts:
            joined_text = "\n".join(texts)
            print(f"텍스트 추출 완료: {len(texts)}개 세그먼트")
            save_transcript_to_json(file_name, joined_text)
            return joined_text
        else:
            print("텍스트를 추출하지 못했습니다.")
            return "No text extracted from video file."

    except Exception as e:
        print(f"영상 파일 처리 중 오류 발생: {e}")
        return f"Error processing video file {file_name}: {e}"

def extract_info_from_video(file_path):
    """기본 정보 추출용 함수"""
    file_name = os.path.basename(file_path)
    print(f"영상 파일에서 정보 추출 시도: {file_name}")
    return f"Extracted media info for video: {file_name}"
