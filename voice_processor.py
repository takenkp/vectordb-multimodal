"""
작성자 : kp
작성일 : 2025-05-13
목적 : 음성 파일에서 STT 텍스트 추출 및 JSON 저장
내용 : faster_whisper 기반 STT + voice_text.json 자동 저장
"""

import os
import json
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf
import tempfile

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

def extract_text_from_voice(file_path, language="ko"):
    """음성 파일에서 텍스트(STT)를 추출하고 JSON 저장"""
    file_name = os.path.basename(file_path)
    print(f"음성 파일에서 텍스트 추출 시도: {file_name}")
    try:
        # mp3, m4a 등은 wav로 변환
        if not file_path.lower().endswith(".wav"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                data, samplerate = sf.read(file_path)
                sf.write(tmpfile.name, data, samplerate)
                wav_path = tmpfile.name
        else:
            wav_path = file_path

        segments, _ = model.transcribe(wav_path, language=language, beam_size=5, condition_on_previous_text=False)
        texts = [f"[{seg.start:.2f} → {seg.end:.2f}] {seg.text}" for seg in segments]

        if wav_path != file_path and os.path.exists(wav_path):
            os.remove(wav_path)

        if texts:
            joined_text = "\n".join(texts)
            print(f"텍스트 추출 완료: {len(texts)}개 세그먼트")
            save_transcript_to_json(file_name, joined_text)
            return joined_text
        else:
            print("텍스트를 추출하지 못했습니다.")
            return "No text extracted from voice file."

    except Exception as e:
        print(f"음성 파일 처리 중 오류 발생: {e}")
        return f"Error processing voice file {file_name}: {e}"

def extract_info_from_voice(file_path):
    """음성 파일에서 기본 정보 추출"""
    file_name = os.path.basename(file_path)
    print(f"음성 파일에서 정보 추출 시도: {file_name}")
    return f"Extracted voice info for: {file_name}"
