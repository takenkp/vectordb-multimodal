"""
작성자 : kp
작성일 : 2025-05-13
목적 : 코사인 유사도 기반 벡터 검색 구현
내용 : 입력된 쿼리를 임베딩 후, Chroma DB에서 가장 유사한 문서 검색
"""

import chromadb
from chromadb.utils import embedding_functions

# Chroma 설정
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="multimodal-documents",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

def search_similar_documents(query, top_k=5):
    """사용자 쿼리로 Chroma에서 유사 문서 검색"""
    print(f"\n검색어로 유사 문서 검색: '{query}' (최대 {top_k}개)\n")

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    for i, doc in enumerate(results['documents'][0]):
        score = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        doc_id = results['ids'][0][i]

        print(f"[{i+1}] ID: {doc_id}")
        print(f"    유형: {metadata.get('유형')}, 파일명: {metadata.get('파일명')}")
        print(f"    유사도: {1 - score:.4f} (높을수록 유사)")  # 코사인 유사도는 1 - distance
        preview = doc.strip().replace("\n", " ")
        print(f"    내용: {preview[:100]}...\n")  # 앞부분 요약

if __name__ == "__main__":
    while True:
        query = input("검색어를 입력하세요 (종료하려면 'exit'): ")
        if query.lower() == "exit":
            break
        search_similar_documents(query)
