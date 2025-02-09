from sentence_transformers import SentenceTransformer, util
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# SBERT 모델 로드
model = SentenceTransformer("all-MiniLM-L6-v2")  # 가벼운 모델 추천

# 가짜 스팸 데이터베이스 (DB 연동 전 테스트용)
spam_texts = [
    "당첨을 축하드립니다! 지금 링크를 눌러 경품을 수령하세요.",
    "저금리 대출이 가능합니다. 한도 조회 무료!",
    "무료 쿠폰이 발급되었습니다. 링크 클릭 후 사용하세요."
]
spam_embeddings = model.encode(spam_texts, convert_to_tensor=True)

# FastAPI 앱 생성
app = FastAPI()

# 요청 데이터 형식 정의
class TextInput(BaseModel):
    text: str

# 유사도 계산 함수
def check_spam(input_text):
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(input_embedding, spam_embeddings)[0]
    max_similarity = float(similarities.max())  # 가장 높은 유사도

    # 스팸 의심률 계산 (기본적으로 100% 변환)
    spam_score = max_similarity * 100
    return spam_score

# API 엔드포인트 정의
@app.post("/check_spam")
async def check_spam_api(input_data: TextInput):
    score = check_spam(input_data.text)
    return {"spam_score": score}
