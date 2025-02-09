from sentence_transformers import SentenceTransformer, util
import torch

# ✅ SBERT 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# ✅ 스팸 문자 DB
spam_texts = [
    "축하드립니다! 상품이 당첨되었습니다. 링크를 클릭하세요.",
    "저금리 대출이 가능합니다. 지금 상담 신청하세요.",
    "무료 쿠폰이 발급되었습니다. 지금 다운로드하세요.",
]
spam_embeddings = model.encode(spam_texts, convert_to_tensor=True)

# ✅ 유사도 비교 함수
def check_spam(input_text):
    if not isinstance(input_text, str) or input_text.strip() == "":
        print("⚠️ 유효한 문장을 입력하세요!")
        return

    input_embedding = model.encode([input_text], convert_to_tensor=True)[0]  # 리스트로 감싸기
    similarities = util.pytorch_cos_sim(input_embedding, spam_embeddings)[0]  # 유사도 계산
    max_similarity = float(similarities.max())  # 가장 높은 유사도 값
    spam_score = max_similarity * 100  # 0~100% 변환

    print("\n📌 입력한 문장:", input_text)
    print(f"📊 스팸 의심률: {spam_score:.2f}%")

    most_similar_index = torch.argmax(similarities).item()
    print("🔍 가장 유사한 스팸 메시지:", spam_texts[most_similar_index])

# ✅ 사용자 입력 실행
if __name__ == "__main__":
    while True:
        user_input = input("\n✉️ 분석할 문장을 입력하세요 (종료하려면 'exit' 입력): ")
        if user_input.lower() == "exit":
            break
        check_spam(user_input)

