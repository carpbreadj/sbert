from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# 1. 데이터 수집 및 전처리
# 확장된 한글 SMS 스팸 데이터
messages = [
    '무료로 응모하세요! FA컵 결승전 티켓을 받을 기회를 잡으세요. 지금 문자 보내기!',
    '너무 일찍 말하지 마... 이미 봤으면 말해...',
    '축하합니다!! 귀하는 네트워크 고객으로서 90만원 상금에 당첨되었습니다!',
    '핸드폰 사용한 지 11개월 넘으셨나요? 최신 컬러 휴대폰으로 무료 업그레이드 기회!',
    '나 금방 집에 갈 거야. 오늘 밤은 이 얘기 더 이상 하고 싶지 않아. 나 오늘 너무 많이 울었어.',
    '현금 당첨의 기회! 10만원에서 2000만원까지 즉시 응모하세요!',
    '나 일요일에 윌이랑 데이트 있어!!',
    '지금 바로 클릭하고 최신 스마트폰을 무료로 받아가세요!',
    '오늘 저녁에 뭐 먹을까?',
    '은행 계좌 정보가 업데이트되었습니다. 즉시 확인하세요.',
    '오랜만이야! 이번 주말에 시간 있어?',
    '당신의 신용카드가 정지되었습니다. 즉시 고객센터로 연락하세요.',
    '내일 아침 9시에 회의 있는 거 잊지 마.',
    '500만원 당첨! 지금 바로 연락 주세요!',
    '오늘 날씨 정말 좋다! 산책 갈래?',
    '회원님, 특별 할인 쿠폰이 도착했습니다. 지금 확인하세요!',
    '내일 시험 준비는 잘하고 있어?',
    '이벤트에 당첨되셨습니다! 무료 여행 기회를 잡으세요!',
    '오늘 저녁 7시에 영화 보러 갈래?',
    '긴급! 계정이 해킹되었습니다. 비밀번호를 즉시 변경하세요!'
]

labels = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1은 스팸, 0은 정상 메시지

# 데이터프레임 생성
df = pd.DataFrame({'message': messages, 'label': labels})

# 2. SBERT 모델 불러오기
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 3. 임베딩 생성
embeddings = model.encode(df['message'].tolist())

# 4. 분류 모델 학습
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(embeddings, df['label'], test_size=0.2, random_state=42)

# Logistic Regression 분류기 학습
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 5. 모델 평가
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. 새로운 입력 테스트 및 학습 (유사도 포함)

def predict_message(message):
    embedding = model.encode([message])
    prediction = classifier.predict(embedding)
    probability = classifier.predict_proba(embedding)[0]
    
    # 스팸일 확률 퍼센트로 변환
    spam_probability = probability[1] * 100
    
    result = '스팸' if prediction[0] == 1 else '정상 메시지'
    return result, spam_probability, embedding

# 7. 새로운 스팸 메시지 학습 함수
def retrain_with_spam(new_spam_messages):
    global df, classifier, model

    # 중복 메시지 제거
    existing_messages = df['message'].tolist()
    unique_new_spam = [msg for msg in new_spam_messages if msg not in existing_messages]

    if not unique_new_spam:
        print("새로운 스팸 메시지가 없습니다.")
        return

    # 새로운 데이터 추가
    new_data = pd.DataFrame({'message': unique_new_spam, 'label': [1] * len(unique_new_spam)})
    df = pd.concat([df, new_data], ignore_index=True)

    # 새로운 임베딩 생성 후 재학습
    embeddings = model.encode(df['message'].tolist())
    X_train, X_test, y_train, y_test = train_test_split(embeddings, df['label'], test_size=0.2, random_state=42)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    print("모델이 새로운 스팸 메시지로 재학습되었습니다.")

# 테스트 메시지
test_messages = [
    '축하합니다! 당신은 100만원 상당의 상품권에 당첨되었습니다. 지금 링크를 클릭하세요.',
    '안녕, 오늘 저녁에 만나기로 한 거 잊지 않았지?',
    '긴급! 귀하의 계정이 해킹되었습니다. 지금 바로 정보를 확인하세요.',
    '10분 후에 도착할게, 곧 봐!',
    '무료 영화 티켓을 받으세요! 지금 바로 신청하세요!',
    '오늘 점심에 뭐 먹을지 고민이야. 추천해줘!'
]

# 테스트 결과 출력 및 새로운 스팸 메시지 재학습
new_spam_messages = []
for msg in test_messages:
    result, probability, embedding = predict_message(msg)
    print(f'메시지: "{msg}" \n예측 결과: {result} (유사도: {probability:.2f}%)\n')
    
    # 스팸으로 예측된 메시지를 새로운 학습 데이터로 추가
    if result == '스팸':
        new_spam_messages.append(msg)

# 새로운 스팸 메시지로 모델 재학습
retrain_with_spam(new_spam_messages)

# 재학습 후 테스트
print("\n재학습 후 테스트 결과:")
for msg in test_messages:
    result, probability, _ = predict_message(msg)
    print(f'메시지: "{msg}" \n예측 결과: {result} (유사도: {probability:.2f}%)\n')
