import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. CSV 파일 불러오기
file_path = 'spam_url_data.csv'
data = pd.read_csv(file_path, encoding='utf-8-sig')  # 한글 인코딩 처리

# 2. 컬럼 이름 공백 제거 및 확인
data.columns = data.columns.str.strip()
print(data.columns)

# 3. 'label' 컬럼이 없으면 추가
if 'label' not in data.columns:
    data['label'] = 1  # 기본값으로 스팸(1) 설정

# 4. 정상 URL 데이터 추가
normal_urls = [
    'http://google.com', 'http://naver.com', 'https://openai.com',
    'http://github.com', 'http://stackoverflow.com'
]
normal_data = pd.DataFrame({'URL주소': normal_urls, 'label': 0})
data = pd.concat([data, normal_data], ignore_index=True)

# 5. SBERT 모델 불러오기 및 임베딩
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(data['URL주소'].tolist(), convert_to_tensor=True)

# 6. 데이터 분할
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

# 7. 로지스틱 회귀 모델 학습
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 8. 예측 및 평가
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))