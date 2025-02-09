import joblib
from db_manager import insert_message
from sentence_transformers import SentenceTransformer

MODEL_PATH = 'models/spam_classifier.pkl'

def load_trained_model():
    """학습된 모델 불러오기"""
    classifier = joblib.load(MODEL_PATH)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model, classifier

def predict_message(message, model, classifier):
    """메시지 예측 및 스팸 여부 판단"""
    embedding = model.encode([message])
    prediction = classifier.predict(embedding)[0]
    probability = classifier.predict_proba(embedding)[0][1] * 100

    result = '스팸' if prediction == 1 else '정상 메시지'
    
    # 스팸으로 분류된 경우 데이터베이스에 저장
    if prediction == 1:
        insert_message(message, 1)
        print('스팸 메시지가 데이터베이스에 저장되었습니다.')
    
    return result, probability

