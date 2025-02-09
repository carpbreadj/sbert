from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

MODEL_PATH = 'models/spam_classifier.pkl'

def train_model(df):
    """SBERT 임베딩 후 Logistic Regression 모델 학습"""
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(df['message'].tolist())

    X_train, X_test, y_train, y_test = train_test_split(embeddings, df['label'], test_size=0.2, random_state=42)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # 학습된 모델 저장
    joblib.dump(classifier, MODEL_PATH)
    print('모델 학습 완료 및 저장됨!')
    return model, classifier

