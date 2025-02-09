from db_manager import initialize_db, load_messages
from model_trainer import train_model
from predictor import load_trained_model, predict_message

# 1. 데이터베이스 초기화
initialize_db()

# 2. 데이터 불러오기 및 모델 학습
df = load_messages()
if not df.empty:
    model, classifier = train_model(df)
else:
    print('데이터베이스에 학습할 데이터가 없습니다.')

# 3. 사용자 입력 메시지 예측
while True:
    user_input = input('메시지를 입력하세요 (종료하려면 "exit" 입력): ')
    if user_input.lower() == 'exit':
        break

    result, probability = predict_message(user_input, model, classifier)
    print(f'예측 결과: {result} (스팸 확률: {probability:.2f}%)\n')

