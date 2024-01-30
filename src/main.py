from flask import Flask, request, jsonify
from flask_cors import CORS
import 기대_수명_예측

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])

# 아래 경로로 요청이 들어올때 해당 함수를 실행
@app.route('/predict_life_expectancy', methods=['POST'])
def predict_life_expectancy():
    # 리액트로부터 받은 데이터 추출
    data = request.json
    year = data['Year']
    bmi = data['BMI']
    alcohol = data['Alcohol']
    country = data['Country']

    print("리액트로부터 받은 기대 수명 예측 데이터 : " + str(data))

    # 예측 모델에 데이터 전달
    prediction = 기대_수명_예측.predict_life_expectancy(year, bmi, alcohol, country)

    # 예측 결과를 JSON 형태로 반환
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
