from flask import Flask
from flask_cors import CORS
from 기대_수명_예측 import life_expectancy

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])

app.add_url_rule('/python/life_expectancy', 'life_expectancy', life_expectancy, methods=['POST'])

if __name__ == '__main__':
	app.run(debug=True)
