from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/predict": {"origins": "*"}})

with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return "Flask server is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        N = float(data['Nitrogen'])
        P = float(data['Phosphorus'])
        K = float(data['Potassium'])
        temperature = float(data['Temperature'])
        humidity = float(data['Humidity'])
        ph = float(data['Ph'])
        rainfall = float(data['Rainfall'])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # ✅ Nếu bạn dùng chuẩn hóa, hãy mở dòng này
        # input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)

        # Ép kiểu về int để tránh lỗi jsonify với numpy.int64
        return jsonify({"result": int(prediction[0])})

    except KeyError as e:
        return jsonify({"error": f"Thiếu trường dữ liệu: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Lỗi không xác định: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
