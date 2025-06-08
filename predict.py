import sys
import json
import pickle
import numpy as np

# Kiểm tra xem có đối số nào được truyền vào không
print("Arguments passed to Python:", sys.argv) 
if len(sys.argv) < 2:
    # Nếu thiếu dữ liệu đầu vào, in thông báo lỗi mà không thoát chương trình
    print(json.dumps({"error": "Thieu du lieu dau vao"}))
else:
    # Tải mô hình từ file
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Nhận dữ liệu từ Java (dưới dạng JSON)
    input_data = json.loads(sys.argv[1])
     # In ra các đối số nhận được
    # Trích xuất dữ liệu từ input
    N = input_data['Nitrogen']
    P = input_data['Phosphorus']
    K = input_data['Potassium']
    temperature = input_data['Temperature']
    humidity = input_data['Humidity']
    ph = input_data['Ph']
    rainfall = input_data['Rainfall']

    # Tạo mảng dữ liệu đầu vào
    input_array = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Dự đoán
    prediction = model.predict(input_array)

    # In kết quả dự đoán về cho Java
    print(json.dumps({"result": prediction[0]}))
