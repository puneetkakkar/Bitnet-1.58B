from flask import Flask, request, jsonify
from predict import predict as predict_function  # Assuming predict function is defined in predict.py

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_handler():
    if request.method == 'POST':
        data = request.json
        input_prompt = data.get('user_input')
        
        processed_input = input_prompt.lower()

        prediction = predict_function(processed_input)

        formatted_predicted = {'text': prediction}

        return jsonify(formatted_predicted)

if __name__ == '__main__':
    app.run(debug=True)