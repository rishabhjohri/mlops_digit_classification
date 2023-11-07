from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from your_model import predict_digit  # This would be your own function to predict the digit

app = Flask(__name__)

@app.route('/compare-digits', methods=['POST'])
def compare_digits():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify(error='Missing images'), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    # Save or process images as needed
    filename1 = secure_filename(image1.filename)
    filename2 = secure_filename(image2.filename)
    image1.save(filename1)
    image2.save(filename2)

    # Predict the digit for each image
    digit1 = predict_digit(filename1)
    digit2 = predict_digit(filename2)

    # Compare and return result
    result = digit1 == digit2
    return jsonify(same_digit=result)

if __name__ == '__main__':
    app.run(debug=True)
