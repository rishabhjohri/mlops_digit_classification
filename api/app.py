from flask import Flask, request
from joblib import load

app = Flask(__name__)

# Models dictionary to store loaded models
models = {}

def load_model():
    models['svm'] = load("svm_model_path.joblib")  # Replace with your actual SVM model path
    models['lr'] = load("lr_model_path.joblib")  # Replace with your actual LR model path
    models['tree'] = load("tree_model_path.joblib")  # Replace with your actual Decision Tree model path

# Load models
load_model()

@app.route("/")
def hello_world():
    return "<b>Hello, World!</b>"

@app.route("/predict/<model_type>", methods=['POST'])
def predict_digit(model_type):
    if model_type not in models:
        return {"error": "Model not found"}, 404

    image = request.json['image']
    predicted = models[model_type].predict([image])
    return {"y_predicted": int(predicted[0])}

if __name__ == "__main__":
    app.run(debug=True)
