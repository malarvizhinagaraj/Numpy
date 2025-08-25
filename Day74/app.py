from flask import Flask, request, jsonify, render_template
import pickle

# Load model & vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form["news"]
        transformed = vectorizer.transform([text])
        prediction = model.predict(transformed)[0]
        result = "REAL News ✅" if prediction == 1 else "FAKE News ❌"
        return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
