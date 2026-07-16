from flask import Flask, request, render_template
import pickle
import pandas as  pd

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            data = request.form
            data = pd.json_normalize(data)

            data = scaler.transform(data)

            result = model.predict(data)
            return f"<h1 style='color: green'>Analyzed performance for the student is approximately {result[0]:0.2f}%. Real performance could be close to this value.</h1><a href='/'>Return home</a>"
        except Exception as e:
            return f"<h1 style='color: red'>Error: {e}</h1><a href='/'>Return home</a>"
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)