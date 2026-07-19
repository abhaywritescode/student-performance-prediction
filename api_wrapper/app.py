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
            return render_template("result.html", result=result)
        except Exception as e:
            return f"<b>Error:</b> {e}"
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)