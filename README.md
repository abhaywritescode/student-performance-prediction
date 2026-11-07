# Student Performance Predictor

A lightweight, machine learning-powered web application that estimates student exam scores based on 11 distinct academic and lifestyle factors.

## 🧠 The Model: Why Linear Regression?

While several machine learning algorithms were tested during development, **Linear Regression** consistently performed the best for this specific use case.

When predicting diverse and continuous exam scores, traditional classification "accuracy" is less meaningful than the actual proximity of the prediction to the real score. By optimizing for **Mean Squared Error (MSE)** rather than strict accuracy, this model prioritized precision. Linear Regression provided the closest estimates to actual student scores, and its inherent simplicity meant it achieved these results efficiently without the need for heavy, complex hyperparameter tuning.

## ⚙️ How It Works

* **Features Assessed:** 11 distinct variables including study hours, average nightly sleep, teacher quality, and school type. *(Note: The `Gender` variable was explicitly excluded during data preprocessing as it is not an important predictive factor here).*
* **Data Pipeline:** Features are standardized using `StandardScaler`, and categorical text data is numerically mapped via `LabelEncoder`.
* **Deployment:** The serialized model (`.pkl`) is served through a minimal Python Flask API and rendered via HTML templates.

## 📁 Project Structure

```text
.
├── api/
│   ├── app.py                  # Main Flask application
│   ├── templates/              # UI templates (index.html, result.html)
│   └── [model.pkl, scaler.pkl] # Serialized model artifacts (generated)
├── images/
│   └── screenshot.png          # Application screenshot
├── notebook/
│   ├── dataset.csv             # Raw dataset
│   └── notebook.ipynb          # Model training and evaluation logic
├── requirements.txt            # Python dependencies
└── README.md

```

## 🚀 Quick Start

**1. Install dependencies**
Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt

```

**2. Generate the model artifacts (If missing)**
If `model.pkl` and `scaler.pkl` are not already in your `api/` directory, navigate to the `notebook/` folder and run `notebook.ipynb`. This will generate both files. Move them into the `api/` folder.

**3. Launch the application**

```bash
cd api
python app.py

```

Navigate to `[http://127.0.0.1:5000/](http://127.0.0.1:5000/)` in your browser to use the tool.

## 📊 Dataset

The model was trained on the [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) dataset from Kaggle.