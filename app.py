from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# ---------------- Flask App ----------------
app = Flask(__name__)

# ---------------- Load Data ----------------
df = pd.read_csv("Housing.csv")   # rename if needed

X = df.drop(columns=["price"])
y = df["price"]

binary_cols = [
    "mainroad","guestroom","basement",
    "hotwaterheating","airconditioning","prefarea"
]

categorical_cols = ["furnishingstatus"]

numeric_cols = [
    "area","bedrooms","bathrooms","stories","parking"
]

# ---------------- Preprocessing ----------------
preprocessor = ColumnTransformer(
    transformers=[
        ("bin", OneHotEncoder(drop="if_binary"), binary_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

# ---------------- Train Model ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = {
            "area": int(request.form["area"]),
            "bedrooms": int(request.form["bedrooms"]),
            "bathrooms": int(request.form["bathrooms"]),
            "stories": int(request.form["stories"]),
            "mainroad": request.form["mainroad"],
            "guestroom": request.form["guestroom"],
            "basement": request.form["basement"],
            "hotwaterheating": request.form["hotwaterheating"],
            "airconditioning": request.form["airconditioning"],
            "parking": int(request.form["parking"]),
            "prefarea": request.form["prefarea"],
            "furnishingstatus": request.form["furnishingstatus"]
        }

        user_df = pd.DataFrame([data])
        pred = pipeline.predict(user_df)[0]
        prediction = f"₹ {pred:,.0f}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
