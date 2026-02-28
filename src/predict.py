import joblib

MODEL_PATH = "../model/wine_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_data(X):
    model = load_model()
    return model.predict(X)

def predict_probability(X):
    model = load_model()
    return model.predict_proba(X)
