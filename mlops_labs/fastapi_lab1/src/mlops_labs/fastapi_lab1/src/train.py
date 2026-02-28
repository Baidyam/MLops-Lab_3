from sklearn.tree import DecisionTreeClassifier
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train Decision Tree on Wine dataset and save model.
    """
    model = DecisionTreeClassifier(
        max_depth=4,
        random_state=12
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "../model/wine_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    fit_model(X_train, y_train)

    print("Wine model trained and saved successfully.")
