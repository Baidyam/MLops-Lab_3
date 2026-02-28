# Wine Classification API using FastAPI (MLOps Lab 3)

## Overview

This project demonstrates how to deploy a Machine Learning model as a REST API using FastAPI and Uvicorn. The API allows users to send input features and receive predictions from a trained classification model.

The original lab used the Iris dataset. In this modified version, the dataset has been changed to the Wine dataset, and an additional endpoint has been added to return prediction probabilities.

This project illustrates key MLOps concepts including model training, serialization, API deployment, and inference serving.

---

## Objectives

- Train a Machine Learning classification model
- Save the trained model using joblib
- Serve the model using FastAPI
- Create REST API endpoints for predictions
- Modify the original lab by changing dataset and adding a new endpoint

---

## Modifications Made

The following changes were made to differentiate this lab from the original repository:

### 1. Dataset Change
- Original dataset: Iris dataset (4 features)
- Modified dataset: Wine dataset (13 features)
- Source: `sklearn.datasets.load_wine()`

Reason:
- The Wine dataset provides more features and a different classification problem, making the implementation unique.

---

### 2. Model Modification

- Trained a Decision Tree Classifier on the Wine dataset
- Saved the trained model as:
  model/wine_model.pkl
- File modified:
  src/train.py
  
---

### 3. New API Endpoint Added

A new endpoint was added:
POST /predict_proba

This endpoint returns prediction probabilities for each class.

Example response:
{
"probabilities": [0.98, 0.02, 0.00]
}

File modified:
src/main.py

---

### 4. Updated Prediction Logic

Prediction code was updated to load the Wine model and support both class prediction and probability prediction.

File modified:
src/predict.py

---

## Project Structure
MLops-Lab_3/
│
├── model/
│ └── wine_model.pkl
│
├── src/
│ ├── data.py
│ ├── train.py
│ ├── predict.py
│ └── main.py
│
├── requirements.txt
└── README.md

---

## Installation and Setup

### Step 1: Clone Repository


git clone https://github.com/Baidyam/MLops-Lab_3.git

cd MLops-Lab_3


### Step 2: Create Virtual Environment (Mac/Linux)


python3 -m venv venv
source venv/bin/activate


### Step 3: Install Dependencies


pip install -r requirements.txt


---

## Train the Model

Navigate to src folder and run:


cd src
python train.py


This creates:


model/wine_model.pkl


---

## Run FastAPI Server

From src folder:


uvicorn main:app --reload


Server will start at:


http://127.0.0.1:8000


---

## API Documentation

Open Swagger UI:


http://127.0.0.1:8000/docs


Available endpoints:

### Health Check


GET /


Response:


{
"status": "Wine API healthy"
}


---

### Predict Class


POST /predict


Returns predicted wine class.

---

### Predict Probability (New Endpoint)


POST /predict_proba


Returns probability for each class.

---

## Example Input


{
"alcohol": 14.23,
"malic_acid": 1.71,
"ash": 2.43,
"alcalinity_of_ash": 15.6,
"magnesium": 127,
"total_phenols": 2.80,
"flavanoids": 3.06,
"nonflavanoid_phenols": 0.28,
"proanthocyanins": 2.29,
"color_intensity": 5.64,
"hue": 1.04,
"od280_od315": 3.92,
"proline": 1065
}


---

## Technologies Used

- Python
- FastAPI
- Uvicorn
- Scikit-learn
- Joblib
- Pydantic

---

## Key MLOps Concepts Demonstrated

- Model training
- Model serialization
- API deployment
- Inference serving
- REST API development
- Model integration with web service

---

## Author

Moumita Baidya  
Master’s in Data Science  
Northeastern University

---

## Conclusion

This project successfully demonstrates deploying a Machine Learning model using FastAPI. T

