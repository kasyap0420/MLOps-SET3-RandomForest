\# SET-3: Random Forest Classifier — Diabetes Prediction



An academic MLOps project implementing a Random Forest Classifier using the Pima Indians Diabetes dataset.



\## Project Overview



This project demonstrates an end-to-end machine learning workflow covering data inspection, preprocessing, feature engineering, feature selection, model training, evaluation, experiment tracking, version control and API deployment.



\## Dataset



\- Dataset: Pima Indians Diabetes

\- Source: Kaggle — UCI Pima Indians Diabetes Database

\- Records: 768

\- Input Features: 8

\- Target Variable: `Outcome`

\- Classification: Binary (0 = Non-Diabetic, 1 = Diabetic)



\## Machine Learning Workflow



1\. Dataset inspection and exploratory data analysis.

2\. Invalid zero-value identification and median imputation.

3\. Outlier detection using the IQR method.

4\. Data validation using Great Expectations.

5\. Feature engineering using `Glucose\_BMI\_Index`.

6\. Feature selection using Recursive Feature Elimination (RFE).

7\. Feature scaling using StandardScaler.

8\. Random Forest model training.

9\. Hyperparameter optimization using RandomizedSearchCV.

10\. Model evaluation using Accuracy, Precision, Recall, F1-Score and ROC-AUC.



\## Technology Stack



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- Matplotlib and Seaborn

\- Great Expectations

\- Git and GitHub

\- DVC

\- MLflow

\- FastAPI and Uvicorn

\- Joblib



\## Project Structure



```text

SET3-RandomForest/

├── data/

│   ├── raw/

│   └── processed/

├── src/

│   ├── data\_loader.py

│   ├── preprocessing.py

│   ├── feature\_engineering.py

│   ├── validation.py

│   ├── train.py

│   ├── evaluate.py

│   └── predict.py

├── models/

├── outputs/

│   └── plots/

├── config/

├── SET3\_RandomForest.ipynb

├── main.py

├── app.py

├── requirements.txt

├── requirements-lock.txt

└── README.md

```



\## Installation



Create and activate a virtual environment:



```cmd

python -m venv .venv

.venv\\Scripts\\activate

```



Install dependencies:



```cmd

python -m pip install -r requirements.txt

```



Place the downloaded `diabetes.csv` file inside `data/raw/`.



Launch Jupyter Notebook:



```cmd

jupyter notebook

```



Open `SET3\_RandomForest.ipynb` and execute the cells sequentially.



\## Model Evaluation



Evaluation results are saved in:



\- `outputs/metrics.json`

\- `outputs/plots/confusion\_matrix.png`



The actual metrics are generated from the trained model.



\## Version Control



Git tracks source code, notebooks, and project changes.



DVC tracks datasets and trained model artifacts.



```cmd

dvc add data/raw/diabetes.csv

dvc add models/rf\_model.pkl

dvc push

```



\## Experiment Tracking



MLflow is used to track:



\- Hyperparameters

\- Accuracy, Precision, Recall, F1-Score and ROC-AUC

\- Confusion matrix and evaluation artifacts

\- Registered model versions



\## Model Deployment



FastAPI and Uvicorn are used to expose the trained model through REST API endpoints:



\- GET `/`

\- GET `/health`

\- GET `/ready`

\- GET `/model-info`

\- POST `/predict`

\- GET `/metrics`



\## Project Status



\- Notebook-based model training and evaluation: Implemented.

\- Dataset tracking using DVC: Implemented.

\- Model artifact versioning: Initial version being recorded.

\- MLflow experiment tracking and registry: Next implementation stage.

\- Modular pipeline and FastAPI deployment: Next implementation stage.



\## Disclaimer



This project is intended for academic and educational purposes. Predictions are not medical diagnoses.

