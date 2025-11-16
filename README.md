Iris Data Poisoning Experiment (MLflow + MLOps)**

This project demonstrates how **data poisoning** impacts machine learning model performance using the Iris dataset. Poisoning is simulated by injecting feature noise and flipping labels, and all experiment runs are logged using **MLflow** for easy comparison.

## **What the Experiment Does**

* Loads the Iris dataset
* Applies data poisoning at **0%, 5%, 10%, 50%** levels
* Adds Gaussian noise to features
* Randomly flips labels to wrong classes
* Trains a Logistic Regression model for each poisoning level
* Logs accuracy, F1-score, parameters, and model artifacts in MLflow

## **Expected Results**

Higher poisoning → lower performance:

| Poison Ratio | Effect                   |
| ------------ | ------------------------ |
| 0%           | Baseline, high accuracy  |
| 5%           | Slight performance drop  |
| 10%          | Noticeable degradation   |
| 50%          | Model becomes unreliable |

MLflow lets you visualize how metrics decline as poisoning increases.

## **How to Run**

### **Local**

```
pip install -r requirements.txt
mlflow ui --port 5000
python iris_poisoning_experiment.py
```

Open: `http://127.0.0.1:5000`

### **GCP (Vertex AI Workbench)**

* Create a Managed Notebook
* Install dependencies
* Run MLflow UI (`mlflow ui --port 5000`)
* Run the script

## **Why This Matters**

Data poisoning attacks can severely damage ML models.
This experiment shows:

* How sensitive models are to corrupted data
* Why monitoring, validation, and clean data are critical in MLOps

## **Mitigation Strategies**

* Data validation (schema + drift detection)
* Monitoring feature distributions
* Robust training techniques
* Human review for suspicious samples

---


