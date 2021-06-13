import os
import time
import random
import numpy as np
import pickle
import statistics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import json
from math import pi
import sklearn.metrics as metrics

## get inputs
def get_case_inputs(case):
    scenario_path = os.path.join("Validation", case)

    with open(os.path.join(scenario_path, "factsheet.txt")) as file:
        factsheet = json.load(file)
        
    with open(os.path.join(scenario_path, "model.sav"),'rb') as file:
        model = pickle.load(file)

    with open(os.path.join(scenario_path, "X_test.pkl"),'rb') as file:
        X_test = pickle.load(file)

    with open(os.path.join(scenario_path, "X_train.pkl"),'rb') as file:
        X_train = pickle.load(file)

    with open(os.path.join(scenario_path, "y_test.pkl"),'rb') as file:
        y_test = pickle.load(file)

    with open(os.path.join(scenario_path, "y_train.pkl"),'rb') as file:
        y_train = pickle.load(file)

    return [factsheet, model, X_test, X_train, y_test, y_train]


# define model inputs
# choose scenario case (case1,case1,..)
case = "case1"
np.random.seed(1)

# load case inputs
factsheet, model, X_test, X_train, y_test, y_train = get_case_inputs(case)

# algo config (load from config file in the future)
config = {
  "fairness": {
    "par1": 0,
    "par2": 0
  },
  "explainability": {
    "par1": 0,
    "par2": 0
  },
  "robustness": {
    "par1": 0,
    "par2": 0
  },
  "methodology": {
    "par1": 0,
    "par2": 0
  },
  "weights": {
    "fairness": {
      "Statistical_Parity": 0.25,
      "Disparate_Mistreatment": 0.25,
      "Class_Imbalance": 0.25,
      "Biased_Data": 0.25
    },
    "explainability": {
      "Algorithm_Class": 0.55,
      "Correlated_Features": 0.15,
      "Model_Size": 0.15,
      "Feature_Relevance": 0.15
    },
    "robustness": {
      "Confidence_Score": 0.5,
      "Class_Specific_Metrics": 0.5
    },
    "methodology": {
      "Normalization": 0.2,
      "Test_F1_score": 0.2,
      "Train_Test_Split": 0.2,
      "Regularization": 0.2,
      "Test_Accuracy": 0.2
    }
  }
}

# functions for fairness score

def score_Statistical_Parity():
    return np.random.randint(1,6)

def score_Disparate_Mistreatment():
    return np.random.randint(1,6)

def score_Class_Imbalance():
    return np.random.randint(1,6)

def score_Biased_Data():
    return np.random.randint(1,6)

def calc_fairness_score():
    
    output = dict(
        Statistical_Parity     = score_Statistical_Parity(),
        Disparate_Mistreatment = score_Disparate_Mistreatment(),
        Class_Imbalance        = score_Class_Imbalance(),
        Biased_Data            = score_Biased_Data()
                 )
    
    return output

# functions for explainability score

def score_Algorithm_Class():
    return np.random.randint(1,6)

def score_Correlated_Features():
    return np.random.randint(1,6)

def score_Model_Size():
    return np.random.randint(1,6)

def score_Feature_Relevance():
    return np.random.randint(1,6)

def calc_explainability_score():
    
    output = dict(
        Algorithm_Class     = score_Algorithm_Class(),
        Correlated_Features = score_Correlated_Features(),
        Model_Size          = score_Model_Size(),
        Feature_Relevance   = score_Feature_Relevance()
                 )
    return output

# functions for robustness score

def score_Confidence_Score():
    return np.random.randint(1,6)

def score_Class_Specific_Metrics():
    return np.random.randint(1,6)

def calc_robustness_score():
    
    output = dict(
        Confidence_Score          = score_Confidence_Score(),
        Class_Specific_Metrics   = score_Class_Specific_Metrics()
                 )
    return output

# functions for methodology score

def score_Normalization():
    return np.random.randint(1,6)

def score_Test_F1_score():
    return np.random.randint(1,6)

def score_Train_Test_Split():
    return np.random.randint(1,6)

def score_Regularization():
    return np.random.randint(1,6)

def score_Test_Accuracy():
    return np.random.randint(1,6)

def calc_methodology_score():
    
    output = dict(
        Normalization    = score_Normalization(),
        Test_F1_score    = score_Test_F1_score(),
        Train_Test_Split = score_Train_Test_Split(),
        Regularization   = score_Regularization(),
        Test_Accuracy    = score_Test_Accuracy()
                 )
    return output

# define algo
def trusting_AI_scores(factsheet, model, X_test, X_train, y_test, y_train, config):
    score = dict(
        fairness       = calc_fairness_score(),
        explainability = calc_explainability_score(),
        robustness     = calc_robustness_score(),
        methodology    = calc_methodology_score()
    )
    return score

# calculate final score with weigths
def get_final_score():
    scores = trusting_AI_scores(factsheet, model, X_test, X_train, y_test, y_train, config)
    final_scores = dict()
    for pillar, item in scores.items():
        weighted_scores = list(map(lambda x: scores[pillar][x] * config["weights"][pillar][x], scores[pillar].keys()))
        final_scores[pillar] = round(np.sum(weighted_scores),1)

    return final_scores, scores

def get_perfomance_metric(clf, y_test, X_test):
    
    y_true =  y_test.values.flatten()
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    labels = np.unique(np.array([y_pred,y_true]).flatten())
    
    performance_metrics = {
        "accuracy" :  metrics.accuracy_score(y_true, y_pred),
        "global recall" :  metrics.recall_score(y_true, y_pred, labels=labels, average="micro"),
        "class weighted recall" : metrics.recall_score(y_true, y_pred,average="weighted"),
        "global precision" : metrics.precision_score(y_true, y_pred, labels=labels, average="micro"),
        "class weighted precision" : metrics.precision_score(y_true, y_pred,average="weighted"),
        "global f1 score" :  metrics.f1_score(y_true, y_pred,average="micro"),
        "class weighted f1 score" :  metrics.f1_score(y_true, y_pred,average="weighted"),
        "cross-entropy loss" : metrics.log_loss(y_true, y_pred_proba),
        "ROC AUC" : metrics.roc_auc_score(y_true, y_pred_proba,average="weighted", multi_class='ovr')#one vs rest method
        }
    
    # for name, value in performance_metrics.items():
    #     print("%1s: %2.2f" %(name, value))
    
    return performance_metrics

def get_performance_table():
    performance_metrics = get_perfomance_metric(model, y_test, X_test)
    df = pd.DataFrame(performance_metrics, index=["Performance Metrics"]).transpose()
    df["Performance Metrics"] = df["Performance Metrics"].round(2)
    df = df.loc[["accuracy","class weighted recall","class weighted precision","class weighted f1 score","cross-entropy loss","ROC AUC"]]
    return df
