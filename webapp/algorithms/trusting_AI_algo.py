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
np.random.seed(6)

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
      "Statistical_Parity": 1/6,
      "Disparate_Mistreatment": 1/6,
      "Class_Imbalance": 1/6,
      "Biased_Data": 1/6,
      "Disparate_Treatment" : 1/6,
      "Disparate_Impact": 1/6
    },
    "explainability": {
      "Algorithm_Class": 0.55,
      "Correlated_Features": 0.15,
      "Model_Size": 0.15,
      "Feature_Relevance": 0.15
    },
    "robustness": {
      "Confidence_Score": 0.25,
      "Clique_Method"    : 0.25,
        "Loss_Sensitivity"  : 0.25, 
        "CLEVER_Score"   : 0.25,
    },
    "methodology": {
      "Normalization":  1/6,
      "Treatment_of_Corrupt_Values":  1/6,
      "Train_Test_Split":  1/6,
      "Regularization":  1/6,
      "Treatment_of_Categorical_Features":  1/6,
      "Feature_Filtering": 1/6
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
        Disparate_Treatment = score_Disparate_Mistreatment(),
        Disparate_Impact = score_Disparate_Mistreatment(),
        Class_Imbalance        = score_Class_Imbalance(),
        Biased_Data            = score_Biased_Data()
                 )
    
    return output

# functions for explainability score

def score_Algorithm_Class():
    return 5

def score_Correlated_Features():
    return 2

def score_Model_Size():
    return 2

def score_Feature_Relevance():
    return 3

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
        Clique_Method    = score_Class_Specific_Metrics(),
        Loss_Sensitivity   = score_Class_Specific_Metrics(),
        CLEVER_Score   = score_Class_Specific_Metrics()
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
        Treatment_of_Corrupt_Values     = score_Test_F1_score(),
        Train_Test_Split = score_Train_Test_Split(),
        Regularization   = score_Regularization(),
        Feature_Filtering    = score_Test_Accuracy(),
        Treatment_of_Categorical_Features  = score_Test_Accuracy()
                 )
    return output

# define algo
def trusting_AI_scores(model, train_data, test_data, config_fairness, config_explainability, config_robustness, config_methodology):
    score = dict(
        fairness       = calc_fairness_score(),
        explainability = calc_explainability_score(model, train_data, test_data, config_explainability),
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



