# -*- coding: utf-8 -*-
import numpy as np
import collections
from helpers import *
import tensorflow as tf
from math import isclose
import re

info = collections.namedtuple('info', 'description value')
# === Methodology Metrics ===
def analyse(model, training_dataset, test_dataset, factsheet, methodology_config):

    #_accuracy_thresholds = methodology_config["score_test_accuracy"]["thresholds"]["value"]
    #_f1_score_thresholds = methodology_config["score_f1"]["thresholds"]["value"]
    normalization_mapping = methodology_config["score_normalization"]["mappings"]["value"]
    missing_data_mapping = methodology_config["score_missing_data"]["mappings"]["value"]
    train_test_split_mapping = methodology_config["score_train_test_split"]["mappings"]["value"]

    metrics = list_of_metrics("methodology")
    output = dict(
        #output[metric] = exec("%s_score(model, training_dataset, test_dataset, factsheet, methodology_config)" % metric)
        normalization  = normalization_score(model, training_dataset, test_dataset, factsheet, normalization_mapping),
        missing_data = missing_data_score(model, training_dataset, test_dataset, factsheet, missing_data_mapping),
        regularization   = regularization_score(model, training_dataset, test_dataset, factsheet, methodology_config),
        train_test_split = train_test_split_score(model, training_dataset, test_dataset, factsheet, train_test_split_mapping),
        #test_accuracy = test_accuracy_score(model, training_dataset, test_dataset, factsheet, accuracy_thresholds),
        #f1_score = f1_score(model, training_dataset, test_dataset, factsheet, f1_score_thresholds),
        factsheet_completeness= factsheet_completeness_score(model, training_dataset, test_dataset, factsheet, methodology_config)
    )

    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)

# --- Normalization ---
def normalization_score(model, train_data, test_data, factsheet, mappings):
    X_train = train_data.iloc[:, :-1]
    X_test = test_data.iloc[:, :-1]

    train_mean = np.mean(np.mean(X_train))
    train_std = np.mean(np.std(X_train))
    test_mean = np.mean(np.mean(X_test))
    test_std = np.mean(np.std(X_test))

    properties = {
        "dep": info('Depends on','Training and Testing Data'),
        "Training_mean": info("Mean of the training data", "{:.2f}".format(train_mean)),
        "Training_std": info("Standard deviation of the training data", "{:.2f}".format(train_std)),
        "Test_mean": info("Mean of the test data", "{:.2f}".format(test_mean)),
        "Test_std": info("Standard deviation of the test data", "{:.2f}".format(test_std))
    }
    if not (any(X_train < 0) or any(X_train > 1)) and not (any(X_test < 0) or any(X_test > 1)):
        score = mappings["training_and_test_normal"]
        properties["normalization"] = info("Normalization", "Training and Testing data are normalized")
    elif isclose(train_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and isclose(train_std, 1, rel_tol=1e-3, abs_tol=1e-6) and (not isclose(test_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and not isclose(test_std, 1, rel_tol=1e-3, abs_tol=1e-6)):
        score = mappings["training_standardized"]
        properties["normalization"] = info("Normalization", "Training data are standardized")
    elif isclose(train_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and isclose(train_std, 1, rel_tol=1e-3, abs_tol=1e-6) and (isclose(test_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and isclose(test_std, 1, rel_tol=1e-3, abs_tol=1e-6)):
        score = mappings["training_and_test_standardize"]
        properties["normalization"] = info("Normalization", "Training and Testing data are standardized")
    elif any(X_train < 0) or any(X_train > 1):
        score = mappings["None"]
        properties["normalization"] = info("Normalization", "None")
    elif not (any(X_train < 0) or any(X_train > 1)) and (any(X_test < 0) or any(X_test > 1)):
        score = mappings["training_normal"]
        properties["normalization"] = info("Normalization", "Training data are normalized")
    return result(score=score, properties=properties)

# --- Missing Data ---
def missing_data_score(model, training_dataset, test_dataset, factsheet, mappings):
    try:
        missing_values = training_dataset.isna().sum().sum() + test_dataset.isna().sum().sum()
        if missing_values > 0:
            score = mappings["null_values_exist"]
        else:
            score = mappings["no_null_values"]
        return result(score=score,properties={"dep" :info('Depends on','Training Data'),
            "null_values": info("Number of the null values", "{}".format(missing_values))})
    except:
        return result(score=np.nan, properties={})

    
# --- Train-Test-Split ---
def train_test_split_score(model, training_dataset, test_dataset, factsheet, mappings):
    try:
        training_data_ratio, test_data_ratio = train_test_split_metric(training_dataset, test_dataset)
        properties= {"dep" :info('Depends on','Training and Testing Data'),
            "train_test_split": info("Train test split", "{:.2f}/{:.2f}".format(training_data_ratio, test_data_ratio))}
        for k in mappings.keys():
            thresholds = re.findall(r'\d+-\d+', k)
            for boundary in thresholds:
                [a, b] = boundary.split("-")
                if training_data_ratio >= int(a) and training_data_ratio < int(b):
                    score = mappings[k]
        return result(score=score, properties=properties)
    except Exception as e:
        print(e)
        return result(score=np.nan, properties={})

def train_test_split_metric(training_dataset, test_dataset):
    n_train = len(training_dataset)
    n_test = len(test_dataset)
    n = n_train + n_test
    return round(n_train/n*100), round(n_test/n*100)

def is_between(a, x, b):
    return min(a, b) < x < max(a, b)

# --- Regularization ---
def regularization_score(model, training_dataset, test_dataset, factsheet, methodology_config):
    score = 1
    regularization = regularization_metric(factsheet)
    properties = {"dep" :info('Depends on','Factsheet'),
        "regularization_technique": info("Regularization technique", regularization)}

    if regularization == "elasticnet_regression":
        score = 5
    elif regularization == "lasso_regression" or regularization == "lasso_regression":
        score = 4
    elif regularization == "Other":
        score = 3
    elif regularization == NOT_SPECIFIED:
        score = np.nan
    else:
        score = 1
    return result(score=score, properties=properties)

def regularization_metric(factsheet):
    if "methodology" in factsheet and "regularization" in factsheet["methodology"]:
        return factsheet["methodology"]["regularization"]
    else:
        return NOT_SPECIFIED
    

# --- Test Accuracy ---
def test_accuracy_score(model, training_dataset, test_dataset, factsheet, thresholds):
    try:
        test_accuracy = test_accuracy_metric(model, test_dataset, factsheet)
        score = np.digitize(test_accuracy, thresholds)
        #return result(score=float(score), properties={"test_accuracy": info("Test Accuracy", "{:.2f}".format(test_accuracy))})
        return result(score=np.nan, properties={})
    except Exception as e:
        #_return result(score=np.nan, properties={})
        return result(score=np.nan, properties={})
        
def test_accuracy_metric(model, test_dataset, factsheet):
    target_column = None
    if "general" in factsheet and "target_column" in factsheet["general"]:
        target_column = factsheet["general"]["target_column"]
    
    if target_column:
        X_test = test_dataset.drop(target_column, axis=1)
        y_test = test_dataset[target_column]
    else:
        X_test = test_dataset.iloc[:,:DEFAULT_TARGET_COLUMN_INDEX]
        y_test = test_dataset.iloc[:,DEFAULT_TARGET_COLUMN_INDEX: ]
    
    y_true =  y_test.values.flatten()
    y_pred = model.predict(X_test)

    if isinstance(model, tf.keras.models.Sequential):
        y_pred = np.argmax(y_pred, axis=1)

    accuracy = metrics.accuracy_score(y_true, y_pred).round(2)
    return accuracy

# --- F1 Score ---
def f1_score(model, training_dataset, test_dataset, factsheet, thresholds):
    try:
        f1_score = f1_metric(model, test_dataset, factsheet)
        score = np.digitize(f1_score, thresholds)
        #return result(score=float(score), properties={"f1_score": info("F1 Score", "{:.2f}".format(f1_score))})
        return result(score=np.nan, properties={})
    except:
        return result(score=np.nan, properties={})
         
        
def f1_metric(model, test_dataset, factsheet):
    target_column = None
    if "general" in factsheet and "target_column" in factsheet["general"]:
        target_column = factsheet["general"]["target_column"]
    
    if target_column:
        X_test = test_dataset.drop(target_column, axis=1)
        y_test = test_dataset[target_column]
    else:
        X_test = test_dataset.iloc[:,:DEFAULT_TARGET_COLUMN_INDEX]
        y_test = test_dataset.iloc[:,DEFAULT_TARGET_COLUMN_INDEX: ]
    
    y_true =  y_test.values.flatten()
    y_pred = model.predict(X_test)
    if isinstance(model, tf.keras.models.Sequential):
        y_pred = np.argmax(y_pred, axis=1)
    f1_metric = metrics.f1_score(y_true, y_pred,average="weighted").round(2)
    return f1_metric
    
# --- Factsheet Completeness ---
def factsheet_completeness_score(model, training_dataset, test_dataset, factsheet, methodology_config):
    score = 0
    properties= {"dep" :info('Depends on','Factsheet')}
    
    n = len(GENERAL_INPUTS)
    ctr = 0
    for e in GENERAL_INPUTS:
        if "general" in factsheet and e in factsheet["general"]:
            ctr+=1
            properties[e] = info("Factsheet Property {}".format(e.replace("_"," ")), "present")
        else:
            properties[e] = info("Factsheet Property {}".format(e.replace("_"," ")), "missing")
    score = round(ctr/n*5)
    return result(score=score, properties=properties)
