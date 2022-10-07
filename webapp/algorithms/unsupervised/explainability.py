# === Explainability ===
import statistics

import numpy as np
import pandas as pd
import json
import collections
import keras
from sklearn.ensemble import IsolationForest

result = collections.namedtuple('result', 'score properties')
info = collections.namedtuple('info', 'description value')


def analyse(clf, train_data, test_data, outliers_data, config, factsheet):
    
    #function parameters
    #clf_type_score = config["score_algorithm_class"]["clf_type_score"]["value"]
    ms_thresholds = config["score_model_size"]["thresholds"]["value"]
    cf_thresholds = config["score_correlated_features"]["thresholds"]["value"]
    pfi_thresholds = config["score_permutation_feature_importance"]["thresholds"]["value"]
    high_cor = config["score_correlated_features"]["high_cor"]["value"]

    print_details = True

    if isKerasAutoencoder(clf):
        outlier_thresh = get_threshold_mse_iqr(clf, train_data)
    else:
        outlier_thresh = 0

    output = dict(
        #algorithm_class     = algorithm_class_score(clf, clf_type_score),
        correlated_features = correlated_features_score(train_data, test_data, thresholds=cf_thresholds, high_cor=high_cor, print_details=print_details),
        model_size          = model_size_score(train_data, ms_thresholds, print_details=print_details),
        permutation_feature_importance   = permutation_feature_importance_score(clf, outliers_data, outlier_thresh, thresholds = pfi_thresholds, print_details = print_details),

        #correlated_features = result(score=int(1), properties={}),
        #model_size = result(score=int(1), properties={}),
        #permutation_feature_importance = result(score=int(1), properties={})
    )

    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return result(score=scores, properties=properties)


def algorithm_class_score(clf, clf_type_score):

    clf_name = type(clf).__name__
    exp_score = clf_type_score.get(clf_name,np.nan)
    properties= {"dep" :info('Depends on','Model'),
        "clf_name": info("model type",clf_name)}
    
    return  result(score=exp_score, properties=properties)

def correlated_features_score(train_data, test_data, thresholds=[0.05, 0.16, 0.28, 0.4], high_cor=0.9, print_details = False):

    test_data = test_data.copy()
    train_data = train_data.copy()

    df_comb = pd.concat([test_data, train_data])
    corr_matrix = df_comb.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > high_cor)]
    pct_drop = len(to_drop) / len(df_comb.columns)
    score = 5 - np.digitize(pct_drop, thresholds, right=True)

    if print_details:
        print("\t CORRELATED FEATURES DEATAILS")
        print("\t feat. to drop", to_drop)

    properties= {
        "dep" : info('Depends on', 'Training Data, Test Data'),
        "pct_drop" : info("Percentage of highly correlated features", "{:.2f}%".format(100*pct_drop))
    }
    
    return  result(score=int(score), properties=properties)


def model_size_score(test_data, thresholds = np.array([10,30,100,500]), print_details = False):
    
    dist_score = 5- np.digitize(test_data.shape[1], thresholds, right=True)

    if print_details:
        print("\t MODEL SIZE DETAILS")
        print("\t num of features: ", test_data.shape[1])

    return result(score=int(dist_score), properties={"dep" :info('Depends on','Test Data'),
        "n_features": info("number of features", test_data.shape[1]-1)})

def permutation_feature_importance_score(model, outliers_data, outlier_thresh, thresholds = [0.2,0.15,0.1,0.05], print_details = False):

    features = list(outliers_data.columns)

    shuffles = 3
    feature_importance = {}
    num_redundant_feat = 0
    num_datapoints = outliers_data.shape[0]

    accuracy_no_permutation = compute_outlier_matrix(model, outliers_data, outlier_thresh, print_details)


    for i, feature in enumerate(features):
        feature_importance[feature] = []
        outliers_data_copy = outliers_data.copy()

        for _ in range(shuffles):
            print(i, feature)
            # compute outlier detection with permutation
            outliers_data_copy[feature] = np.random.permutation(outliers_data[feature])
            accuracy_permutation = compute_outlier_matrix(model, outliers_data_copy, outlier_thresh, print_details)

            num_diff_val = np.sum(accuracy_no_permutation != accuracy_permutation)

            permutation = num_diff_val / num_datapoints
            print("permutation: ", permutation)
            feature_importance[feature].append(permutation)

        feature_importance[feature] = statistics.mean(feature_importance[feature])
        if (feature_importance[feature] == 0):
            num_redundant_feat += 1

    ratio_redundant_feat = num_redundant_feat / len(feature_importance)
    feature_importance_desc = list(dict(sorted(feature_importance.items(), key=lambda item: item[1])).keys())[::-1]
    print(thresholds)

    score = np.digitize(ratio_redundant_feat, thresholds, right=True)+1
    properties = {
        "dep": info('Depends on', 'Model, Outliers Data'),
        "num_redundant_features": info("number of redundant features", num_redundant_feat),
        "num_features": info("number of features", len(feature_importance)),
        "ratio_redundant_features": info("ratio of redundant features", ratio_redundant_feat),
        "importance": info("feature importance descending", {"value": feature_importance_desc})
    }

    return result(score=int(score), properties=properties)

def isKerasAutoencoder(model):
    return isinstance(model, keras.engine.functional.Functional)

def isIsolationForest(model):
    return isinstance(model, IsolationForest)

#Get anomaly threshold from "autoencoder" setting the threshold in Q1,Q3+-1.5IQR
def get_threshold_mse_iqr(autoencoder,train_data):
    train_predicted = autoencoder.predict(train_data)
    mse = np.mean(np.power(train_data - train_predicted, 2), axis=1)
    iqr = np.quantile(mse,0.75) - np.quantile(mse, 0.25) # interquartile range
    up_bound = np.quantile(mse,0.75) + 1.5*iqr
    bottom_bound = np.quantile(mse,0.25) - 1.5*iqr
    thres = [up_bound,bottom_bound]
    return thres

#Predict outliers in "df" using "autoencoder" model and "threshold_mse" as anomaly limit
def detect_outliers(autoencoder, df, threshold_mse):
    if(len(threshold_mse)==2):
        return detect_outliers_range(autoencoder, df, threshold_mse)
    pred=autoencoder.predict(df)
    mse = np.mean(np.power(df - pred, 2), axis=1)
    #plt.hist(mse, bins=100)
    #plt.show()
    outliers = [np.array(mse) < threshold_mse]
    return outliers

def detect_outliers_range(autoencoder, df, threshold_mse):
    pred=autoencoder.predict(df)
    mse = np.mean(np.power(df - pred, 2), axis=1)
    up_bound = threshold_mse[0]
    bottom_bound = threshold_mse[1]
    outliers = [(np.array(mse) < up_bound)&(np.array(mse) > bottom_bound)]
    return outliers

def compute_outlier_matrix(model, data, outlier_thresh, print_details=False):
    if isKerasAutoencoder(model):
        mad_outliers = detect_outliers(model, data, outlier_thresh)[0]

    elif isIsolationForest(model):
        mad_outliers = model.predict(data)

    else:
        mad_outliers = model.predict(data)

    if print_details:
        print("\t outlier matrix: ", mad_outliers)

    return mad_outliers
