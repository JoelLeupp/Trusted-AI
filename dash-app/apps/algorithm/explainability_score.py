# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json
import collections

result = collections.namedtuple('result', 'score properties')
info = collections.namedtuple('info', 'description value')
#for testing
# import os


# os.chdir(r"C:\Users\Besitzer\Documents\GitHub\Trusted-AI\Scenario")
# from apps.algorithm.helper_functions import get_case_inputs

# case = "case1"
# # load case inputs
# factsheet, model, X_test, X_train, y_test, y_train = get_case_inputs(case)

# # #get config
# with open("apps/algorithm/config_explainability.json") as file:
#         config = json.load(file)
    

# functions for explainability score

def score_Algorithm_Class(clf, clf_type_score):

    clf_name = type(clf).__name__
    exp_score = clf_type_score.get(clf_name,np.nan)
    properties= {"clf_name": info("model type",clf_name)}
    
    return  result(score=exp_score, properties=properties)

def score_Correlated_Features(train_data, test_data):
    
    X_train = train_data.iloc[:,:-1]
    X_test = test_data.iloc[:,:-1]
    
    df_comb = pd.concat([X_test, X_train])
    corr_matrix = df_comb.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    pct_drop = len(to_drop)/len(df_comb.columns)
    
    bins = np.linspace(0.05, 0.4, 4)
    score = 5-np.digitize(pct_drop, bins, right=False) 
    properties= {"pct_drop" : info("Percentage of highly correlated features","{:.2f}%".format(100*pct_drop))}
    
    return  result(score=score, properties=properties)


def score_Model_Size(clf, bins = np.array([10,30,100,500])):
        
    dist_score = 5- np.digitize(clf.n_features_, bins, right=False) 
         
    return result(score=dist_score, properties={"n_features": info("number of features", clf.n_features_)})

def score_Feature_Relevance(clf, train_data, scale_factor):
    
    # scale_factor = 1.5
    # distri_threshold = 0.6
    X_train = train_data.iloc[:,:-1]
    importance=clf.feature_importances_
    
    feat_labels = X_train.columns
    indices = np.argsort(importance)[::-1]
    feat_labels = feat_labels[indices]

    importance = importance[indices]
    
    # calculate quantiles for outlier detection
    q1, q2, q3 = np.percentile(importance, [25, 50 ,75])
    lower_threshold , upper_threshold = q1 - scale_factor*(q3-q1),  q3 + scale_factor*(q3-q1) 
    
    #get the number of outliers defined by the two thresholds
    n_outliers = sum(map(lambda x: (x < lower_threshold) or (x > upper_threshold), importance))
    
    # percentage of features that concentrate distri_threshold percent of all importance
    pct_dist = sum(np.cumsum(importance) < 0.6) / len(importance)
    
    bins = np.linspace(0.05, 0.3, 4) # np.array([0.05,0.13333333,0.21666667, 0.3])
    dist_score = np.digitize(pct_dist, bins, right=False) + 1 
    
    if n_outliers/len(importance) > 0.03:
        dist_score -= 0.5
    
    score =  max(dist_score,1)
    properties = {"n_outliers":  info("number of outliers in the importance distribution",n_outliers),
                  "pct_dist":  info("percentage of feature that make up over 60% of all features importance", "{:.2f}%".format(100*pct_dist)),
                  "importance":  info("feature importance", {"value": importance, "labels": feat_labels})
                  }
    
    return result(score=score, properties=properties)
    # import seaborn as sns
    # sns.boxplot(data=importance)
    
def calc_explainability_score(clf, train_data, test_data, config):
    
    #function parameters
    clf_type_score = config["parameters"]["score_Algorithm_Class"]["clf_type_score"]["value"]
    scale_factor = config["parameters"]["score_Feature_Relevance"]["scale_factor"]["value"]
    distri_threshold = config["parameters"]["score_Feature_Relevance"]["distri_threshold"]["value"]
    
    output = dict(
        Algorithm_Class     = score_Algorithm_Class(clf, clf_type_score),
        Correlated_Features = score_Correlated_Features(train_data, test_data),
        Model_Size          = score_Model_Size(clf),
        Feature_Relevance   = score_Feature_Relevance(clf, train_data, scale_factor)
                 )
    
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)