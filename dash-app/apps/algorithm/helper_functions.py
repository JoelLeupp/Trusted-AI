# -*- coding: utf-8 -*-
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
from apps.algorithm.fairness_score import calc_fairness_score
from apps.algorithm.explainability_score import calc_explainability_score
from apps.algorithm.robustness_score import calc_robustness_score
from apps.algorithm.methodology_score import calc_methodology_score

def get_perfomance_metric(clf,test_data):
    
    X_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1: ]
    
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


## get inputs
def get_case_inputs(case):
    scenario_path = os.path.join("Validation", case)

    # with open(os.path.join(scenario_path, "factsheet.txt")) as file:
    #     factsheet = json.load(file)
        
    with open(os.path.join(scenario_path, "model.sav"),'rb') as file:
        model = pickle.load(file)

    with open(os.path.join(scenario_path, "train_data.pkl"),'rb') as file:
        train_data = pickle.load(file)

    with open(os.path.join(scenario_path, "test_data.pkl"),'rb') as file:
        test_data = pickle.load(file)

    return [model, train_data, test_data]


def draw_spider(categories, values, ax, color='lightblue', title='Trusting AI Final Score'):
    
    values += values[:1]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.xaxis.set_tick_params(pad=6)
    # Add a title
    plt.title(title, size=11, y=1.1,
             bbox=dict(facecolor=color, edgecolor=color, pad=2.0))

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1,2,3], ["1","2","3"], color="grey", size=7)
    plt.ylim(0,4)

    # Plot data
    ax.plot(angles, values,color=color, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, color=color, alpha=0.4)
    
def draw_bar_plot(categories, values, ax, color='lightblue', title='Trusting AI Final Score',size=12):
    
    # drop top and right spine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #plt.grid(True,axis='y',zorder=0)
    
    # create barplot
    x_pos = np.arange(len(categories))
    plt.bar(x_pos, values,zorder=4,color=color)

    for i, v in enumerate(values):
        plt.text(i , v+0.1 , str(v),color='dimgray', fontweight='bold',ha='center')

    # Create names on the x-axis
    plt.xticks(x_pos, categories,size=size,wrap=True)

    plt.yticks([1,2,3,4], ["1","2","3","4"], size=12)
    plt.ylim(0,4)
    
    if isinstance(color, list):
        plt.title(title, size=11, y=1.1)
    else:
        plt.title(title, size=11, y=1.1,
             bbox=dict(facecolor=color, edgecolor=color, pad=2.0))
        plt.title(title, size=11, y=1.1)
        
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
def get_final_score(model, train_data, test_data, config_fairness, config_explainability, config_robustness, config_methodology):
    scores = trusting_AI_scores(model, train_data, test_data, config_fairness, config_explainability, config_robustness, config_methodology)
    final_scores = dict()
    for pillar, item in scores.items():
        config = eval("config_"+pillar)
        weighted_scores = list(map(lambda x: scores[pillar][x] * config["weights"][x], scores[pillar].keys()))
        final_scores[pillar] = round(np.sum(weighted_scores),1)

    return final_scores, scores
        
### delete later
# define model inputs
# choose scenario case (case1,case1,..)
# case = "case1"
# np.random.seed(6)

# # load case inputs
# model, train_data, test_data = get_case_inputs(case)

# config_fairness, config_explainability, config_robustness, config_methodology = 0, 0, 0 ,0
# for config in ["config_fairness", "config_explainability", "config_robustness", "config_methodology"]:
#     with open("apps/algorithm/"+config+".json") as file:
#             exec("%s = json.load(file)" % config)

# trusting_AI_scores(model, train_data, test_data, config_fairness, config_explainability, config_robustness, config_methodology)
# final_scores, scores = get_final_score(model, train_data, test_data, config_fairness, config_explainability, config_robustness, config_methodology)
# final_scores


def get_performance_table(model, test_data):
    performance_metrics = get_perfomance_metric(model, test_data)
    df = pd.DataFrame(performance_metrics, index=["Performance Metrics"]).transpose()
    df["Performance Metrics"] = df["Performance Metrics"].round(2)
    df = df.loc[["accuracy","class weighted recall","class weighted precision","class weighted f1 score","cross-entropy loss","ROC AUC"]]
    return df