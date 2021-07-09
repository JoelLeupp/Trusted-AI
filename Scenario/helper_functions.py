# -*- coding: utf-8 -*-
import json
import os
import pickle
import matplotlib.pyplot as plt
from math import pi
import numpy as np
import sklearn.metrics as metrics

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