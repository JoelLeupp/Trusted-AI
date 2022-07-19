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
from config import *
from math import pi
import sklearn.metrics as metrics
from algorithms.fairness import analyse as analyse_fairness
from algorithms.explainability import analyse as analyse_explainability
from algorithms.robustness import analyse as analyse_robustness
from algorithms.methodology import analyse as analyse_methodology
import collections
from helpers import *

# define algo
def trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, methodology_config, solution_set_path):
    # if "scores" in factsheet.keys() and "properties" in factsheet.keys():
    #     scores = factsheet["scores"]
    #     properties = factsheet["properties"]
    # else:
        output = dict(
            fairness       = analyse_fairness(model, train_data, test_data, factsheet, config_fairness),
            explainability = analyse_explainability(model, train_data, test_data, config_explainability, factsheet),
            robustness     = analyse_robustness(model, train_data, test_data, config_robustness, factsheet),
            methodology    = analyse_methodology(model, train_data, test_data, factsheet, methodology_config)
        )
        scores = dict((k, v.score) for k, v in output.items())
        properties = dict((k, v.properties) for k, v in output.items())
        # factsheet["scores"] = scores
        # factsheet["properties"] = properties
        # write_into_factsheet(factsheet, solution_set_path)
    
        return  result(score=scores, properties=properties)

# calculate final score with weigths
def get_final_score(model, train_data, test_data, config_weights, mappings_config, factsheet, solution_set_path, recalc=False):
    config_fairness = mappings_config["fairness"]
    config_explainability = mappings_config["explainability"]
    config_robustness = mappings_config["robustness"]
    config_methodology = mappings_config["methodology"]
    
    with open('configs/supervised/mappings/default.json', 'r') as f:
          default_map = json.loads(f.read())
    #print("mapping is default:")
    #print(default_map == mappings_config)
    if default_map == mappings_config:
        if "scores" in factsheet.keys() and "properties" in factsheet.keys() and not recalc:
            scores = factsheet["scores"]
            properties = factsheet["properties"]
        else:
            result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, config_methodology, solution_set_path)
            scores = result.score
            factsheet["scores"] = scores
            properties = result.properties
            factsheet["properties"] = properties
            try:
                write_into_factsheet(factsheet, solution_set_path)
            except Exception as e:
                print("ERROR in write_into_factsheet: {}".format(e))
    else:
        result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, config_methodology, solution_set_path)
        scores = result.score
        properties = result.properties
    
    final_scores = dict()
    for pillar, item in scores.items():
        config = config_weights[pillar]
        weighted_scores = list(map(lambda x: scores[pillar][x] * config[x], scores[pillar].keys()))
        sum_weights = np.nansum(np.array(list(config.values()))[~np.isnan(weighted_scores)])
        if sum_weights == 0:
            result = 0
        else:
            result = round(np.nansum(weighted_scores)/sum_weights,1)
        final_scores[pillar] = result

    return final_scores, scores, properties

def get_final_score_unsupervised(model, train_data, test_data, outliers_data, config_weights, mappings_config, factsheet, solution_set_path, recalc=False):
    config_fairness = mappings_config["fairness"]
    config_explainability = mappings_config["explainability"]
    config_robustness = mappings_config["robustness"]
    config_methodology = mappings_config["methodology"]

    with open('configs/unsupervised/mappings/default.json', 'r') as f:
        default_map = json.loads(f.read())
    # print("mapping is default:")
    # print(default_map == mappings_config)
    if default_map == mappings_config:
        if "scores" in factsheet.keys() and "properties" in factsheet.keys() and not recalc:
            scores = factsheet["scores"]
            properties = factsheet["properties"]
        else:
            result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability,
                                        config_robustness, config_methodology, solution_set_path)
            scores = result.score
            factsheet["scores"] = scores
            properties = result.properties
            factsheet["properties"] = properties
            try:
                write_into_factsheet(factsheet, solution_set_path)
            except Exception as e:
                print("ERROR in write_into_factsheet: {}".format(e))
    else:
        result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability,
                                    config_robustness, config_methodology, solution_set_path)
        scores = result.score
        properties = result.properties

    final_scores = dict()
    for pillar, item in scores.items():
        config = config_weights[pillar]
        weighted_scores = list(map(lambda x: scores[pillar][x] * config[x], scores[pillar].keys()))
        sum_weights = np.nansum(np.array(list(config.values()))[~np.isnan(weighted_scores)])
        if sum_weights == 0:
            result = 0
        else:
            result = round(np.nansum(weighted_scores) / sum_weights, 1)
        final_scores[pillar] = result

    return final_scores, scores, properties

def get_trust_score(final_score, config):
    if sum(config.values()) == 0:
        return 0
    return round(np.nansum(list(map(lambda x: final_score[x] * config[x], final_score.keys())))/np.sum(list(config.values())),1)