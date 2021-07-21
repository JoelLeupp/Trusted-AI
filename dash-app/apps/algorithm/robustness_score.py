# -*- coding: utf-8 -*-
import numpy as np
import collections

result = collections.namedtuple('result', 'score properties')

# functions for robustness score

def score_Confidence_Score():
    return result(score=np.random.randint(1,6), properties={}) 

def score_Class_Specific_Metrics():
    return result(score=np.random.randint(1,6), properties={}) 

def calc_robustness_score():
    
    output = dict(
        Confidence_Score          = score_Confidence_Score(),
        Clique_Method    = score_Class_Specific_Metrics(),
        Loss_Sensitivity   = score_Class_Specific_Metrics(),
        CLEVER_Score   = score_Class_Specific_Metrics()
                 )
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)
