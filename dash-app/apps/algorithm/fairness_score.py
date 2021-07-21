# -*- coding: utf-8 -*-
# functions for fairness score

import numpy as np
import collections

result = collections.namedtuple('result', 'score properties')

def score_Statistical_Parity():
    return result(score=np.random.randint(1,6), properties={}) 

def score_Disparate_Mistreatment():
    return result(score=np.random.randint(1,6), properties={}) 

def score_Class_Imbalance():
    return result(score=np.random.randint(1,6), properties={}) 

def score_Biased_Data():
    return result(score=np.random.randint(1,6), properties={}) 

def calc_fairness_score():
    
    output = dict(
        Statistical_Parity     = score_Statistical_Parity(),
        Disparate_Mistreatment = score_Disparate_Mistreatment(),
        Disparate_Treatment = score_Disparate_Mistreatment(),
        Disparate_Impact = score_Disparate_Mistreatment(),
        Class_Imbalance        = score_Class_Imbalance(),
        Biased_Data            = score_Biased_Data()
                 )
    
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)
    
