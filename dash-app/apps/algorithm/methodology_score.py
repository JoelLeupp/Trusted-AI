# -*- coding: utf-8 -*-
import numpy as np
import collections

result = collections.namedtuple('result', 'score properties')

# functions for methodology score

def score_Normalization():
    return result(score=np.random.randint(1,6), properties={}) 

def score_Test_F1_score():
    return result(score=np.random.randint(1,6), properties={}) 

def score_Train_Test_Split():
    return result(score=np.random.randint(1,6), properties={}) 

def score_Regularization():
    return result(score=np.random.randint(1,6), properties={}) 

def score_Test_Accuracy():
    return result(score=np.random.randint(1,6), properties={}) 

def calc_methodology_score():
    
    output = dict(
        Normalization    = score_Normalization(),
        Treatment_of_Corrupt_Values     = score_Test_F1_score(),
        Train_Test_Split = score_Train_Test_Split(),
        Regularization   = score_Regularization(),
        Feature_Filtering    = score_Test_Accuracy(),
        Treatment_of_Categorical_Features  = score_Test_Accuracy()
                 )
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)