# -*- coding: utf-8 -*-
import numpy as np
import collections
import random
import sklearn.metrics as metrics
from art.attacks.evasion import FastGradientMethod, CarliniL2Method, DeepFool
from art.estimators.classification import SklearnClassifier
from sklearn.preprocessing import OneHotEncoder
from art.metrics import clever_u, RobustnessVerificationTreeModelsCliqueMethod
from art.estimators.classification import KerasClassifier
from art.metrics import loss_sensitivity
import tensorflow as tf
import numpy.linalg as la
import json

info = collections.namedtuple('info', 'description value')
result = collections.namedtuple('result', 'score properties')

# === ROBUSTNESS ===
def analyse(model, train_data, test_data, outliers_data, config, factsheet):
    """Reads the thresholds from the config file.
    Calls all robustness metric functions with correct arguments.
    Organizes all robustness metrics in a dict. Then returns the scores and the properties.
        Args:
            model: ML-model.
            training_dataset: pd.DataFrame containing the used training data.
            test_dataset: pd.DataFrame containing the used test data.
            config: Config file containing the threshold values for the metrics.
            factsheet: json document containing all information about the particular solution.

        Returns:
            Returns a result object containing all metric scores
            and matching properties for every metric
    """

    clever_score_thresholds = config["score_clever_score"]["thresholds"]["value"]
    
    output = dict(
        clever_score = clever_score(model, train_data, test_data, clever_score_thresholds),
        #clever_score = result(score=int(1), properties={}),
    )
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)


def clever_score(model, train_data, test_data, thresholds):
    """For a given Keras-NN model this function calculates the Untargeted-Clever score.
    It uses clever_u function from IBM art library.
    Returns a score according to the thresholds.
        Args:
            model: ML-model (Keras).
            train_data: pd.DataFrame containing the data.
            test_data: pd.DataFrame containing the data.
            threshold: list of threshold values

        Returns:
            Clever score
    """
    try:
        classifier = KerasClassifier(model, False)

        min_score = 100

        randomX = X_test.sample(10)
        randomX = np.array(randomX)

        for x in randomX:
            temp = clever_u(classifier=classifier, x=x, nb_batches=1, batch_size=1, radius=500, norm=1)
            if min_score > temp:
                min_score = temp
        score = np.digitize(min_score, thresholds) + 1
        return result(score=int(score), properties={"clever_score": info("CLEVER Score", "{:.2f}".format(min_score)),
                                                    "depends_on": info("Depends on", "Model")})
    except Exception as e:
        print(e)
        return result(score=np.nan, properties={"non_computable": info("Non Computable Because",
                                                                       "Can only be calculated on Keras models.")})