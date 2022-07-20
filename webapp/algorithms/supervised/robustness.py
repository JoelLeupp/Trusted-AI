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
def analyse(model, train_data, test_data, config, factsheet):
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

    clique_method_thresholds = config["score_clique_method"]["thresholds"]["value"]
    clever_score_thresholds = config["score_clever_score"]["thresholds"]["value"]
    loss_sensitivity_thresholds = config["score_loss_sensitivity"]["thresholds"]["value"]
    confidence_score_thresholds = config["score_confidence_score"]["thresholds"]["value"]
    fsg_attack_thresholds = config["score_fast_gradient_attack"]["thresholds"]["value"]
    cw_attack_thresholds = config["score_carlini_wagner_attack"]["thresholds"]["value"]
    deepfool_thresholds = config["score_carlini_wagner_attack"]["thresholds"]["value"]
    
    output = dict(
        confidence_score   = confidence_score(model, train_data, test_data, confidence_score_thresholds),
        clique_method      = clique_method(model, train_data, test_data, clique_method_thresholds, factsheet ),
        loss_sensitivity   = loss_sensitivity_score(model, train_data, test_data, loss_sensitivity_thresholds),
        clever_score       = clever_score(model, train_data, test_data, clever_score_thresholds),
        er_fast_gradient_attack = fast_gradient_attack_score(model, train_data, test_data, fsg_attack_thresholds),
        er_carlini_wagner_attack = carlini_wagner_attack_score(model, train_data, test_data, cw_attack_thresholds),
        er_deepfool_attack = deepfool_attack_score(model, train_data, test_data, deepfool_thresholds)
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
        X_test = test_data.iloc[:,:-1]
        X_train = train_data.iloc[:, :-1]
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

def loss_sensitivity_score(model, train_data, test_data, thresholds):
    """For a given Keras-NN model this function calculates the Loss Sensitivity score.
    It uses loss_sensitivity function from IBM art library.
    Returns a score according to the thresholds.
        Args:
            model: ML-model (Keras).
            train_data: pd.DataFrame containing the data.
            test_data: pd.DataFrame containing the data.
            threshold: list of threshold values

        Returns:
            Loss Sensitivity score
    """
    try:
        X_test = test_data.iloc[:,:-1]
        X_test = np.array(X_test)
        y = model.predict(X_test)

        classifier = KerasClassifier(model=model, use_logits=False)
        l_s = loss_sensitivity(classifier, X_test, y)
        score = np.digitize(l_s, thresholds, right=True) + 1
        return result(score=int(score), properties={"loss_sensitivity": info("Average gradient value of the loss function", "{:.2f}".format(l_s)),
                                                    "depends_on": info("Depends on", "Model")})
    except Exception as e:
        print(e)
        return result(score=np.nan, properties={"non_computable": info("Non Computable Because",
                                                                       "Can only be calculated on Keras models.")})

def confidence_score(model, train_data, test_data, thresholds):
    """For a given model this function calculates the Confidence score.
    It takes the average over confusion_matrix. Then returns a score according to the thresholds.
        Args:
            model: ML-model.
            train_data: pd.DataFrame containing the data.
            test_data: pd.DataFrame containing the data.
            threshold: list of threshold values

        Returns:
            Confidence score
        """
    try:
        X_test = test_data.iloc[:,:-1]
        y_test = test_data.iloc[:,-1: ]
        y_pred = model.predict(X_test)

        confidence = metrics.confusion_matrix(y_test, y_pred)/metrics.confusion_matrix(y_test, y_pred).sum(axis=1) 
        confidence_score = np.average(confidence.diagonal())*100
        score = np.digitize(confidence_score, thresholds, right=True) + 1
        return result(score=int(score), properties={"confidence_score": info("Average confidence score", "{:.2f}%".format(confidence_score)),
                                                    "depends_on": info("Depends on", "Model and Data")})
    except:
        return result(score=np.nan, properties={"non_computable": info("Non Computable Because", "Can only be calculated on models which provide prediction probabilities.")})

def clique_method(model, train_data, test_data, thresholds, factsheet):
    """For a given tree-based model this function calculates the Clique score.
    First checks the factsheet to see if the score is already calculated.
    If not it uses RobustnessVerificationTreeModelsCliqueMethod function from
    IBM art library to calculate the score. Returns a score according to the thresholds.

    Args:
        model: ML-model (Tree-based).
        train_data: pd.DataFrame containing the data.
        test_data: pd.DataFrame containing the data.
        threshold: list of threshold values
        factsheet: factsheet dict

    Returns:
        Clique score
        Error bound
        Error
    """
    with open('configs/supervised/mappings/robustness/default.json', 'r') as f:
          default_map = json.loads(f.read())
    
    if thresholds == default_map["score_clique_method"]["thresholds"]["value"]:
        if "scores" in factsheet.keys() and "properties" in factsheet.keys():
            score = factsheet["scores"]["robustness"]["clique_method"]
            properties = factsheet["properties"]["robustness"]["clique_method"]
            return result(score=score, properties=properties)
    
    try:
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1:]
        classifier = SklearnClassifier(model)
        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=classifier, verbose=True)

        bound, error = rt.verify(x=X_test.to_numpy()[100:103], y=y_test[100:103].to_numpy(), eps_init=0.5, norm=1,
                                 nb_search_steps=5, max_clique=2, max_level=2)
        score = np.digitize(bound, thresholds) + 1
        return result(score=int(score), properties={
            "error_bound": info("Average error bound", "{:.2f}".format(bound)),
            "error": info("Error", "{:.1f}".format(error)),
            "depends_on": info("Depends on", "Model")
        })
    except:
        return result(score=np.nan, properties={"non_computable": info("Non Computable Because", "Can only be calculated on Tree-Based models.")})

def fast_gradient_attack_score(model, train_data, test_data, thresholds):
    """For a given model this function calculates the fast gradient attack score.
    First from the test data selects a random small test subset.
    Then measures the accuracy of the model on this subset.
    Next creates FSG attacks on this test set and measures the model's
    accuracy on the attacks. Compares the before attack and after attack accuracies.
    Returns a score according to the thresholds.

    Args:
        model: ML-model (Logistic Regression, SVM).
        train_data: pd.DataFrame containing the data.
        test_data: pd.DataFrame containing the data.
        threshold: list of threshold values

    Returns:
        FSG attack score
        FSG Before attack accuracy
        FSG After attack accuracy
    """
    try:
        randomData = test_data.sample(50)
        randomX = randomData.iloc[:,:-1]
        randomY = randomData.iloc[:,-1: ]

        y_pred = model.predict(randomX)
        before_attack = metrics.accuracy_score(randomY,y_pred)

        classifier = SklearnClassifier(model=model)
        attack = FastGradientMethod(estimator=classifier, eps=0.2)
        x_test_adv = attack.generate(x=randomX)

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(test_data.iloc[:,-1: ])
        randomY = enc.transform(randomY).toarray()

        predictions = model.predict(x_test_adv)
        predictions = enc.transform(predictions.reshape(-1,1)).toarray()
        after_attack = metrics.accuracy_score(randomY,predictions)

        print("Accuracy on before_attacks: {}%".format(before_attack * 100))
        print("Accuracy on after_attack: {}%".format(after_attack * 100))

        score = np.digitize((before_attack - after_attack)/before_attack*100, thresholds) + 1
        return result(score=int(score), properties={"before_attack": info("FGM Before attack accuracy", "{:.2f}%".format(100 * before_attack)),
                                  "after_attack": info("FGM After attack accuracy", "{:.2f}%".format(100 * after_attack)),
                                  "difference": info("FGM Proportional difference (After-Att Acc - Before-Att Acc)/Before-Att Acc", "{:.2f}%".format(100 * (before_attack - after_attack) / before_attack)),
                                  "depends_on": info("Depends on", "Model and Data")})
    except:
        return result(score=np.nan, properties={"non_computable": info("Non Computable Because",
                                                                       "Can be calculated on either SVC or Logistic Regression models.")})

def carlini_wagner_attack_score(model, train_data, test_data, thresholds):
    """For a given model this function calculates the CW attack score.
    First from the test data selects a random small test subset.
    Then measures the accuracy of the model on this subset.
    Next creates CW attacks on this test set and measures the model's
    accuracy on the attacks. Compares the before attack and after attack accuracies.
    Returns a score according to the thresholds.

    Args:
        model: ML-model (Logistic Regression, SVM).
        train_data: pd.DataFrame containing the data.
        test_data: pd.DataFrame containing the data.
        threshold: list of threshold values

    Returns:
        CW attack score
        CW Before attack accuracy
        CW After attack accuracy
    """
    try:
        randomData = test_data.sample(5)
        randomX = randomData.iloc[:,:-1]
        randomY = randomData.iloc[:,-1: ]

        y_pred = model.predict(randomX)
        before_attack = metrics.accuracy_score(randomY,y_pred)

        classifier = SklearnClassifier(model=model)
        attack = CarliniL2Method(classifier)
        x_test_adv = attack.generate(x=randomX)

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(test_data.iloc[:,-1: ])
        randomY = enc.transform(randomY).toarray()

        predictions = model.predict(x_test_adv)
        predictions = enc.transform(predictions.reshape(-1,1)).toarray()
        after_attack = metrics.accuracy_score(randomY,predictions)
        print("Accuracy on before_attacks: {}%".format(before_attack * 100))
        print("Accuracy on after_attack: {}%".format(after_attack * 100))
        score = np.digitize((before_attack - after_attack)/before_attack*100, thresholds) + 1
        return result(score=int(score),
                      properties={
                          "before_attack": info("CW Before attack accuracy", "{:.2f}%".format(100 * before_attack)),
                          "after_attack": info("CW After attack accuracy", "{:.2f}%".format(100 * after_attack)),
                          "difference": info(
                              "CW Proportional difference (After-Att Acc - Before-Att Acc)/Before-Att Acc",
                              "{:.2f}%".format(100 * (before_attack - after_attack) / before_attack)),
                          "depends_on": info("Depends on", "Model and Data")})
    except:
        return result(score=np.nan, properties={"non_computable": info("Non Computable Because",
                                                                       "Can be calculated on either SVC or Logistic Regression models.")})

def deepfool_attack_score(model, train_data, test_data, thresholds):
    """For a given model this function calculates the deepfool attack score.
    First from the test data selects a random small test subset.
    Then measures the accuracy of the model on this subset.
    Next creates deepfool attacks on this test set and measures the model's
    accuracy on the attacks. Compares the before attack and after attack accuracies.
    Returns a score according to the thresholds.

    Args:
        model: ML-model (Logistic Regression, SVM).
        train_data: pd.DataFrame containing the data.
        test_data: pd.DataFrame containing the data.
        threshold: list of threshold values

    Returns:
        Deepfool attack score
        DF Before attack accuracy
        DF After attack accuracy
    """
    try:
        randomData = test_data.sample(4)
        randomX = randomData.iloc[:,:-1]
        randomY = randomData.iloc[:,-1: ]

        y_pred = model.predict(randomX)
        before_attack = metrics.accuracy_score(randomY,y_pred)

        classifier = SklearnClassifier(model=model)
        attack = DeepFool(classifier)
        x_test_adv = attack.generate(x=randomX)

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(test_data.iloc[:,-1: ])
        randomY = enc.transform(randomY).toarray()

        predictions = model.predict(x_test_adv)
        predictions = enc.transform(predictions.reshape(-1,1)).toarray()
        after_attack = metrics.accuracy_score(randomY,predictions)
        print("Accuracy on before_attacks: {}%".format(before_attack * 100))
        print("Accuracy on after_attack: {}%".format(after_attack * 100))

        score = np.digitize((before_attack - after_attack)/before_attack*100, thresholds) + 1
        return result(score=int(score),
                      properties={"before_attack": info("DF Before attack accuracy", "{:.2f}%".format(100 * before_attack)),
                                  "after_attack": info("DF After attack accuracy", "{:.2f}%".format(100 * after_attack)),
                                  "difference": info("DF Proportional difference (After-Att Acc - Before-Att Acc)/Before-Att Acc", "{:.2f}%".format(100 * (before_attack - after_attack) / before_attack)),
                                  "depends_on": info("Depends on", "Model and Data")})
    except:
        return result(score=np.nan, properties={"non_computable": info("Non Computable Because",
                                                                       "Can be calculated on either SVC or Logistic Regression models.")})