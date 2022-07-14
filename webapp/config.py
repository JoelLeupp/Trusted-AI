import sys

DEBUG = True # Turns on debugging features in Flask
BCRYPT_LOG_ROUNDS = 12 # Configuration for the Flask-Bcrypt extension
MAIL_FROM_EMAIL = "robert@example.com" # For use in application emails

PORT = 8080
HOST= '0.0.0.0'

if len(sys.argv) > 1:
    BASE_PATH = '/trusted-ai'
    DEBUG = False
else:
    BASE_PATH = ''
    DEBUG = False
print("DEBUG: {}".format(DEBUG))
print("BASE_PATH: {}".format(BASE_PATH))

SCENARIOS_FOLDER_PATH = "scenarios"
SOLUTIONS_FOLDER = "solutions"
SOLUTIONS_FOLDER_UNSUPERVISED = "solutions_unsupervised"
TRAINING_DATA_FILE_NAME_REGEX = "train.*"
TEST_DATA_FILE_NAME_REGEX = "test.*"
MODEL_REGEX = "model.*"
PICKLE_FILE_EXTENSIONS = [".sav", ".pkl", ".pickle"]
JOBLIB_FILE_EXTENSIONS = [".joblib"]

PILLARS = ['fairness', 'explainability', 'robustness', 'methodology']
SECTIONS = ['trust'] + PILLARS

# === FACTSHEET ===
FACTSHEET_SECTIONS = "general", "fairness", "explainability", "robustness", "methodology"
FACTSHEET_NAME = "factsheet.json"
GENERAL_INPUTS = ["model_name", "purpose_description", "domain_description", "training_data_description", "model_information", "authors", "contact_information"]
FAIRNESS_INPUTS = ["question_fairness", "protected_feature", "protected_group", "favorable_outcome"]
EXPLAINABILITY_INPUTS = ["protected_feature", "privileged_class_definition"]
ROBUSTNESS_INPUTS = []
METHODOLOGY_INPUTS = ["regularization"]

FAVORABLE_OUTCOME_DEFINITION_EXAMPLE = "e.g. lambda x: x[target_column] == 1"
PROTECTED_GROUP_DEFINITION_EXAMPLE = "e.g. lambda x: x[protected_feature] < 25"


SCENARIO_DESCRIPTION_FILE = "description.md"
SCENARIO_LINK_FILE = "link.md"

# If no target column name is given, we assume 
# that the last column to the right is containing the label (or predicted value)
DEFAULT_TARGET_COLUMN_INDEX = -1
DEFAULT_TARGET_COLUMN_NAME = 'target_column'

# === COLORS ===
PRIMARY_COLOR = '#000080'
SECONDARY_COLOR = '#EEEEEE'
TERTIARY_COLOR = '#1a1a1a'
TRUST_COLOR = '#1a1a1a'
FAIRNESS_COLOR = '#06d6a0'
EXPLAINABILITY_COLOR = '#ffd166'
ROBUSTNESS_COLOR = '#ef476f'
METHODOLOGY_COLOR = '#118ab2'
CONFIG_COLOR = "rgba(255,228,181,0.5)"

# === CONFIGURATION ===
METRICS_CONFIG_PATH = "configs/metrics"
METRICS_CONFIG_PATH_UNSUPERVISED = "configs_unsupervised/metrics"
DEFAULT_METRICS_FILE ="default.json"
WEIGHTS_CONFIG_PATH = "configs/weights"
DEFAULT_WEIGHTS_FILE = "default.json"

XAXIS_TICKANGLE = 30

NOT_SPECIFIED = "not specified"
NO_DETAILS = "No details available."
NO_SCORE = "X"
NO_SCORE_FULL = "(X/5)"

# === METRICS ===
import os
import json

def list_of_metrics(pillar):
    metrics = []
    with open(os.path.join(METRICS_CONFIG_PATH, "config_{}.json".format(pillar))) as file:
        config_file = json.load(file)
        for metric_name in config_file["weights"]:
            metrics.append(metric_name.lower())
    return metrics

FAIRNESS_METRICS = list_of_metrics("fairness")
EXPLAINABILITY_METRICS = list_of_metrics("explainability")
ROBUSTNESS_METRICS = list_of_metrics("robustness")
METHODOLOGY_METRICS = list_of_metrics("methodology")

def list_of_metrics_unsupervised(pillar):
    metrics = []
    with open(os.path.join(METRICS_CONFIG_PATH_UNSUPERVISED, "config_{}.json".format(pillar))) as file:
        config_file = json.load(file)
        for metric_name in config_file["weights"]:
            metrics.append(metric_name.lower())
    return metrics

FAIRNESS_METRICS_UNSUPERVISED = list_of_metrics_unsupervised("fairness")
EXPLAINABILITY_METRICS_UNSUPERVISED = list_of_metrics_unsupervised("explainability")
ROBUSTNESS_METRICS_UNSUPERVISED = list_of_metrics_unsupervised("robustness")
METHODOLOGY_METRICS_UNSUPERVISED = list_of_metrics_unsupervised("methodology")