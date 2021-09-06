DEBUG = True # Turns on debugging features in Flask
BCRYPT_LOG_ROUNDS = 12 # Configuration for the Flask-Bcrypt extension
MAIL_FROM_EMAIL = "robert@example.com" # For use in application emails
SCENARIOS_FOLDER_PATH = "scenarios"
TRAINING_DATA_FILE_NAME_REGEX = "train.*"
TEST_DATA_FILE_NAME_REGEX = "test.*"
MODEL_REGEX = "model.*"
PICKLE_FILE_EXTENSIONS = [".sav", ".pkl", ".pickle"]
JOBLIB_FILE_EXTENSIONS = [".joblib"]

# === FACTSHEET ===
FACTSHEET_NAME = "factsheet.json"
GENERAL_INPUTS = ["model_name", "purpose_description", "domain_description", "training_data_description", "model_information",    "data_normalization", "target_column", "contact_information"]
FAIRNESS_INPUTS = ["protected_feature", "privileged_class_definition"]
EXPLAINABILITY_INPUTS = ["protected_feature", "privileged_class_definition"]
ROBUSTNESS_INPUTS = []
METHODOLOGY_INPUTS = ["data_normalization", "regularization", "missing_data"]


SCENARIO_DESCRIPTION_FILE = "description.md"
SCENARIO_LINK_FILE = "link.md"

SOLUTIONS_FOLDER = "solutions"

# If no target column name is given, we assume 
# that the last column to the right is containing the label (or predicted value)
DEFAULT_TARGET_COLUMN_INDEX = -1

# === COLORS ===
PRIMARY_COLOR = '#000080'
SECONDARY_COLOR = '#EEEEEE'
TERTIARY_COLOR = '#1a1a1a'
TRUST_COLOR = '#1a1a1a'
FAIRNESS_COLOR = '#06d6a0'
EXPLAINABILITY_COLOR = '#ffd166'
ROBUSTNESS_COLOR = '#ef476f'
METHODOLOGY_COLOR = '#118ab2'

# === CONFIGURATION ===
METRICS_CONFIG_PATH = "configs/metrics"
DEFAULT_METRICS_FILE ="default.json"
WEIGHTS_CONFIG_PATH = "configs/weights"
DEFAULT_WEIGHTS_FILE = "default.json"

XAXIS_TICKANGLE = 30

NOT_SPECIFIED = "not specified"
NO_DETAILS = "No details available."
NO_SCORE = "X"





