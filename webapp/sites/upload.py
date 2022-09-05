import dash_daq as daq
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import json
import pickle
import os
import io
import base64
from joblib import dump, load
from app import app
from config import SCENARIOS_FOLDER_PATH, PICKLE_FILE_EXTENSIONS, JOBLIB_FILE_EXTENSIONS, FACTSHEET_NAME, BASE_PATH
from helpers import * 
from sites import factsheet
from keras.models import load_model

# === CALLBACKS ===
# --- Preview Callbacks --- #
@app.callback([Output('training_data_upload', 'children'),
               Output('training_data_summary', 'children'), 
               Output('target_column_dropdown', 'options'),
               Output('protected_feature_dropdown', 'options')],
              [Input('training_data_upload', 'contents'),
              State('training_data_upload', 'filename'), State("toggle_supervised_unsupervised", "on")], prevent_initial_call=True)
def training_data_preview(content, name, unsupervised):

    message = html.Div(['Drag and Drop or Select File'])
    summary = []
    options = []
    if content is not None:
        print("IF")
        message = html.Div(name)
        df, summary, columns = parse_contents(content, name)
        print(columns)
        for c in columns:
            options.append({"label": str(c), "value": str(c)})

        return [message, summary, options, options]


@app.callback(Output('protected_value_dropdown_upload', 'options'),
              [Input('protected_feature_dropdown', 'value'),
               State('training_data_upload', 'contents'),
              State('training_data_upload', 'filename')], prevent_initial_call=True)
def protected_group_value_options(protected_feature, training_data_content, training_data_filename):
    options = []
    if training_data_content is not None:
        df, _, _ = parse_contents(training_data_content, training_data_filename)
        unique_protected_feature_values = np.unique(df[protected_feature])
        for unique_protected_feature_value in unique_protected_feature_values:
            options.append({"label": "{0}=={1}".format(protected_feature, str(unique_protected_feature_value)), "value": unique_protected_feature_value})
    return options

@app.callback(Output('favorable_outcome_dropdown_upload', 'options'),
              [Input('target_column_dropdown', 'value'),
               State('training_data_upload', 'contents'),
              State('training_data_upload', 'filename')], prevent_initial_call=True)
def favorable_outcome_value_options(column, content, filename):
    options = []
    if content is not None:
        df, _, _ = parse_contents(content, filename)
        unique_values = np.unique(df[column])
        for unique_value in unique_values:
            options.append({"label": "{0}=={1}".format(column, str(unique_value)), "value": unique_value})
    return options

@app.callback([Output('test_data_upload', 'children'),
               Output('test_data_summary', 'children')],
              [Input('test_data_upload', 'contents'),
              State('test_data_upload', 'filename')], prevent_initial_call=True)
def test_data_preview(content, name):
    message = html.Div(['Drag and Drop or Select File'])
    summary = []
    if content is not None:
        message = html.Div(name)
        df, summary, columns = parse_contents(content, name)
    return [message, summary]

@app.callback([Output('outlier_data_upload', 'children'),
               Output('outlier_data_summary', 'children')],
              [Input('outlier_data_upload', 'contents'),
              Input('outlier_data_upload', 'filename')], prevent_initial_call=True)
def outlier_data_preview(content, name):
    message = html.Div(['Drag and Drop or Select File'])
    summary = []
    if content is not None:
        message = html.Div(name)
        df, summary, columns = parse_contents(content, name)
    return [message, summary]

@app.callback([Output('factsheet_upload', 'children'),
               Output('factsheet_summary', 'children')],
              [Input('factsheet_upload', 'contents'),
              State('factsheet_upload', 'filename')], prevent_initial_call=True)
def factsheet_preview(content, name):
    if content is not None:
        message = html.Div(name)
        summary = html.Div()
        return [message, summary]
    return [html.Div(['Drag and Drop or Select File']), None]

@app.callback([Output('model_upload', 'children'),
               Output('model_summary', 'children')],
              [Input('model_upload', 'contents'),
              State('model_upload', 'filename')], prevent_initial_call=True)
def model_preview(content, name):
    if content is not None:
        message = html.Div(name)
        summary = html.Div()
        return [message, summary]
    return [html.Div(['Drag and Drop or Select File']), None]

# --- Validation Callbacks --- #
@app.callback(Output('scenario_alert', 'children'),
              [Input('upload_button', 'n_clicks'),
               State('upload_scenario_id', 'value'),
               ], prevent_initial_call=True)
def validate_scenario_id(n_clicks, scenario_id):
    print("scenario_id {}".format(scenario_id))
    if scenario_id is not None:
        return None
    else:
        return html.H6("No scenario was selected", style={"color":"Red"})
  
@app.callback(Output('solution_name_alert', 'children'),
              [Input('upload_button', 'n_clicks'),
               Input('upload_scenario_id', 'value'),
               Input('solution_name', 'value'),
               ],
              [State('toggle_supervised_unsupervised', 'on')], prevent_initial_call=True)
def validate_solution_name(n_clicks, scenario_id, solution_name, unsupervised):
    if n_clicks is not None:
        if not solution_name:
            return html.H6("Please enter a name for your model", style={"color":"Red"})
        else:
            # check if a model with this name already exists for this problem set
            solution_id = name_to_id(solution_name)
            if unsupervised:
                solution_path = get_solution_unsupervised_path(scenario_id, solution_id)
            else:
                solution_path = get_solution_path(scenario_id, solution_id)
            print("solution_path {}".format(solution_path))
            if os.path.isdir(solution_path):
                return html.H6("A model with this name already exists", style={"color":"Red"})
            else:  
                return None
            
@app.callback(Output('training_data_alert', 'children'),
               [Input('upload_button', 'n_clicks'),
                Input('training_data_upload', 'contents')], prevent_initial_call=True)
def validate_training_data(n_clicks, training_data):
    print("VALIDATE TRAIN")
    print(n_clicks)
    print(training_data)
    if n_clicks is not None:
        if training_data is None:
            print("NONE")

            return html.H6("No training data uploaded", style={"color":"Red"})
        else:
            print("ELSE")

            return None
        
@app.callback(Output('test_data_alert', 'children'),
               [Input('upload_button', 'n_clicks'),
                Input('test_data_upload', 'contents')], prevent_initial_call=True
                )
def validate_test_data(n_clicks, test_data):
    if n_clicks is not None:
        if test_data is None:
            return html.H6("No test data uploaded", style={"color":"Red"})
        else:
            return None

@app.callback(Output('outlier_data_alert', 'children'),
               [Input('upload_button', 'n_clicks'),
                Input('outlier_data_upload', 'contents')], prevent_initial_call=True
                )
def validate_outlier_data(n_clicks, outlier_data):
    if n_clicks is not None:
        if outlier_data is None:
            return html.H6("No outlier data uploaded", style={"color":"Red"})
        else:
            return None


@app.callback(Output('factsheet_alert', 'children'),
               [Input('upload_button', 'n_clicks'),
                Input('factsheet_upload', 'filename'),
                Input('factsheet_upload', 'contents')
               ], prevent_initial_call=True)
def validate_factsheet(n_clicks, factsheet_name, factsheet_content):
    if n_clicks is not None:
        if factsheet_content is None:
            return html.H6("No factsheet provided", style={"color":"Red"})
        else:
            file_name, file_extension = os.path.splitext(factsheet_name)
            if file_extension not in ['.json']:
                return html.H6("Please select a .json file", style={"color":"Red"})   
            return None
        
@app.callback(Output('model_alert', 'children'),
               [Input('upload_button', 'n_clicks'),
                Input('model_upload', 'contents')],
                prevent_initial_call=True)
def validate_model(n_clicks, model):
    if n_clicks is not None:
        if model is None:
            return html.H6("No model uploaded", style={"color":"Red"})
        else:
            return None

@app.callback([Output('upload_alert', 'children'),
              Output('uploaded_scenario_id', 'data'),
              Output('uploaded_solution_id', 'data')],
              [
               Input('upload_button', 'n_clicks'),
               Input('created_factsheet', 'data'),
               State('upload_scenario_id', 'value'),
               State('solution_name', 'value'),
               State('general_description', 'value'),
               State('training_data_upload', 'contents'),
               State('training_data_upload', 'filename'),
               State('test_data_upload', 'contents'),
               State('test_data_upload', 'filename'),
               State('outlier_data_upload', 'contents'),
               State('outlier_data_upload', 'filename'),
               State('protected_feature_dropdown', 'value'),
               State('protected_value_dropdown_upload', 'value'),
               State('target_column_dropdown', 'value'),
               State('favorable_outcome_dropdown_upload', 'value'),
               State('factsheet_upload', 'contents'),
               State('factsheet_upload', 'filename'),
               State('model_upload', 'contents'),
               State('model_upload', 'filename'),
               State('toggle_supervised_unsupervised', 'on'),
], prevent_initial_call=True)
def upload_data(
    n_clicks,
    created_factsheet,
    scenario_id,
    solution_name,
    general_description,
    training_data,
    training_data_filename,
    test_data,
    test_data_filename,
    outlier_data,
    outlier_data_filename,
    protected_feature,
    protected_values,
    target_column,
    favorable_outcomes,
    factsheet,
    factsheet_filename,
    model,
    model_filename,
    unsupervised):
    if n_clicks is None:
        return "", "", ""
    else:
        if None in (scenario_id, solution_name, training_data, test_data, model):
            return html.H5("Please provide all necessary data", style={"color":"Red"},  className="text-center"), '', ''
        else:
            # Create directory within the problem set to contain the data
            solution_id = name_to_id(solution_name)
            if unsupervised:
                solution_path = get_solution_unsupervised_path(scenario_id, solution_id)
            else:
                solution_path = get_solution_path(scenario_id, solution_id)
            # Check if directory does not exists yet
            if not os.path.isdir(solution_path):
                os.mkdir(solution_path)

                # Upload all the data to the new directory.
                # Saving Training Data
                save_training_data(solution_path, training_data_filename, training_data)

                # Saving Test Data
                save_test_data(solution_path, test_data_filename, test_data)

                if unsupervised:
                    save_outlier_data(solution_path, outlier_data_filename, outlier_data)

                if created_factsheet:
                    new_factsheet = created_factsheet
                else:
                    new_factsheet = {}
                if "general" not in new_factsheet:
                    new_factsheet["general"] = {}
                if "fairness" not in new_factsheet:
                    new_factsheet["fairness"] = {}
                new_factsheet["general"]["target_column"] = target_column
                new_factsheet["general"]["description"] = general_description

                new_factsheet["fairness"]["protected_feature"] = protected_feature
                new_factsheet["fairness"]["protected_values"] = protected_values
                new_factsheet["fairness"]["favorable_outcomes"] = favorable_outcomes

                # Saving Factsheet
                save_factsheet(solution_path, FACTSHEET_NAME, factsheet, new_factsheet)

                # Saving Model
                save_model(solution_path, model_filename, model)

            else:
                return html.H4("Directory already exists", style={"color":"Red"}, className="text-center"), '', ''

            return dcc.Location(pathname="{}/analyze".format(BASE_PATH), id="someid_doesnt_matter"), scenario_id, solution_id

modals = ["upload_scenario_id", "solution_name", "general_description", "training_data", "test_data", "target_column_name" ,"factsheet", "model"]
for m in modals:
    @app.callback(
        Output("{}_info_modal".format(m), "is_open"),
        [Input("{}_info_button".format(m), "n_clicks"), Input("{}_close".format(m), "n_clicks")],
        [State("{}_info_modal".format(m), "is_open")], prevent_initial_call=True
    )
    def toggle_input_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

def save_training_data(path, name, content):
    print("SAVE TRAIN")
    file_name, file_extension = os.path.splitext(name)
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(path, "train" + file_extension), "wb") as fp:
        fp.write(base64.decodebytes(data))
    
def save_test_data(path, name, content):
    file_name, file_extension = os.path.splitext(name)
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(path, "test" + file_extension), "wb") as fp:
        fp.write(base64.decodebytes(data))

def save_outlier_data(path, name, content):
    file_name, file_extension = os.path.splitext(name)
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(path, "outliers" + file_extension), "wb") as fp:
        fp.write(base64.decodebytes(data))
  
def save_model(path, name, content):
    file_name, file_extension = os.path.splitext(name)
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)

    print("*************************************", file_extension)
    if file_extension in PICKLE_FILE_EXTENSIONS:
        model = pd.read_pickle(io.BytesIO(decoded))
        pickle.dump(model, open(os.path.join(path, "model" + file_extension), 'wb'))
        
    elif file_extension in JOBLIB_FILE_EXTENSIONS:
        print("JOOOOOOOOOOOOOOBLIIIIIIIIIB")
        model = load(io.BytesIO(decoded))
        dump(model, os.path.join(path, "model" + file_extension))

    else:
        print("NOOOOOOOOOOOOOOOOOOOOOO")




@app.callback(
    [Output("upload_scenario_id", "options"),
     Output("conditional_outlier", "style"),
     Output("conditional_output_target", "style"),
     Output("conditional_output_fav_out", "style")],
    Input('toggle_supervised_unsupervised', 'on')
)
def toggle_mode_upload(unsupervised):
    display = {"display": "inline",
               'width': '100%',
               'height': '60px',
               'textAlign': 'center',
               }
    hide = {"display": "none"}
    if unsupervised:
        return get_scenario_options(unsupervised), display, hide, hide
    else:
        return get_scenario_options(unsupervised), hide, display, display


# === SITE ===
#scenarios = [{'label': f.name, 'value': f.path} for f in os.scandir(SCENARIOS_FOLDER_PATH) if f.is_dir()]

factsheet.get_factsheet_callbacks(app)
layout = dbc.Container([
    dbc.Col([html.H1("Upload", className="text-center")], width=12, className="mb-2 mt-1"),
    dbc.Col([
    html.Div([
        html.Div([
            create_info_modal("upload_scenario_id", "Scenario", "All different solutions found should belong to the same scenario", ""),
            html.Div(id="scenario_alert"),
            html.H3(["1. Scenario", html.Sup("*")]),
            html.H5("Please select the scenario your solution belongs to")
        ], className="text-center"),
        dcc.Dropdown(
            id='upload_scenario_id',
            options=get_scenario_options(),
        ),
    ], 
    className="mb-4"
    ),
    
    html.Div([
        html.Div([
            create_info_modal("solution_name", "Solution", "One specifically trained model including its training-, test data and factsheet are part of a solution. Your solution set will be saved under the name you entered here.", ""),
            html.Div(id="solution_name_alert"),
            html.H3(["2. Solution", html.Sup("*")]),
            html.H5("Please enter a name for your solution")
        ], 
        className="text-center"
        ),

        dcc.Input(id="solution_name", type="text", placeholder="", value="", debounce=True, style={'width': '100%', 'textAlign': 'center', 'backgroundColor': '#FFFFFF'})
    ], 
    className="mb-4"
    ),
    
    html.Div([
    html.Div([
        create_info_modal("general_description", "Description", "Please add a brief description for your solution. This description will be displayed to all users.", "*e.g Detect multiple objects within an image, with bounding boxes. The model is trained to recognize 80 different classes of objects in the COCO Dataset. The model consists of a deep convolutional net base model for image feature extraction, together with additional convolutional layers specialized for the task of object detection, that was trained on the COCO data set. It is based on SSD MobileNetV1 using the TensorFlow framework.*"),
        html.H3("3. Description", className="text-center"),
        html.H5("Please enter a description for your solution", className="text-center"),
        dcc.Textarea(
            id='general_description',
            value='',
            style={'width': '100%', 'height': 100},
        )], className="mb-4"),
    ]),
        # --- TRAIN DATA UPLOAD --- #

        html.Div([
            html.Div([
                create_info_modal("training_data", "Training Data",
                                  "Please upload the training data that was used to train the model. Csv and pickle (pkl) files are accepted.",
                                  ""),
                html.Div(id="training_data_alert"),
                html.H3(["4. Training Data", html.Sup("*")]),
                html.H5("Please upload the training data"),
                dcc.Upload(
                    id='training_data_upload',
                    children=[
                        'Drag and Drop or Select a File'
                    ],
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'backgroundColor': '#FFFFFF'
                    }
                ),
                html.Div(id='training_data_summary'),
                # (csv and pickle files are accepted).
                # "Please place the label to the last column of the dataframe."
            ], className="text-center"),
        ],
            className="mb-4"),
    
    # --- TEST DATA UPLOAD --- #
    
    html.Div([
        html.Div([
            create_info_modal("test_data", "Test Data", "Please upload the test data that was used to evaluate the model. Csv and pickle (pkl) files are accepted.", ""),
            html.Div(id="test_data_alert"),
            html.H3(["5. Test Data", html.Sup("*")]),
            html.H5("Please upload the test data"),
                dcc.Upload(
        id='test_data_upload',
        children=[
            'Drag and Drop or Select a File'
        ],
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'backgroundColor': '#FFFFFF'
        }
    ),
    html.Div(id='test_data_summary'),
            #(csv and pickle files are accepted).
            #"Please place the label to the last column of the dataframe."
        ], className="text-center"),
    ],
            className="mb-4"),

    # --- OUTLIER DATA UPLOAD --- #
    html.Div([
                 html.Div(
                 [
                     create_info_modal("outlier_data", "Outlier Data",
                                       "Please upload the outlier data. Csv and pickle (pkl) files are accepted.", ""),
                     html.Div(id="outlier_data_alert"),
                     html.Div([
                        html.H3(["6. Outlier Data", html.Sup("*")]),
                        html.H5("Please upload the outlier data"),
                        dcc.Upload(
                             id='outlier_data_upload',
                             children=[
                                 'Drag and Drop or Select a File'
                             ],
                             style={
                                 'width': '100%',
                                 'height': '60px',
                                 'lineHeight': '60px',
                                 'borderWidth': '1px',
                                 'borderStyle': 'dashed',
                                 'borderRadius': '5px',
                                 'textAlign': 'center',
                                 'backgroundColor': '#FFFFFF'
                             }
                         ),
                     ],id = "conditional_outlier"),

                     html.Div(id='outlier_data_summary'),
                 ], className="text-center")
             ]),

        # --- PROTECTED FEATURE --- #
        html.Div([
            create_info_modal("protected_feature", "Protected Feature", "A protected feature (like age, race, gender) is not supposed to influence he prediction.", ""),
            html.H3("Protected Feature"),
            html.H5("Please select the Protected Feature"),
            dcc.Dropdown(
                id='protected_feature_dropdown',
                options=[],
                placeholder='Select Protected Feature'
            ),
        ], className="mb-4 mt-4 text-center"),
            
        # --- PROTECTED VALUES --- #
        html.Div([
            create_info_modal("protected_group", "Protected Values", "Please select the values of the protected feature which should be considered as belonging to observations from the protected group", ""),
            html.H3("Protected Values"),
            html.H5("Please select the Protected Values for the Protected Feature"),
            dcc.Dropdown(
                id='protected_value_dropdown_upload',
                options=[],
                placeholder='Select Values of the Protected Feature belonging to the Protected Group',
                multi=True,
                style={'width': '100%'}
            ),

        ], className="mb-4 mt-4 text-center"),
    
    # --- TARGET COLUMN --- #
    

        html.Div([
            create_info_modal("target_column_name", "Target Column", "The target column contains the values that you want to predict with your model.", ""),
            html.Div(id="target_column_alert"),
            html.H3(["6. Target Column", html.Sup("*")]),
            html.H5("Please select the Target Column"),
                dcc.Dropdown(
        id='target_column_dropdown',
        options=[],
        placeholder='Select Target Column'
    ),
        ], className="mb-4 text-center", id="conditional_output_target"),

        
    # --- FAVORABLE OUTCOME --- #
    html.Div([
        create_info_modal("favorable_outcomes", "Favorable Outcome", "The favorable outcomes are values of the target column that are considered to be positive predictions.", "It would be considered favorable for example to get the credit card successfully approved in a credit scoring scenario."),
        html.H3("Favorable Outcomes"),
        html.H5("Please select the Favorable Outcomes for the Target Column"),
        dcc.Dropdown(
            id='favorable_outcome_dropdown_upload',
            options=[],
            placeholder='Select Favorable Outcomes',
            multi=True,
            style={'width': '100%'}
        ),
    ], className="mb-4 mt-4 text-center", id="conditional_output_fav_out"),
       
    # --- FACTSHEET --- #
    
    html.Div([
        html.Div([
            create_info_modal("factsheet", "Factsheet", "The factsheet contains the most important information about the model.", ""),
            html.Div(id="factsheet_alert"),
            html.H3(["7. Factsheet", html.Sup("*")]),
            html.H5("Please upload the factsheet or create a new one using the button below", id="factsheet_info"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Upload(
                            id='factsheet_upload',
                            children=html.Div(['Drag and Drop or Select File']),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px',
                                'backgroundColor': '#FFFFFF'
                            }
                        )
                    ),
                    dbc.Col(
                        html.Div(dbc.Button("Create Factsheet",  id='create_factsheet', color="primary", className="mt-3 mb-3"))
                    )
                ]),
        ], className="text-center"),
        factsheet.layout,
    ],
            className="mb-4"),

    

    html.Div(id='factsheet_summary'),
    
    # --- MODEL --- #
    
    html.Div([
        
        html.Div([
            create_info_modal("model", "Model", "Please upload the model to be analyzed.", ""),
            html.Div(id="model_alert"),
            html.H3(["8. Model", html.Sup("*")]),
            html.H5("Please upload the model")
        ], className="text-center")
        # 5. Please upload the model as a .sav file.
    ], className="mb-4"),
    dcc.Upload(
        id='model_upload',
        children=html.Div([
            'Drag and Drop or Select File'
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'backgroundColor': '#FFFFFF'
        }
    ),
    html.Div(id='model_summary'),
    html.Div(html.Span(id="upload_alert")),
    html.Div(dbc.Button("Analyze",  id='upload_button', color="primary", className="mt-3 mb-3"), className="text-center"),
            ], style={
    "border": "1px solid #d8d8d8",
    "borderRadius": "6px",
    "backgroundColor": SECONDARY_COLOR
}),
],
fluid=False
)



if __name__ == '__main__':
    app.run_server(debug=True)
