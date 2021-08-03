import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import os

existing_problem_sets = [{'label': f.name, 'value': f.path} for f in os.scandir('./problem_sets') if f.is_dir()]


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

layout = dbc.Container([
    dbc.Col([
        html.H3("1. Please select the correpsonding Problem Set.", className="text-center"),
        html.H5("Please place the label to the last column of the dataframe.", className="text-center"),
        dcc.Dropdown(
            id='demo-dropdown',
            options=existing_problem_sets,
            value='Select Problem Set A'
        )], 
        className="mb-4"
    ),
    
    dbc.Col([
        html.H3("2. Enter a name for your model.", className="text-center"),
        html.H5("Your model will be saved under this name.", className="text-center"),
        dcc.Input(id="input2", type="text", placeholder="", debounce=True, style={'width': '100%', 'textAlign': 'center'})
        ], 
        className="mb-4"
    ),
    
    dbc.Col([html.H3("3. Please upload the train data (csv and pickle files are accepted).", className="text-center"),
            html.H5("Please place the label to the last column of the dataframe.", className="text-center")],
            className="mb-4"),
    dcc.Upload(
        id='upload-train-data',
        children=html.Div([
            'Drag and Drop or Select Files'
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    html.Div(id='output-train-data-upload'),
    
    dbc.Col([html.H3("4. Please upload the test data (csv and pickle files are accepted).", className="text-center"),
            html.H5("Please place the label to the last column of the dataframe.", className="text-center")],
            className="mb-4"),
    
    dcc.Upload(
        id='upload-test-data',
        children=html.Div([
            'Drag and Drop or Select Files'
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    html.Div(id='output-test-data-upload'),
    dbc.Col(html.H3("5. Please upload the model as a .sav file.", className="text-center")
                    , className="mb-4"),
    dcc.Upload(
        id='upload-model',
        children=html.Div([
            'Drag and Drop or Select Files'
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    html.Div(id="model-uploaded-div", className="text-center"),
    # dbc.Col(html.H3("4. Is regularization used during the training?", className="text-center")
    #                 , className="mb-2"),
    # dbc.Col(dcc.RadioItems(id='regularization', options=[{'label': 'Yes', 'value': 1},{'label': 'No', 'value': 0},],value=1,
    #                        labelStyle={'display': 'inline-block', "marginRight": "20px", "marginLeft": "10px"}, className="text-center", inputStyle={'display': 'inline-block', "marginRight": "10px", })
    #                 , className="text-center"),
    html.Div(html.Span(id="hidden-div")),
    html.Div(dbc.Button("Calculate Trust Score",  id='trustscore-button', color="primary", className="mt-3"), className="text-center"),
    
],
fluid=False
)


#href="/visualisation",





if __name__ == '__main__':
    app.run_server(debug=True)
