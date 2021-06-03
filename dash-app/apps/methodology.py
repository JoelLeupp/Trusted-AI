import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import numpy as np




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

layout = html.Div([
    dbc.Col(html.H5("Is the data normalized?", className="text-center")
                    , className="mb-2"),
    dbc.Col(dcc.RadioItems(options=[{'label': 'Yes', 'value': 1},{'label': 'No', 'value': 0},],value=1,
                           labelStyle={'display': 'inline-block'},className="text-center", inputStyle={"margin-right": "10px", "margin-left": "30px"})
                    , className="mb-2"),
    dbc.Col(html.H5("Is regularization used during the training?", className="text-center")
                    , className="mb-2"),
    dbc.Col(dcc.RadioItems(options=[{'label': 'Yes', 'value': 1},{'label': 'No', 'value': 0},],value=1,
                           labelStyle={'display': 'inline-block'},className="text-center", inputStyle={"margin-right": "10px", "margin-left": "30px"})
                    , className="mb-2"),
    dbc.Col(html.H5("What is the train-test split?", className="text-center")
                    , className="mb-2"),
    html.Div([html.P("Train Data", style={'float': 'left'}), html.P("Test Data", style={'float': 'right'})]),
    html.Br(),
    dbc.Col(dcc.Slider(min=0, max=100, marks={10:'10%', 20:'20%', 30:'30%', 40:'40%', 50:'50%', 60:'60%', 70:'70%', 80:'80%', 90:'90%', 100:'100%' }, value=50,
)  
                    , className="mb-2"),
    
   dbc.Col(dbc.Button("Calculate Trust Score",  href="/visualisation", color="primary", className="mt-3"),className="text-center")
])








if __name__ == '__main__':
    app.run_server(debug=True)
