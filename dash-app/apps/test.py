import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import time
import random
import numpy as np
import pickle
import statistics
import seaborn as sn
import pandas as pd
import json
from math import pi
import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from apps.algorithm.helper_functions import get_performance_table, get_final_score, get_case_inputs, trusting_AI_scores
from apps.algorithm.explainability_panel import explainability_panel
from apps.algorithm.fairness_panel import fairness_panel
from apps.algorithm.robustness_panel import robustness_panel
from apps.algorithm.methodology_panel import methodology_panel

children=[
     dbc.Col(
         [html.H5("Please select the type of the visualisations.", style={'float': 'left', "width": "70%","margin-right": "-30%", "margin-left": "10%" }),
          html.Div(
          dcc.Dropdown(
                id='plot_type',
                options=[
                     {'label': 'Bar Charts', 'value': 'bar'},
                    {'label': 'Spider Plots', 'value': 'spider'}
                    ],
                value='bar',
                clearable=False),
          style={'display': 'inline-block', "width": "20%", "margin-top": "-10px" }
          )],
         className="text-center")]

# visualize final score
### delete later
# define model inputs
# choose scenario case (case1,case1,..)
case = "case1"
np.random.seed(6)

# load case inputs
model, train_data, test_data = get_case_inputs(case)

config_fairness, config_explainability, config_robustness, config_methodology = 0, 0, 0 ,0
for config in ["config_fairness", "config_explainability", "config_robustness", "config_methodology"]:
    with open("apps/algorithm/"+config+".json") as file:
            exec("%s = json.load(file)" % config)

# panels
exp_panel_comp = [html.H3("Explainability Panel", style={'text-align':'center'}),html.Br(),] + explainability_panel
exp_panel = html.Div(exp_panel_comp, style={'width': '22%', 'display': 'inline-block','height': '1500px',"vertical-align": "top",'margin-left': 10})

fair_panel_comp = [html.H3("Fairness Panel", style={'text-align':'center'}),html.Br(),] + fairness_panel
fair_panel = html.Div(exp_panel_comp, style={'width': '22%', 'display': 'inline-block','height': '1500px',"vertical-align": "top",'margin-left': 10})

rob_panel_comp = [html.H3("Robustness Panel", style={'text-align':'center'}),html.Br(),] + robustness_panel
rob_panel = html.Div(exp_panel_comp, style={'width': '22%', 'display': 'inline-block','height': '1500px',"vertical-align": "top",'margin-left': 10})

meth_panel_comp = [html.H3("Methodology Panel", style={'text-align':'center'}),html.Br(),] + methodology_panel
meth_panel = html.Div(exp_panel_comp, style={'width': '22%', 'display': 'inline-block','height': '1500px',"vertical-align": "top",'margin-left': 10})


children.append(html.Div([html.H3("Configuration",style={'text-align':'center'}),exp_panel,fairness_panel,robustness_panel,methodology_panel,]))


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

layout = html.Div(children=children, style={"margin-left": "100px","margin-right": "50px"})


    
if __name__ == '__main__':
    app.run_server(debug=True)
