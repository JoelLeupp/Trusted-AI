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
import dash_daq as daq

children=[]

# load case inputs
# model, train_data, test_data = get_case_inputs(case)

config_fairness, config_explainability, config_robustness, config_methodology = 0, 0, 0 ,0
for config in ["config_fairness", "config_explainability", "config_robustness", "config_methodology"]:
    with open("apps/algorithm/"+config+".json") as file:
            exec("%s = json.load(file)" % config)

#panels
exp_panel_comp = [html.H3("Explainability Panel TEST", style={'text-align':'center'}),html.Br(),] + explainability_panel
exp_panel = html.Div(exp_panel_comp, style={'width': '22%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10})

fair_panel_comp = [html.H3("Fairness Panel", style={'text-align':'center'}),html.Br(),] + fairness_panel
fair_panel = html.Div(fair_panel_comp, style={'width': '22%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10})

rob_panel_comp = [html.H3("Robustness Panel", style={'text-align':'center'}),html.Br(),] + robustness_panel
rob_panel = html.Div(rob_panel_comp, style={'width': '22%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10})

meth_panel_comp = [html.H3("Methodology Panel", style={'text-align':'center'}),html.Br(),] + methodology_panel
meth_panel = html.Div(meth_panel_comp, style={'width': '22%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10})


children.append(html.Div([html.H3("Configuration",style={'text-align':'center'}),exp_panel,fair_panel,rob_panel,meth_panel,]))

panel_div = html.Div(children, id= "panel", style= {'display': 'block'})

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#layout = html.Div(children=children, style={"margin-left": "100px","margin-right": "50px"})

layout = html.Div([
    panel_div,
    daq.BooleanSwitch(id='toggle-hide',
                      on=False,
                      label="show configuration",
                      labelPosition="top",
                      color = "green"
                     
                    ),  
        ], style={"margin-left": "100px","margin-right": "50px"})

    
if __name__ == '__main__':
    app.run_server(debug=True)
