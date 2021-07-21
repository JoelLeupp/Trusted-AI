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
exp_panel_comp = [html.H3("Explainability Panel",style={'text-align':'center'}),html.Br(),]
comp_weight = [html.H4("Weights",style={'text-align':'center'})]
for key, val in config_explainability["weights"].items():
    comp_weight.append(html.Label(key.replace("_",' '))) 
    comp_weight.append(html.Br())
    comp_weight.append(dcc.Input(id="w_"+key,value=val, type='text'))
    comp_weight.append(html.Br())

exp_panel_comp.append(html.Div(comp_weight))
exp_panel_comp.append(html.Br())
exp_panel_comp.append(html.H4("Parameters",style={'text-align':'center'}))
for key, val in config_explainability["parameters"].items():
    param_comp = [html.H4(key.replace("_",' '))]
    for param, info in val.items():
         param_comp.append(html.Label(html.Strong(key.replace("_",' '))))
         param_comp.append(html.Br())
         param_comp.append(dcc.Input(id="p_"+param,value=str(info["value"]), type='text'))
         param_comp.append(html.P(info["description"])) 
    exp_panel_comp.append(html.Div(param_comp))

exp_panel = html.Div(exp_panel_comp, style={'width': '22%', 'display': 'inline-block','height': '1500px',"vertical-align": "top",'margin-left': 10})

fairness_panel = html.Div([html.H3("Fairness Panel",style={'text-align':'center'}),html.Br(),html.H4("Weights"),html.Br(),html.H4("Parameters")], 
                          style={'width': '22%', 'display': 'inline-block','height': '1500px', "vertical-align": "top",'margin-left': 10})

robustness_panel = html.Div([html.H3("Robustness Panel",style={'text-align':'center'}),html.Br(),html.H4("Weights"),html.Br(),html.H4("Parameters")], 
                          style={'width': '22%', 'display': 'inline-block','height': '1500px',"vertical-align": "top",'margin-left': 10})

methodology_panel = html.Div([html.H3("Methodology Panel",style={'text-align':'center'}),html.Br(),html.H4("Weights"),html.Br(),html.H4("Parameters")], 
                          style={'width': '22%', 'display': 'inline-block','height': '1500px', "vertical-align": "top",'margin-left': 10})


children.append(html.Div([html.H3("Configuration",style={'text-align':'center'}),exp_panel,fairness_panel,robustness_panel,methodology_panel,]))


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

layout = html.Div(children=children, style={"margin-left": "100px","margin-right": "50px"})


    
if __name__ == '__main__':
    app.run_server(debug=True)
