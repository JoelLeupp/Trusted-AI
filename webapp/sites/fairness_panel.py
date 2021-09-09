# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 09:44:26 2021

@author: Besitzer
"""
import os
import dash_core_components as dcc
import dash_html_components as html
import json
from config import METRICS_CONFIG_PATH


config_fairness, config_pillars = 0, 0
for config in ["config_pillars", "config_fairness"]:
    with open(os.path.join(METRICS_CONFIG_PATH, config + ".json")) as file:
            exec("%s = json.load(file)" % config)

# create panel
fairness_panel = [html.H4("Weights"),html.Br(),html.H4("Parameters")]

exp_panel_comp = []
input_ids = ["w_fair_pillar"]

comp_weight = []
comp_weight.append(html.H5("Pillar Weight",style={'text-align':'center'}))

comp_weight.append(html.Div(dcc.Input(id="w_"+"fair_pillar",value=config_pillars["fairness"], type='text'), 
                            style=dict(display='flex', justifyContent='center')))
comp_weight.append(html.Br())


comp_weight.append(html.H5("Metrics Weights",style={'text-align':'center'}))
for key, val in config_fairness["weights"].items():
    # comp_weight.append(html.Label(key.replace("_",' '))) 
    # comp_weight.append(html.Br())
    # comp_weight.append(dcc.Input(id="w_"+key,value=val, type='text'))
    # comp_weight.append(html.Br())
    input_id = "w_"+key
    input_ids.append(input_id)
    
    comp_weight.append(html.Div([
        html.Div(html.Label(key.replace("_",' ')), style={'width': '40%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
        html.Div(dcc.Input(id="w_"+key,value=val, type='text'), style={'width': '40%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
        ]))
# parameter panel
exp_panel_comp.append(html.Div(comp_weight))
# exp_panel_comp.append(html.Br())
# exp_panel_comp.append(html.H4("Parameters",style={'text-align':'center'}))

fairness_panel = exp_panel_comp
fair_input_ids = input_ids