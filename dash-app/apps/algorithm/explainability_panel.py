# -*- coding: utf-8 -*-
import dash_core_components as dcc
import dash_html_components as html
import json

with open("apps/algorithm/config_explainability.json") as file:
           config_explainability = json.load(file)

# create panel
exp_panel_comp = []

#weight panel
comp_weight = [html.H4("Weights",style={'text-align':'center'})]
for key, val in config_explainability["weights"].items():
    comp_weight.append(html.Label(key.replace("_",' '))) 
    comp_weight.append(html.Br())
    comp_weight.append(dcc.Input(id="w_"+key,value=val, type='text'))
    comp_weight.append(html.Br())

# parameter panel
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
    
explainability_panel = exp_panel_comp