# -*- coding: utf-8 -*-
import dash_core_components as dcc
import dash_html_components as html
import json

config_explainability,config_pillars = 0, 0
for config in ["config_pillars", "config_explainability"]:
    with open("apps/algorithm/"+config+".json") as file:
            exec("%s = json.load(file)" % config)

# create panel
exp_panel_comp = []

#weight panel
comp_weight = [html.H4("Weights",style={'text-align':'center'})]
comp_weight.append(html.H5("Pillar Weight",style={'text-align':'center'}))

comp_weight.append(html.Div(dcc.Input(id="w_"+"exp_pillar",value=config_pillars["explainability"], type='text'), 
                            style=dict(display='flex', justifyContent='center')))
comp_weight.append(html.Br())


comp_weight.append(html.H5("Metrics Weights",style={'text-align':'center'}))
for key, val in config_explainability["weights"].items():
    # comp_weight.append(html.Label(key.replace("_",' '))) 
    # comp_weight.append(html.Br())
    # comp_weight.append(dcc.Input(id="w_"+key,value=val, type='text'))
    # comp_weight.append(html.Br())
    comp_weight.append(html.Div([
        html.Div(html.Label(key.replace("_",' ')), style={'width': '40%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
        html.Div(dcc.Input(id="w_"+key,value=val, type='text'), style={'width': '40%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
        ]))
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