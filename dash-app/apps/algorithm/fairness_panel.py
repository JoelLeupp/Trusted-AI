# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 09:44:26 2021

@author: Besitzer
"""
import dash_core_components as dcc
import dash_html_components as html
import json

with open("apps/algorithm/config_fairness.json") as file:
           config_explainability = json.load(file)

# create panel
fairness_panel = [html.H4("Weights"),html.Br(),html.H4("Parameters")]