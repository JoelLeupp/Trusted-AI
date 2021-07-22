# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 09:51:20 2021

@author: Besitzer
"""
import dash_core_components as dcc
import dash_html_components as html
import json

with open("apps/algorithm/config_methodology.json") as file:
           config_explainability = json.load(file)

# create panel
methodology_panel = [html.H4("Weights"),html.Br(),html.H4("Parameters")]