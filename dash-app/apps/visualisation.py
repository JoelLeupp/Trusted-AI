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
from apps.algorithm.trusting_AI_algo import get_final_score

children=[
    html.H3(children='Spider Plots'),
    ]

# visualize final score
final_score, results = get_final_score()
pillars = list(final_score.keys())
values = list(final_score.values())
spider_plt = px.line_polar(r=values, theta=pillars, line_close=True, title='Trusting AI Final Score')
spider_plt.update_layout(title_x=0.5)
children.append(dcc.Graph(id='trusting_ai_score',figure=spider_plt))

my_palette = ['yellow','cornflowerblue','lightgrey','lightseagreen']
spider_plt_pillars=[]
for n, (pillar , sub_scores) in enumerate(results.items()):
    title = pillar
    categories = list(sub_scores.keys())
    val = list(sub_scores.values())
    spider_plt_pillar = px.line_polar(r=val, theta=categories, line_close=True, title=title)
    spider_plt_pillar.update_traces(fill='toself', fillcolor=my_palette[n], marker_color='rgb(250,00,00)',marker_line_width=1.5, opacity=0.6)
    spider_plt_pillar.update_layout(title_x=0.5)

    spider_plt_pillars.append(dcc.Graph(id=pillar, figure=spider_plt_pillar, style={'display': 'inline-block','width': '50%'}))

children.append(html.Div(children=spider_plt_pillars))

children.append(html.H3(children='Bar Charts'))
bar_chart = go.Figure(data=[go.Bar(
    x=pillars,
    y=values,
    marker_color=my_palette
)])
bar_chart.update_layout(title_text="Trusting AI Final Score", title_x=0.5)
children.append(dcc.Graph(id='trusting_ai_score_bar',figure=bar_chart))


bar_chart_pillars=[]
for n, (pillar , sub_scores) in enumerate(results.items()):
    title = pillar
    categories = list(sub_scores.keys())
    val = list(sub_scores.values())
    bar_chart_pillar = go.Figure(data=[go.Bar(x=pillars, y=values, marker_color=my_palette[n])])
    bar_chart_pillar.update_layout(title_text=title, title_x=0.5)
    bar_chart_pillars.append(dcc.Graph(id=str(pillar+"bar"), figure=bar_chart_pillar, style={'display': 'inline-block','width': '50%'}))
children.append(html.Div(children=bar_chart_pillars))  


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

#fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

layout = html.Div(children=children)


    
if __name__ == '__main__':
    app.run_server(debug=True)
