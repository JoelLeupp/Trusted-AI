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
import dash_bootstrap_components as dbc
from apps import test
from apps.algorithm.helper_functions import get_performance_table, get_final_score, get_case_inputs, trusting_AI_scores, get_trust_score

children=[]
     

# visualize final score
### delete later
# define model inputs
# choose scenario case (case1,case1,..)
case = "case1"
np.random.seed(6)

# load case inputs
model, train_data, test_data = get_case_inputs(case)

config_fairness, config_explainability, config_robustness, config_methodology, config_pillars = 0, 0, 0 ,0,0
for config in ["config_pillars","config_fairness", "config_explainability", "config_robustness", "config_methodology"]:
    with open("apps/algorithm/"+config+".json") as file:
            exec("%s = json.load(file)" % config)


properties = trusting_AI_scores(model, train_data, test_data, config_fairness, config_explainability, config_robustness, config_methodology).properties
final_score, results = get_final_score(model, train_data, test_data, config_fairness, config_explainability, config_robustness, config_methodology)
trust_score = get_trust_score(final_score, config_pillars)
performance =  get_performance_table(model, test_data).transpose()
pillars = list(final_score.keys())
values = list(final_score.values())

performance_table = dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in performance.columns],
                        data=performance.to_dict('records'),
                         style_table={
                'width': '70%',
                'margin-left': 'auto', 
                'margin-right': 'auto'
            }
                    )
children.append(html.Br())
children.append(html.H5("Performance metrics", style={"width": "70%","text-align": "center", "margin-right": "auto", "margin-left": "auto" }))
children.append(performance_table)
children.append(html.Br())
children.append(html.Hr())
children.append(html.Br())

children.append(html.Br())
children.append(test.layout)
children.append(html.Br())

children.append(dbc.Col(
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
         className="text-center"))


spider_plt = px.line_polar(r=values, theta=pillars, line_close=True, title='AI Final Trust Score: {}'.format(trust_score))
spider_plt.update_layout(title_x=0.5)
children.append(dcc.Graph(id='spider',figure=spider_plt, style={'display': 'none'}))

my_palette = ['yellow','cornflowerblue','lightgrey','lightseagreen']
spider_plt_pillars=[]
for n, (pillar , sub_scores) in enumerate(results.items()):
    title = pillar
    categories = list(map(lambda x: x.replace("_",' '), sub_scores.keys())) 
    val = list(map(int, sub_scores.values()))
    spider_plt_pillar = px.line_polar(r=val, theta=categories, line_close=True, title=title)
    spider_plt_pillar.update_traces(fill='toself', fillcolor=my_palette[n], marker_color='rgb(250,00,00)',marker_line_width=1.5, opacity=0.6)
    spider_plt_pillar.update_layout(title_x=0.5)
    # spider_plt_pillar.update_yaxes(range=[0,5],autorange=False)

    spider_plt_pillars.append(dcc.Graph(id=pillar, figure=spider_plt_pillar, style={'display': 'inline-block','width': '50%'}))

children.append(html.Div(id="spider_pillars",children=spider_plt_pillars, style={'display': 'none'}))

bar_chart = go.Figure(data=[go.Bar(
    x=pillars,
    y=values,
    marker_color=my_palette
)])
bar_chart.update_layout(title_text="AI Final Trust Score: 3.1", title_x=0.5)
children.append(dcc.Graph(id='bar',figure=bar_chart, style={'display': 'block'}))


bar_chart_pillars=[]
for n, (pillar , sub_scores) in enumerate(results.items()):
    title = pillar
    categories = list(map(lambda x: x.replace("_",' '), sub_scores.keys())) 
    values = list(map(int, sub_scores.values()))
    bar_chart_pillar = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=my_palette[n])])
    bar_chart_pillar.update_layout(title_text=title, title_x=0.5)
    bar_chart_pillars.append(dcc.Graph(id=str(pillar+"bar"), figure=bar_chart_pillar, style={'display': 'inline-block','width': '50%'}))
children.append(html.Div(id="bar_pillars", children=bar_chart_pillars, style={'display': 'block'}))  


# properties explainability
prop_exp = properties["explainability"]

importance = prop_exp["Feature_Relevance"]['importance'].value
pct_dist = prop_exp["Feature_Relevance"]["pct_dist"].value 
pct_mark = importance["labels"][int(float(pct_dist[:-1])/100 * len(importance["value"]))-1]

fig = go.Figure(data=go.Scatter(x=importance["labels"], y=importance["value"],mode='lines+markers'))
fig.add_vline(x=pct_mark, line_width=3, line_dash="dash")
fig.add_trace(go.Scatter(
    x=[importance["labels"][int(float(pct_dist[:-1])/100 * len(importance["value"]))-3]], y=[max(importance["value"])],
    text="60% Threshold",
    mode="text",
))

def get_properties_exp(p):
    prop = p[1]
    if p[0]=="importance":
        return dcc.Graph(id='test',figure=fig, style={'display': 'block'})
    else:
       return html.H5(str(prop.description) + ": " + str(prop.value),style={'text-align':'center'})
        
children.append(html.Div([
    html.H2('Extracted properties from the Explainability pillar'),
    html.Br(),
    html.Div(
        list(map(lambda item: html.Div([
        html.H3(item[0].replace("_",' '),style={'text-align':'center'}),
        html.Div(list(map(get_properties_exp ,item[1].items()))),
        html.Br()
    ]), prop_exp.items()))
        )
] ,style={'text-align':'center'}))



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

layout = html.Div(children=children)


    
if __name__ == '__main__':
    app.run_server(debug=True)
