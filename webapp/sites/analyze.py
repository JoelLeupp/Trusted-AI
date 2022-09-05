# === IMPORTS ===
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.express as px
from app import app
import pandas as pd
import os
import json
import glob
import shutil
from helpers import *
from config import *
from algorithms.trustworthiness import trusting_AI_scores, get_trust_score, get_final_score, get_final_score_unsupervised
import dash_table
import numpy as np
from sites import config_panel
import plotly.express as px
import plotly.graph_objects as go
import dash
import warnings 
import plotly
from dash_extensions.snippets import send_file

warnings.filterwarnings('ignore')

# === CONFIG SUPERVISED ===
config_fairness, config_explainability, config_robustness, config_methodology, config_pillars = 0, 0, 0 ,0,0

for config in ["config_pillars","config_fairness", "config_explainability", "config_robustness", "config_methodology"]:
    with open(os.path.join(METRICS_CONFIG_PATH, config + ".json")) as file:
        exec("%s = json.load(file)" % config)

pillars = ['fairness', 'explainability', 'robustness', 'methodology']
weight_config = dict(fairness=config_fairness["weights"], explainability=config_explainability["weights"],
                   robustness=config_robustness["weights"], methodology=config_methodology["weights"], pillars=config_pillars)

mappings_config = dict(fairness=config_fairness["parameters"], explainability=config_explainability["parameters"],
                   robustness=config_robustness["parameters"], methodology=config_methodology["parameters"])

metric_description = {**config_fairness["metrics"], **config_explainability["metrics"], **config_robustness["metrics"], **config_methodology["metrics"]}

with open('configs/supervised/weights/default.json', 'w') as outfile:
                json.dump(weight_config, outfile, indent=4)

with open('configs/supervised/mappings/default.json', 'w') as outfile:
                json.dump(mappings_config, outfile, indent=4)

for s in SECTIONS[1:]:
    with open('configs/supervised/mappings/{}/default.json'.format(s), 'w') as outfile:
                json.dump(mappings_config[s], outfile, indent=4)
charts = []

# === METRICS SUPERVISED===
fairness_metrics = FAIRNESS_METRICS
explainability_metrics = EXPLAINABILITY_METRICS
robustness_metrics = ROBUSTNESS_METRICS
methodology_metrics = METHODOLOGY_METRICS

# === CONFIG UNSUPERVISED =========================================================================
config_fairness, config_explainability, config_robustness, config_methodology, config_pillars = 0, 0, 0 ,0,0

for config in ["config_pillars","config_fairness", "config_explainability", "config_robustness", "config_methodology"]:
    with open(os.path.join(METRICS_CONFIG_PATH_UNSUPERVISED, config + ".json")) as file:
        exec("%s = json.load(file)" % config)

pillars = ['fairness', 'explainability', 'robustness', 'methodology']
weight_config_unsupervised = dict(fairness=config_fairness["weights"], explainability=config_explainability["weights"],
                   robustness=config_robustness["weights"], methodology=config_methodology["weights"], pillars=config_pillars)

mappings_config_unsupervised = dict(fairness=config_fairness["parameters"], explainability=config_explainability["parameters"],
                   robustness=config_robustness["parameters"], methodology=config_methodology["parameters"])

metric_description_unsupervised = {**config_fairness["metrics"], **config_explainability["metrics"], **config_robustness["metrics"], **config_methodology["metrics"]}

with open('configs/unsupervised/weights/default.json', 'w') as outfile:
                json.dump(weight_config_unsupervised, outfile, indent=4)

with open('configs/unsupervised/mappings/default.json', 'w') as outfile:
                json.dump(mappings_config_unsupervised, outfile, indent=4)

for s in SECTIONS[1:]:
    with open('configs/unsupervised/mappings/{}/default.json'.format(s), 'w') as outfile:
                json.dump(mappings_config_unsupervised[s], outfile, indent=4)
charts = []

# === METRICS UNSUPERVISED ===
fairness_metrics_unsupervised = FAIRNESS_METRICS_UNSUPERVISED
explainability_metrics_unsupervised = EXPLAINABILITY_METRICS_UNSUPERVISED
robustness_metrics_unsupervised = ROBUSTNESS_METRICS_UNSUPERVISED
methodology_metrics_unsupervised = METHODOLOGY_METRICS_UNSUPERVISED

# === SECTIONS ===
def general_section():
    return html.Div([
        dbc.Button(
            html.I(className="fas fa-backspace"),
            id="delete_solution_button", 
            n_clicks=0,
            style={"float": "right", "backgroundColor": SECONDARY_COLOR}
        ),
daq.BooleanSwitch(id='toggle_charts',
                on=False,
                label='Alternative Style',
                labelPosition="top",
                color = TRUST_COLOR,   
                style={"float": "right"}
            ),
                html.Div([html.H2("• General Information")]),
                html.Div([], id="general_description"),
                dbc.Row([
                    dbc.Col(html.Div(["Performance Metrics Section"], id="performance_metrics_section")),
                    dbc.Col(html.Div(["Properties Section"], id="properties_section"))
                ]),
                dcc.Store(id='input-mappings'),
                ])

def trust_section():
    return html.Div([ 
        html.Div([
            html.Div([daq.BooleanSwitch(id='show_weighting',
                      on=False,
                      label='Show Weights',
                      labelPosition="top",
                      color = TRUST_COLOR,
                      style={"float": "right"}
                     
                    )]),
            html.H2("• Trustworthiness"),
        ], id="trust_section_heading", className="mt-2 mb-5"),

        html.Div([], id="trust_overview"),
        html.H3("Overall Score", className="text-center mt-2"),
        html.Div([], id="trust_star_rating", className="star_rating, text-center"),
        html.B(["X/5"], id="trust_score", className="text-center", style={"display": "block","font-size":"32px"}),
        dcc.Graph(id='spider', style={'display': 'none'}),
        dcc.Graph(id='bar', style={'display': 'block'}),
        html.Div([], id="trust_details"),
        html.Hr()
    ], id="trust_section", style={"display": "None"})

def alert_section(name):
    return html.Div([], id="{}_alert_section".format(name), className="text-center", style={"color":"Red"})

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


for pillar in SECTIONS[1:]:
    @app.callback(
        Output("modal-{}-mapping".format(pillar), "is_open"),
        [Input('save-mapping-{}'.format(pillar), "n_clicks"),Input("save-{}-mapping".format(pillar), "n_clicks")],
        State("modal-{}-mapping".format(pillar), "is_open"))
    def update_output(n, n2, is_open):
        if n or n2:
            return not is_open
        else:
            return is_open
    
    
    @app.callback(
        Output("{}_available_metrics".format(pillar), "children"),
        Input("result", "data"), State("{}_available_metrics".format(pillar), "className"))
    def update_metric_availability(data, pillar):
        if not data:
            return []
        else:
            result = json.loads(data)
            trust_score = result["results"][pillar]
            all_metrics = list(trust_score.keys())
            calculated_metrics = list(filter(lambda k: not np.isnan(trust_score[k]), trust_score))
            # return html.Div([html.Br(),
            #             dcc.Markdown("Total available metrics: *{}*".format(", ".join(list(map(lambda x: x.replace("_", " ").title() ,all_metrics))))),
            #                  dcc.Markdown("Metrics available for the given solution ({}/{}): *{}*".format(len(calculated_metrics),len(all_metrics),", ".join(list(map(lambda x: x.replace("_", " ").title() ,calculated_metrics)))))])
            return html.Div([html.Br(),dcc.Markdown("Metrics Computed ({}/{})".format(len(calculated_metrics),len(all_metrics)))],style={"text-align":"center"})
        
    @app.callback(
            Output("mapping-dropdown-{}".format(pillar), "options"),
            Input("modal-saved-{}".format(pillar), "is_open"),
            State("mapping-dropdown-{}".format(pillar), "className"))
    def update_options(n, pillar):
        options = list(map(lambda name:{'label': name[:-5], 'value': "configs/supervised/mappings/{}/{}".format(pillar,name)} ,listdir_nohidden("configs/supervised/mappings/{}".format(pillar))))
        return options
    
    @app.callback(
                 list(map(lambda i: Output(i, "value"),mapping_panel(pillar)[1])), 
                Input("mapping-dropdown-{}".format(pillar), 'value'),
                State("mapping-dropdown-{}".format(pillar), "className"))
    def update_mapping_input(conf_name,pillar):
        
        with open(conf_name ,'r') as f:
                config = json.loads(f.read())
                
        output = []
        ids = mapping_panel(pillar)[1]
        for i in ids:
            metric, param = i.split("-")
            output.append(str(config[metric][param]["value"]))
            
        return output
    
    @app.callback(
        [Output("modal-saved-{}".format(pillar), "is_open"),Output("mapping-dropdown-{}".format(pillar), "value")],
        [Input("save-{}-mapping".format(pillar), "n_clicks")],
        [State("modal-saved-{}".format(pillar), "is_open"),State("mapping-name-{}".format(pillar), "value"), State(mapping_panel(pillar)[1][0],"name")]+ 
        list(map(lambda i: State(i, "value"),mapping_panel(pillar)[1])))
    def save_mapping(n, is_open, conf_name, pillar, *args):
        
        if conf_name:  
            inputs= dict()
        
            for name, val in zip(mapping_panel(pillar)[1] , args):
                try:
                    res = eval(val)
                except (SyntaxError, NameError, TypeError, ZeroDivisionError):
                    res = val
               
                inputs[name] = res
            
            with open('configs/supervised/mappings/{}/default.json'.format(pillar),'r') as f:
                config_file = json.loads(f.read())
                
            for i in mapping_panel(pillar)[1]:
                 metric, param = i.split("-")
                 config_file[metric][param]["value"] = inputs[i]
            with open('configs/supervised/mappings/{}/{}.json'.format(pillar,conf_name), 'w') as outfile:
                json.dump(config_file, outfile, indent=4)
                    
            return not is_open, 'configs/supervised/mappings/{}/{}.json'.format(pillar,conf_name)
        else:
            return is_open, 'configs/supervised/mappings/{}/default.json'.format(pillar)

for s in SECTIONS:
    @app.callback(
        [Output("{0}_mapping".format(s), "is_open"),
        Output("{0}_mapping".format(s), "style")],
        [Input("toggle_{}_mapping".format(s), "on")],
        [State("{0}_mapping".format(s), "is_open")],
        prevent_initial_call=True
    )
    def toggle_mapping_section(n, is_open):
        #app.logger.info("toggle {0} detail section".format(s))
        if is_open:
            return (not is_open, {'display': 'None'})
        else:
            return (not is_open,  {"background-color": "rgba(255,228,181,0.5)",'padding-bottom': 20, 'display': 'Block'})
        
for s in SECTIONS[1:]:
    @app.callback(
        [Output("{0}_details".format(s), "is_open"),
         Output("{0}_configuration".format(s), "is_open")],
        [Input("toggle_{0}_details".format(s), "on")])
    def toggle_detail_section(is_open):
        return is_open, is_open

        
@app.callback(
    Output(component_id="bar", component_property='style'),
    Output(component_id="fairness_bar", component_property='style'),
    Output(component_id="explainability_bar", component_property='style'),
    Output(component_id="robustness_bar", component_property='style'),
    Output(component_id="methodology_bar", component_property='style'),
    Output(component_id="spider", component_property='style'),
    Output(component_id="fairness_spider", component_property='style'),
    Output(component_id="explainability_spider", component_property='style'),
    Output(component_id="robustness_spider", component_property='style'),
    Output(component_id="methodology_spider", component_property='style'),
    Input('toggle_charts', 'on')
)
def toggle_charts(visibility_state):
    if visibility_state == True:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    if visibility_state == False:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

    
@app.callback(
    list(map(lambda x: Output("{}_section".format(x), "hidden"),  SECTIONS[1:])),
    list(map(lambda x: Input("{}_s".format(x), "n_clicks"),  SECTIONS[1:])) + [Input("all_s", "n_clicks")],
    list(map(lambda x: State("{}_section".format(x), "hidden"),  SECTIONS[1:])),
    prevent_initial_call=False
)
def toggle_hide_pillar_section(fn,en,rn,mn,alln, fis_open, eis_open, ris_open, mis_open):
    if fn or en or rn or mn or alln:
        pillars = np.array(['fairness', 'explainability', 'robustness', 'methodology'])
        out= np.array( [True,True,True,True])
        ctx = dash.callback_context
        pillar = ctx.triggered[0]['prop_id'][:-11]
        if pillar=="all":
            if fis_open or eis_open or ris_open or mis_open:
                return [False,False,False,False]
            else:
                return [True,True,True,True]
        else:
            is_open = eval(pillar[0]+"is_open")
            out[pillars==pillar]= not is_open 
            return list(out)
    else:
        return [True,True,True,True]

@app.callback(Output('input-mappings', 'data'), 
        list(map(lambda i: Input('apply-mapping-{}'.format(i), "n_clicks"),SECTIONS[1:])),
        list(map(lambda i: State(i, "value"), sum(list(map(lambda s: mapping_panel(s)[1],SECTIONS[1:])),[]) )))
       
def store_mappings_config(n1, n2, n3, n4, *args):
    
    inputs= dict()
    
    for name, val in zip(sum(list(map(lambda s: mapping_panel(s)[1],SECTIONS[1:])),[]) , args):
        try:
               res = eval(val)
        except (SyntaxError, NameError, TypeError, ZeroDivisionError):
           res = val
           
        inputs[name] = res
        
    with open('configs/supervised/mappings/default.json','r') as f:
            config_file = json.loads(f.read())
    
    
    pillars = SECTIONS[1:]
    ids = list(map(lambda s: mapping_panel(s)[1],SECTIONS[1:]))
    for pillar, map_ids in zip(pillars, ids):
        for i in map_ids:
            metric, param = i.split("-")
            config_file[pillar][metric][param]["value"] = inputs[i]
    
    return json.dumps(config_file)
    
@app.callback(Output('general_description', 'children'),
              [Input('scenario_dropdown', 'value'),
              Input('solution_set_dropdown', 'value')], prevent_initial_call=True)
def show_general_description(scenario_id, solution_set_path):
    description = []
    if scenario_id and solution_set_path:
        scenario_factsheet = read_scenario_factsheet(scenario_id)
        scenario_description = get_scenario_description(scenario_id)
        scenario_description_header = html.H5("Scenario Description")
        scenario_description_table = dash_table.DataTable(
            id='scenario_description_table',
            columns=[{"name": i, "id": i} for i in scenario_description.columns],
            data=scenario_description.to_dict('records'),
            style_table={
                "width": "100%",
                'overflowX': 'hidden',
                'textAlign': 'left'
            },
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': SECONDARY_COLOR,
            },
            style_cell={
                'textAlign': 'left',
                'backgroundColor': SECONDARY_COLOR,
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'key'},
                    'fontWeight': 'bold',
                    'width': '30%'
                } 
            ],
            style_as_list_view=True,
            css=[
              {
                 'selector': 'tr:first-child',
                 'rule': 'display: none',
              },
            ],
        )
        description.append(html.Div([scenario_description_header, scenario_description_table], className="mt-4 mb-4"))
        

        factsheet = read_factsheet(solution_set_path)

        solution_description_header = html.H5("Model Information")
        solution_description= get_solution_description(factsheet)
        solution_description_table = dash_table.DataTable(
            id='solution_description_table',
            columns=[{"name": i, "id": i} for i in solution_description.columns],
            data=solution_description.to_dict('records'),
            style_table={
                                    "width": "100%",
                                    'overflowX': 'hidden',
                                    'textAlign': 'left'
                                },
                                style_data={
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                },
                                style_header={
                                    'backgroundColor': SECONDARY_COLOR,
                                },
                                style_cell={
                                    'textAlign': 'left',
                                    'backgroundColor': SECONDARY_COLOR,
                                },
                                style_cell_conditional=[
                                    {
                                        'if': {'column_id': 'key'},
                                        'fontWeight': 'bold',
                                        'width': '30%'
                                    } 
                                ],
                                style_as_list_view=True,
                                css=[
                                  {
                                     'selector': 'tr:first-child',
                                     'rule': 'display: none',
                                  },
                                ],
        )
        description.append(html.Div([solution_description_header, solution_description_table], className="mt-4 mb-4"))
        return description
    else:
        return ""
    
# === FAIRNESS ===
@app.callback(
    Output("fairness_configuration", 'children'),
    [Input('solution_set_dropdown', 'value')], prevent_initial_call=True)
def fairness_configuration(solution_path):
    if solution_path is not None:
        data =  read_train(solution_path)
        features = list(data.columns)
        
        factsheet = read_factsheet(solution_path)
        try:
            protected_feature, protected_values, target_column, favorable_outcomes = load_fairness_config(factsheet)
        except Exception as e:
            protected_feature = ""
            protected_values = [] 
            target_column = ""
            favorable_outcomes = []
            

        
        title_section = html.Div([html.H3("▶ Fairness Configuration")])
        explanation_section = html.Div([html.Img(src=app.get_asset_url('fairness.png'), height="300px"), html.Div("For computing the fairness metrics, the dataset is divided into a protected and an unprotected group based on the values of the protected feature (e.g Gender). The fairness metrics always compare different values (e.g TPR) across these two groups.")], style={'textAlign': 'center'})
        alert_section = html.Div([], id="fairness_configuration_alert")
        
        try:
            protected_feature_dropdown_options = list(map(lambda x: {"label": x, "value": x}, features))
        except Exception as e:
            protected_feature_dropdown_options = []
        protected_feature_dropdown = html.Div([
            "Select Protected Feature", 
            dcc.Dropdown(
                id="protected_feature_dropdown",
                options=protected_feature_dropdown_options,
                value=protected_feature
            ),
        ])
        
        try:
            protected_value_dropdown_options = list(map(lambda x: {"label": "{0}=={1}".format(protected_feature, x), "value": x}, np.unique(data[protected_feature])))
        except Exception as e:
            protected_value_dropdown_options = []
    
        protected_value_dropdown = html.Div([
            "Select Protected Values",
            dcc.Dropdown(
                id='protected_value_dropdown',
                options=protected_value_dropdown_options,
                value=protected_values,
                multi=True,
                style={'width': '100%'}
            )
        ])
        
        try:
            target_column_dropdown_options = list(map(lambda x: {"label": x, "value": x}, features))
        except Exception as e:
            target_column_dropdown_options = []
            
        target_column_dropdown = html.Div([
            "Select Target Column", 
            dcc.Dropdown(
                id="target_column_dropdown",
                options=target_column_dropdown_options,
                value=target_column
            )
        ])
        
        try:
            favorable_outcome_dropdown_options = list(map(lambda x: {"label": "{0}=={1}".format(target_column, x), "value": x}, np.unique(data[target_column])))
        except Exception as e:
            favorable_outcome_dropdown_options = []
            
        favorable_outcome_dropdown = html.Div([
            "Select Favorable Outcomes",
            dcc.Dropdown(
                id='favorable_outcome_dropdown',
                options=favorable_outcome_dropdown_options,
                value=favorable_outcomes,
                multi=True,
                style={'width': '100%'}
            )
        ])
        
        sections = [
                    html.Hr(),
                    title_section,
                    explanation_section,
                    alert_section,
                    protected_feature_dropdown,
                    protected_value_dropdown,
                    target_column_dropdown,
                    favorable_outcome_dropdown,
                    html.Hr()
                   ]
        return sections
    else:
        return []

    '''
The following function updates
'''
@app.callback(
    Output('protected_value_dropdown', 'options'),
     Output('protected_value_dropdown', 'value'),
    [Input('protected_feature_dropdown', 'value'),
     State('solution_set_dropdown', 'value')], prevent_initial_call=True)
def update_protected_value_dropdown_options(protected_feature, solution_path):
    if solution_path is not None:
        data =  read_train(solution_path)
        
        try:
            protected_value_dropdown_options = list(map(lambda x: {"label": "{0}=={1}".format(protected_feature, x), "value": x}, np.unique(data[protected_feature])))
        except Exception as e:
            protected_value_dropdown_options = []
    
        new_factsheet = {"fairness": {}}
        new_factsheet["fairness"]["protected_values"] = ""
        factsheet_path = os.path.join(solution_path, FACTSHEET_NAME)
        update_factsheet(factsheet_path, new_factsheet)
        
        return protected_value_dropdown_options, ''
 
    '''
The following function updates
'''
@app.callback(
    Output('favorable_outcome_dropdown', 'options'),
    Output('favorable_outcome_dropdown', 'value'),
    [Input('target_column_dropdown', 'value'),
     State('solution_set_dropdown', 'value')], prevent_initial_call=True)
def update_facorable_outcome_dropdown_options(target_column, solution_path):
    if solution_path is not None:
        data =  read_train(solution_path)
        
        try:
            favorable_outcome_dropdown_options = list(map(lambda x: {"label": "{0}=={1}".format(target_column, x), "value": x}, np.unique(data[target_column])))
            
        except Exception as e:
            target_column_dropdown_options = []
    
        new_factsheet = {"fairness": {}}
        new_factsheet["fairness"]["favorable_outcomes"] = ""
        factsheet_path = os.path.join(solution_path, FACTSHEET_NAME)
        update_factsheet(factsheet_path, new_factsheet)
        
        return favorable_outcome_dropdown_options, ''


'''
The following function updates
'''
@app.callback(
    Output("fairness_configuration_alert", 'children'),
    [Input('protected_feature_dropdown', 'value'),
     Input('protected_value_dropdown', 'value'),
     Input('target_column_dropdown', 'value'),
     Input('favorable_outcome_dropdown', 'value'),
     State('scenario_dropdown', 'value'),
     State('solution_set_dropdown', 'value')
    ], prevent_initial_call=True)
def update_fairness_configuration(protected_feature, protected_values, target_column, favorable_outcomes, scenario_id, solution_id):
    if scenario_id and solution_id:
        new_factsheet = {"general": {}, "fairness": {}}
        new_factsheet["fairness"]["protected_feature"] = protected_feature
        new_factsheet["fairness"]["protected_values"] = protected_values
        new_factsheet["general"]["target_column"] = target_column
        new_factsheet["fairness"]["favorable_outcomes"] = favorable_outcomes

        factsheet_path = os.path.join(solution_id, FACTSHEET_NAME)
        
        update_factsheet(factsheet_path, new_factsheet)
    return html.Div("Updated factsheet")
    
# === EXPLAINABILITY ===
@app.callback(
    Output("explainability_details", 'children'),
    Input('result', 'data'), prevent_initial_call=True)
def explainability_details(data):
    if not data:
        return []
    result = json.loads(data)
    properties = result["properties"]
    metrics = list(properties["explainability"].keys())
    
    sections = [html.H3("▶ Explainability Metrics")]
    comp_list = []
    non_comp_list = []
    for i in range(len(metrics)):
            metric_id = metrics[i]
            score = result["results"]["explainability"][metric_id]
            if np.isnan(score):
                metric_name = metric_id.replace("_", " ")
                non_comp_list.append(create_metric_details_section(metric_id, i, 2, True,score)
                   # html.H4("{2}.{0} {1}".format(i+1, metric_name, 2))
                    )
            else:
                comp_list.append(create_metric_details_section(metric_id, i, 2, True,score))
            
    sections.append(html.Div([
        html.Div(comp_list),
        html.H5("Non-Computable Metrics") if not []==non_comp_list else [],
        html.Div(non_comp_list)]))
    return sections

@app.callback(
    list(map(lambda o: Output("{}_details".format(o), 'children'), explainability_metrics)),
    Input('result', 'data'),
    State('toggle_supervised_unsupervised', 'on'), prevent_initial_call=False)
def metric_detail(data, unsupervised):
  if data is None:
      return [], [], [], []
  else:
      output = []
      result = json.loads(data)
      properties = result["properties"]
      for metric in explainability_metrics:
          metric_properties = properties["explainability"][metric]
          if not metric_properties:
              output.append(html.Div())
          else:
              prop = []
              for k, p in metric_properties.items():

                  if k == "importance" :
                        importance = p[1]
                        pct_dist = metric_properties["pct_dist"][1]
                        pct_mark = importance["labels"][int(float(pct_dist[:-1])/100 * len(importance["value"]))-1]
                        
                        fig = go.Figure(data=go.Scatter(x=importance["labels"][:30], y=importance["value"][:30],mode='lines+markers'))
                        fig.add_vline(x=pct_mark, line_width=3, line_dash="dash")
                        fig.add_trace(go.Scatter(
                            x=[importance["labels"][int(float(pct_dist[:-1])/100 * len(importance["value"]))-3]], y=[max(importance["value"])],
                            text="60% Threshold",
                            mode="text",
                        ))
                        fig.update_layout(showlegend=False)    
                        prop.append(html.Div(["Features importance of top 30 features:",
                            dcc.Graph(id='test',figure=fig, style={'display': 'block'})]))
                  else:
                      prop.append(html.Div("{} : {}".format(p[0], p[1])))
                  prop.append(html.Br())
              output.append(html.Div(prop))
      return output

'''
The following function updates
'''
@app.callback(
    Output("fairness_details", 'children'),
    [Input('result', 'data')], prevent_initial_call=True)
def fairness_metric_details(data):
    if data is None:
        return []
    else:
        result = json.loads(data)
        properties = result["properties"]
        FAIRNESS_SECTION_INDEX = 1
        metric_index = 0
        fairness_metrics_details = []
        fairness_metrics_details.append(html.H3("▶ Fairness Metrics"))
        calculated_metrics = []
        non_calculated_metrics = [html.H5("Non-Computable Metrics")]
        
        for metric_id, metric_score in (result["results"]["fairness"]).items():
            metric_properties = properties.get("fairness", {}).get(metric_id, {})
            if not math.isnan(metric_score):
                metric_index +=1
                calculated_metrics.append(show_metric_details_section(metric_id, metric_score, metric_properties, metric_index, FAIRNESS_SECTION_INDEX))
            else:
                non_calculated_metrics.append(show_metric_details_section(metric_id, metric_score, metric_properties))
        fairness_metrics_details.append(html.Div(calculated_metrics))
        fairness_metrics_details.append(html.Div(non_calculated_metrics))
        return html.Div(fairness_metrics_details)
 
@app.callback(
    [Output('training_data', 'data'),
     Output('test_data', 'data')],
    [Input('solution_set_dropdown', 'value')], prevent_initial_call=True)
def load_data(solution_set_path):
    if solution_set_path:
        training_data = read_train(solution_set_path)
        test_data = read_test(solution_set_path)
        return training_data.to_json(date_format='iso', orient='split'), test_data.to_json(date_format='iso', orient='split'), 
    else:
        return None, None

# === METHODOLOGY ===
@app.callback(
    [Output("trust_score", 'children')],
    [Input('result', 'data')], prevent_initial_call=True)
def trust_score(analysis):
    if analysis:
        analysis = json.loads(analysis)
        score = analysis["trust_score"]
        return ["{}/5".format(score)]
    else:
        return ["{}/5".format(NO_SCORE)]
    
@app.callback(
    [Output("methodology_score", 'children')],
    [Input('result', 'data')], prevent_initial_call=True)
def methodology_score(analysis):
    if analysis:
        analysis = json.loads(analysis)
        score = analysis["final_score"]["methodology"]
        return ["{}/5".format(score)]
    else:
        return ["{}/5".format(NO_SCORE)]
  
@app.callback(
    [Output("fairness_score", 'children')],
    [Input('result', 'data')], prevent_initial_call=True)
def fairness_score(analysis):
    if analysis:
        analysis = json.loads(analysis)
        score = analysis["final_score"]["fairness"]
        return ["{}/5".format(score)]
    else:
        return ["{}/5".format(NO_SCORE)]

@app.callback(
    [Output("robustness_score", 'children')],
    [Input('result', 'data')], prevent_initial_call=True)
def robustness_score(analysis):
    if analysis:
        analysis = json.loads(analysis)
        score = analysis["final_score"]["robustness"]
        return ["{}/5".format(score)]
    else:
        return ["{}/5".format(NO_SCORE)]
    
@app.callback(
    [Output("explainability_score", 'children')],
    [Input('result', 'data')], prevent_initial_call=True)
def explainability_score(analysis):
    if analysis:
        analysis = json.loads(analysis)
        score = analysis["final_score"]["explainability"]
        return ["{}/5".format(score)]
    else:
        return ["{}/5".format(NO_SCORE)]

@app.callback(
    [Output("f1_score_details", 'children'), Output("f1_score_score", 'children')],
    [Input('result', 'data')])
def f1_score(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["methodology"]["f1_score"]
        metric_scores = result["results"]
        if math.isnan(metric_scores["methodology"]["f1_score"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["methodology"]["f1_score"]))


# --- Normalization ---
@app.callback(
    [Output("normalization_details", 'children'), Output("normalization_score", 'children')],
    [Input('result', 'data')])
def normalization(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["methodology"]["normalization"]
        metric_scores = result["results"]
        if math.isnan(metric_scores["methodology"]["normalization"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["methodology"]["normalization"]))

#permutation feature importance
@app.callback(
    [Output("permutation_feature_importance_details", 'children'), Output("permutation_feature_importance_score", 'children')],
    [Input('result', 'data')])
def perm_feat_importance_details(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["explainability"]["permutation_feature_importance"]
        metric_scores = result["results"]

        if math.isnan(metric_scores["explainability"]["permutation_feature_importance"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["explainability"]["permutation_feature_importance"]))

'''
#model size unsupervised
@app.callback(
    [Output("model_size_details", 'children'), Output("model_size_score", 'children')],
    [Input('result', 'data')])
def model_size_details(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["explainability"]["model_size"]
        metric_scores = result["results"]

        if math.isnan(metric_scores["explainability"]["model_size"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["explainability"]["model_size"]))
'''
@app.callback(
    [Output("test_accuracy_details", 'children'), Output("test_accuracy_score", 'children')],
    [Input('result', 'data')])
def test_accuracy(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["methodology"]["test_accuracy"]
        metric_scores = result["results"]
        if math.isnan(metric_scores["methodology"]["test_accuracy"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["methodology"]["test_accuracy"]))

@app.callback(
    [Output("missing_data_details", 'children'), Output("missing_data_score", 'children')],
    [Input('result', 'data')])
def missing_data(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["methodology"]["missing_data"]
        metric_scores = result["results"]
        if math.isnan(metric_scores["methodology"]["missing_data"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["methodology"]["missing_data"]))


# --- Regularization ---
@app.callback(
    [Output("regularization_details", 'children'), Output("regularization_score", 'children')],
    [Input('result', 'data'),
     State('solution_set_dropdown', 'value')], prevent_initial_call=True)
def regularization(analysis, solution_set_path):
    if analysis and solution_set_path:
          analysis = json.loads(analysis)
          _, metric_scores, metric_properties = analysis["final_score"] , analysis["results"], analysis["properties"]
          metric_score = metric_scores["methodology"]["regularization"]
          regularization_technique = metric_properties["methodology"]["regularization"]
          if math.isnan(metric_score):
              return metric_detail_div(metric_properties), []
          return metric_detail_div(regularization_technique), html.H4("({}/5)".format(metric_score))
    else:
        return [], []
    
# --- Train Test Split ---
@app.callback(
    [Output("train_test_split_details", 'children'), Output("train_test_split_score", 'children')],
    [Input('result', 'data')])
def train_test_split(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["methodology"]["train_test_split"]
        metric_scores = result["results"]
        if math.isnan(metric_scores["methodology"]["train_test_split"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4(
            "({}/5)".format(metric_scores["methodology"]["train_test_split"]))

# --- Factsheet Completeness ---
@app.callback(
    [Output("factsheet_completeness_details", 'children'), Output("factsheet_completeness_score", 'children')],
    [Input('result', 'data'),
     State('solution_set_dropdown', 'value')], prevent_initial_call=True)
def factsheet_completeness(analysis, solution_set_path):
      if analysis is None:
          return [], []
      else:
          analysis = json.loads(analysis)
          _, metric_scores, metric_properties = analysis["final_score"] , analysis["results"], analysis["properties"]
          metric_score = metric_scores["methodology"]["factsheet_completeness"]
          metric_properties= metric_properties["methodology"]["factsheet_completeness"]
          return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_score))

@app.callback(
    Output(component_id="trust_section", component_property='style'),
    Output(component_id="fairness_section", component_property='style'),
    Output(component_id="explainability_section", component_property='style'),
    Output(component_id="robustness_section", component_property='style'),
    Output(component_id="methodology_section", component_property='style'),
    [Input("bar", 'figure')], prevent_initial_call=True)
def toggle_pillar_section_visibility(path):
    if path is not None:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
   
@app.callback(
    [Output('analyze_alert_section', 'children'),
    Output('analysis_section', 'style')],
    [Input('solution_set_dropdown', 'value')], prevent_initial_call=True)
def analyze_solution_completeness(solution_set_path):
    button = []
    alerts = []
    style={'display': 'none'}
    if solution_set_path is not None:
        style={'display': 'block'}
        button = dbc.Button(
            html.I(className="fas fa-backspace"),
            id="delete_button", 
            n_clicks=0,
            style={"float": "right", "backgroundColor": SECONDARY_COLOR}
        ),
        
        factsheet_path = "{}/factsheet.*".format(solution_set_path)
        test_data_path = "{}/test.*".format(solution_set_path)
        training_data_path = "{}/train.*".format(solution_set_path)
        model_path = "{}/model.*".format(solution_set_path)


        if not glob.glob(factsheet_path):
            alerts.append(html.H5("No factsheet provided", className="text-center", style={"color":"Red"}))
        if not glob.glob(training_data_path):
            alerts.append(html.H5("No training data provided", className="text-center", style={"color":"Red"}))
        if not glob.glob(test_data_path):
            alerts.append(html.H5("No test data provided", className="text-center", style={"color":"Red"}))
        if not glob.glob(model_path):
            alerts.append(html.H5("No model provided", className="text-center", style={"color":"Red"}))
        if alerts:
            style={'display': 'none'}
            alerts.append(html.H5("Please provide a complete dataset", className="text-center", style={"color":"Red"}))

    return alerts, style

@app.callback(Output('delete_solution_confirm', 'displayed'),
              Input('delete_solution_button', 'n_clicks'), prevent_initial_call=True)
def display_confirm(n_clicks):
    if n_clicks:
        return True
    else:
        return False
    
@app.callback(Output('performance_metrics_section', 'children'),
              Input('solution_set_dropdown', 'value'),
              State('toggle_supervised_unsupervised', 'on'), prevent_initial_call=True)
def show_performance_metrics(solution_set_path, unsupervised):
    if unsupervised:
        return []
    if not solution_set_path:
        return []
    else:
        test_data, training_data, model, factsheet = read_solution(solution_set_path)
        target_column = factsheet.get("general", {}).get("target_column", "")

        performance_metrics =  get_performance_metrics(model, test_data, target_column)
        performance_metrics_table = dash_table.DataTable(
                                id='performance_metrics_table',
                                columns=[{"name": i, "id": i} for i in performance_metrics.columns],
                                data=performance_metrics.to_dict('records'),
                                style_table={
                                    "width": "100%",
                                    'overflowX': 'hidden',
                                    'textAlign': 'left'
                                },
                                style_data={
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                },
                                style_header={
                                    'backgroundColor': SECONDARY_COLOR,
                                },
                                style_cell={
                                    'textAlign': 'left',
                                    'backgroundColor': SECONDARY_COLOR,
                                },
                                style_cell_conditional=[
                                    {
                                        'if': {'column_id': 'key'},
                                        'fontWeight': 'bold',
                                        'width': '30%'
                                    } 
                                ],
                                style_as_list_view=True,
                                css=[
                                  {
                                     'selector': 'tr:first-child',
                                     'rule': 'display: none',
                                  },
                                ],
        )
        return html.Div([html.H5("Performance Metrics"), performance_metrics_table], className="mt-4 mb-4 p-2", style={"border": "4px solid {}".format(TRUST_COLOR)})


@app.callback(Output('properties_section', 'children'),
              [Input('result', 'data'),
               State('solution_set_dropdown', 'value'),
               State('toggle_supervised_unsupervised', 'on')], prevent_initial_call=True)
def show_properties(data, solution_set_path, unsupervised):
    if data is None:
        return []
    else:
        if not unsupervised:
            test_data, training_data, model, factsheet = read_solution(solution_set_path)
            properties = get_properties_section(training_data, test_data, factsheet)
        else:
            test_data, training_data, outliers_data, model, factsheet = read_solution_unsupervised(solution_set_path)
            properties = get_properties_section_unsupervised(training_data, test_data, factsheet)

        if properties is None:
            return []
        properties_table = dash_table.DataTable(
            id='properties_table',
            columns=[{"name": i, "id": i} for i in properties.columns],
            data=properties.to_dict('records'),
            style_table={
                "width": "100%",
                'overflowX': 'hidden',
                'textAlign': 'left'
            },
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': SECONDARY_COLOR,
            },
            style_cell={
                'textAlign': 'left',
                'backgroundColor': SECONDARY_COLOR,
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'key'},
                    'fontWeight': 'bold',
                    'width': '30%'
                }
            ],
            style_as_list_view=True,
            css=[
                {
                    'selector': 'tr:first-child',
                    'rule': 'display: none',
                },
            ],
        )
        return html.Div([html.H5("Properties"), properties_table], className="mt-4 mb-4")


@app.callback(Output('result', 'data'), 
          [Input('solution_set_dropdown', 'value'),
           Input("input-config","data"),Input('input-mappings', 'data')],
          [State("recalc","on"),
           State('toggle_supervised_unsupervised', 'on')], prevent_initial_call = True)
def store_trust_analysis(solution_set_dropdown, config_weights, config_mappings,recalc, unsupervised):

    if not solution_set_dropdown:
        return None

    if unsupervised:
        with open('configs/unsupervised/weights/default.json', 'r') as f:
            default_weight = json.loads(f.read())
        with open('configs/unsupervised/mappings/default.json', 'r') as f:
            default_map = json.loads(f.read())

        weight_config = default_weight
        mappings_config = default_map

        test, train, outliers, model, factsheet = read_solution_unsupervised(solution_set_dropdown)
        final_score, results, properties = get_final_score_unsupervised(model, train, test, outliers, weight_config, mappings_config,
                                                           factsheet, solution_set_dropdown, recalc)
        trust_score = get_trust_score(final_score, weight_config["pillars"])

    else:
        with open('configs/supervised/weights/default.json', 'r') as f:
            default_weight = json.loads(f.read())
        with open('configs/supervised/mappings/default.json', 'r') as f:
            default_map = json.loads(f.read())

        if not config_weights:
            weight_config = default_weight
        else:
            weight_config = json.loads(config_weights)

        if not config_mappings:
            mappings_config = default_map
        else:
            mappings_config = json.loads(config_mappings)

        test, train, model, factsheet = read_solution(solution_set_dropdown)
        final_score, results, properties = get_final_score(model, train, test, weight_config, mappings_config, factsheet, solution_set_dropdown, recalc)
        trust_score = get_trust_score(final_score, weight_config["pillars"])


    def convert(o):
        if isinstance(o, np.int64): return int(o)


    data = {"final_score":final_score,
            "results":results,
            "trust_score":trust_score,
            "properties" : properties}
    return json.dumps(data,default=convert)
  
@app.callback(
      [Output('bar', 'figure'),
       Output('spider', 'figure'),
       Output('fairness_bar', 'figure'),
       Output('explainability_bar', 'figure'),
       Output('robustness_bar', 'figure'),
       Output('methodology_bar', 'figure'),
       Output('fairness_spider', 'figure'),
       Output('explainability_spider', 'figure'),
       Output('robustness_spider', 'figure'),
       Output('methodology_spider', 'figure'),
       Output('trust_star_rating', 'children'),
       Output('fairness_star_rating', 'children'),
       Output('explainability_star_rating', 'children'),
       Output('robustness_star_rating', 'children'),
       Output('methodology_star_rating', 'children')],
      [Input('result', 'data'),Input("hidden-trigger", "value")],
    State('toggle_supervised_unsupervised', 'on'))
def update_figure(data, trig, unsupervised):
    global charts
    charts = []

    if data is None:
        return [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "", "", "", "", ""]
    result = json.loads(data)
    final_score, results, properties = result["final_score"] , result["results"], result["properties"]
    trust_score = result["trust_score"]
    pillars = list(map(lambda x: x.upper(),list(final_score.keys())))
    values = list(final_score.values())

    colors = [FAIRNESS_COLOR, EXPLAINABILITY_COLOR, ROBUSTNESS_COLOR, METHODOLOGY_COLOR]

    # barchart
    chart_list=[]
    bar_chart = go.Figure(data=[go.Bar(
        x=pillars,
        y=values,
        marker_color=colors
            )])
    bar_chart.update_yaxes(range=[0, 5], fixedrange=True)
    bar_chart.update_layout(title_text='', title_x=0.5, paper_bgcolor='#FFFFFF', plot_bgcolor=SECONDARY_COLOR)
    chart_list.append(bar_chart)
    charts.append(bar_chart)

    #spider
    radar_chart = px.line_polar(r=values, theta=pillars, line_close=True, title='')
    radar_chart.update_layout(title_x=0.5)
    radar_chart.update_traces(fill='toself', fillcolor=TRUST_COLOR, marker_color=TRUST_COLOR, marker_line_color=TRUST_COLOR, marker_line_width=0, opacity=0.6)
    chart_list.append(radar_chart)

    #barcharts
    for n, (pillar , sub_scores) in enumerate(results.items()):
        print(n, pillar, sub_scores)
        title = "<b style='font-size:32px;''>{}/5</b>".format(final_score[pillar])
        categories = list(map(lambda x: x.replace("_",' ').title(), sub_scores.keys()))
        values = list(map(float, sub_scores.values()))
        if np.isnan(values).any():
            nonNanCategories = list()
            nonNanValues = list()
            for c, v in zip(categories, values):
                if not np.isnan(v):
                    nonNanCategories.append(c)
                    nonNanValues.append(v)
            categories = nonNanCategories
            values = nonNanValues
        print(categories)
        if not unsupervised:
            desc = list(map(lambda x: metric_description[x.lower().replace(' ','_')], categories))
        else:
            desc = list(map(lambda x: metric_description_unsupervised[x.lower().replace(' ','_')], categories))

        bar_chart_pillar = go.Figure(data=[go.Bar(x=categories, y=values, customdata = desc, marker_color=colors[n],hovertemplate = "(%{x}: %{y})<br>%{customdata}<extra></extra>")])
        bar_chart_pillar.update_yaxes(range=[0, 5], fixedrange=True)
        #bar_chart_pillar.update_yaxes(fixedrange=True)
        #bar_chart_pillar.update_yaxes(range=[0, 8])
        bar_chart_pillar.update_layout(title_text='', title_x=0.5, xaxis_tickangle=XAXIS_TICKANGLE, paper_bgcolor='#FFFFFF', plot_bgcolor=SECONDARY_COLOR)


          #fig.update_layout(barmode='group', xaxis_tickangle=-45)
        chart_list.append(bar_chart_pillar)
        charts.append(bar_chart_pillar)

    #spider charts
    for n, (pillar , sub_scores) in enumerate(results.items()):
        title = "<b style='font-size:32px;''>{}/5</b>".format(final_score[pillar])
        categories = list(map(lambda x: x.replace("_",' ').title(), sub_scores.keys()))
        val = list(map(float, sub_scores.values()))
        exc = np.isnan(val)
        r = np.array(val)[~exc]
        theta=np.array(categories)[~exc]
        radar_chart_pillar = px.line_polar(r=r, theta=theta, line_close=True, title='')
        radar_chart_pillar.update_traces(fill='toself', fillcolor=colors[n], marker_color=colors[n],marker_line_width=1.5, opacity=0.6)
        radar_chart_pillar.update_layout(title_x=0.5)
        chart_list.append(radar_chart_pillar)
    star_ratings = []
    star_ratings.append(show_star_rating(trust_score))
    star_ratings.append(show_star_rating(final_score["fairness"]))
    star_ratings.append(show_star_rating(final_score["explainability"]))
    star_ratings.append(show_star_rating(final_score["robustness"]))
    star_ratings.append(show_star_rating(final_score["methodology"]))
    return chart_list + star_ratings

@app.callback(
    Output("robustness_details", 'children'),
    Input('result', 'data'), prevent_initial_call=True)
def robustness_details(data):
    if not data:
        return []
    result = json.loads(data)
    properties = result["properties"]
    metrics = list(properties["robustness"].keys())

    sections = [html.H3("▶ Robustness Metrics")]
    non_calculated_metrics = []
    count = 0
    for i in range(len(metrics)):
        metric_id = metrics[i]
        score = result["results"]["robustness"][metric_id]
        if not math.isnan(score):
            sections.append(create_metric_details_section(metric_id, count, 3, True,score ))
            count = count + 1
        else:
            non_calculated_metrics.append(create_metric_details_section(metric_id, count, 3, True,score ))
    sections.append(html.H5("Non-Computable Metrics") if not [] == non_calculated_metrics else [])
    sections.append(html.Div(non_calculated_metrics))
    return sections

@app.callback(
[Output("er_deepfool_attack_details", 'children'), Output("er_deepfool_attack_score", 'children')],
Input('result', 'data'),
State('toggle_supervised_unsupervised', 'on'), prevent_initial_call=False)
def Deepfool_Attack_metric_detail(data, unsupervised):
    if unsupervised:
        return [], []
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["robustness"]["er_deepfool_attack"]
        metric_scores = result["results"]
        if math.isnan(metric_scores["robustness"]["er_deepfool_attack"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["robustness"]["er_deepfool_attack"]))

@app.callback(
[Output("er_carlini_wagner_attack_details", 'children'), Output("er_carlini_wagner_attack_score", 'children')],
Input('result', 'data'),
State('toggle_supervised_unsupervised', 'on'), prevent_initial_call=False)
def carlini_wagner_attack_analysis(data, unsupervised):
    if unsupervised:
        return [],[]

    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["robustness"]["er_carlini_wagner_attack"]
        metric_scores = result["results"]
        if math.isnan(metric_scores["robustness"]["er_carlini_wagner_attack"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["robustness"]["er_carlini_wagner_attack"]))

@app.callback(
    [Output("er_fast_gradient_attack_details", 'children'), Output("er_fast_gradient_attack_score", 'children')],
    Input('result', 'data'),
    State('toggle_supervised_unsupervised', 'on'), prevent_initial_call=False)
def fast_gradient_attack_analysis(data, unsupervised):
    if unsupervised:
        return [],[]

        if data is None:
            return [], []
        else:
            result = json.loads(data)
            properties = result["properties"]
            metric_properties = properties["robustness"]["er_fast_gradient_attack"]
            metric_scores = result["results"]
            if math.isnan(metric_scores["robustness"]["er_fast_gradient_attack"]):
                return metric_detail_div(metric_properties), []
            return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["robustness"]["er_fast_gradient_attack"]))

@app.callback(
    [Output("clique_method_details", 'children'), Output("clique_method_score", 'children')],
    Input('result', 'data'),
    State('toggle_supervised_unsupervised', 'on'), prevent_initial_call=False)
def clique_method_analysis(data, unsupervised):
    if unsupervised:
        return [], []

    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["robustness"]["clique_method"]
        metric_scores = result["results"]
        if math.isnan(metric_scores["robustness"]["clique_method"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["robustness"]["clique_method"]))

@app.callback(
    [Output("confidence_score_details", 'children'), Output("confidence_score_score", 'children')],
    Input('result', 'data'),
    State('toggle_supervised_unsupervised', 'on'), prevent_initial_call=False)
def confidence_analysis(data, unsupervised):
    if unsupervised:
        return [], []

    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["robustness"]["confidence_score"]
        metric_scores = result["results"]
        if math.isnan(metric_scores["robustness"]["confidence_score"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4(
            "({}/5)".format(metric_scores["robustness"]["confidence_score"]))

@app.callback(
    [Output("loss_sensitivity_details", 'children'), Output("loss_sensitivity_score", 'children')],
    Input('result', 'data'),
    State('toggle_supervised_unsupervised', 'on'), prevent_initial_call=False)
def loss_sensitivity_analysis(data, unsupervised):
    if unsupervised:
        return [], []

    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["robustness"]["loss_sensitivity"]
        metric_scores = result["results"]
        if math.isnan(metric_scores["robustness"]["loss_sensitivity"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4(
            "({}/5)".format(metric_scores["robustness"]["loss_sensitivity"]))


@app.callback(
    [Output("clever_score_details", 'children'), Output("clever_score_score", 'children')],
    Input('result', 'data'), prevent_initial_call=False)
def clever_score(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["robustness"]["clever_score"]
        metric_scores = result["results"]
        if math.isnan(metric_scores["robustness"]["clever_score"]):
            return metric_detail_div(metric_properties), []
        return metric_detail_div(metric_properties), html.H4(
            "({}/5)".format(metric_scores["robustness"]["clever_score"]))
 
@app.callback(
    Output("solution_set_dropdown", 'options'),
    Input('scenario_dropdown', 'value'),
    State('toggle_supervised_unsupervised', 'on'), prevent_initial_call=False)
def show_scenario_solution_options(scenario_id, unsupervised):
    print("unsupervised: ", unsupervised)
    if scenario_id and not unsupervised:
        solutions = get_scenario_solutions_options(scenario_id)
        return solutions
    elif scenario_id and unsupervised:
        solutions = get_scenario_solutions_options_unsupervised(scenario_id)
        return solutions
    else:
        return []

@app.callback(
    [Output("modal-report", "is_open"),Output("download-report", "data")],
    Input('download_report_button', 'n_clicks'), 
    [State('solution_set_dropdown', 'value'),
     State("modal-report", "is_open"),
     State('result', 'data'),
     State("config-dropdown", "value")
     ] + list(map(lambda x: State("mapping-dropdown-{}".format(x), 'value' ), SECTIONS[1:])) , prevent_initial_call=True)
def download_report(n_clicks, solution_set_path, is_open, data, weight, map_f, map_e, map_r, map_m):
    if n_clicks and solution_set_path:
        configs = [weight, map_f, map_e, map_r, map_m]
        result = json.loads(data)
        test_data, training_data, model, factsheet = read_solution(solution_set_path)
        target_column = factsheet.get("general", {}).get("target_column", "")
        save_report_as_pdf(result, model, test_data, target_column, factsheet,  charts, configs)
        data = send_file("report.pdf")
        del data["mime_type"]
        return is_open, data
    else:
        return is_open, data


@app.callback([Output("scenario_dropdown", 'value'),
               Output("solution_set_dropdown", 'value')],
    [Input('uploaded_scenario_id', 'data'),
    Input('uploaded_solution_id', 'data')])
def set_uploaded_model(scenario_id, solution_id):
    if scenario_id and solution_id:
        solution_path = get_solution_path(scenario_id, solution_id)
        return scenario_id, solution_path
    else:
        return None, None

@app.callback(
    Output("scenario_dropdown", "options"),
    Input('toggle_supervised_unsupervised', 'on'), prevent_initial_call = True
)
def toggle_mode_analyze(unsupervised):

    print("toggle on analyze site")
    return get_scenario_options(unsupervised)

config_panel.get_callbacks(app)
    
# === LAYOUT ===
layout = html.Div([
    config_panel.layout,
    dbc.Container([
        dbc.Row([
            dcc.Store(id='result'),
            
            dbc.Col([html.H1("Analyze", className="text-center")], width=12, className="mb-2 mt-1"),
            html.Div([daq.BooleanSwitch(id='recalc',
                      on=False,
                      label='Recompute',
                      labelPosition="top",
                      color = "green",
                    
                    )], style= {'display': 'Block'} if DEBUG else {"display": "None"}),
            # daq.BooleanSwitch(id='toggle_supervised_unsupervised_analyze',
            #           on=False,
            #           label="enable Unsupervised Mode",
            #           labelPosition="top",
            #           color = TRUST_COLOR,
            #           style={"float": "right",'margin-left': "44%"}
            #         ),
            dbc.Col([html.H5("Scenario"),
                 dcc.Dropdown(
                    id='scenario_dropdown',
                    options= get_scenario_options(),
                    value = None,
                    placeholder='Select Scenario'
                )], width=12, style={"marginLeft": "0 px", "marginRight": "0 px"}, className="mb-1 mt-1"
            ),
            dbc.Col([html.H5("Solution"),
                dcc.Dropdown(
                    id='solution_set_dropdown',
                    options = [],
                    value=None,

                    placeholder='Select Solution'
                )], width=12, style={"marginLeft": "0 px", "marginRight": "0 px"}, className="mb-1 mt-1"
            ),
            dbc.Modal([
                    dbc.ModalHeader("Success"),
                    dbc.ModalBody([dbc.Alert(id="report-success",children ="You successfully saved the report", color="success"),]),
            ],
                    id="modal-report", is_open=False, backdrop=True),
            dbc.Col([
                dcc.ConfirmDialog(
                    id='delete_solution_confirm',
                    message='Are you sure that you want to delete this solution?',
                ),
                html.Div([], id="delete_solution_alert"),
                html.Div([], id="analyze_alert_section"),

                html.Div([
                    html.Div([dbc.Button("Download Report", id='download_report_button', color="primary", className="mt-3", style={"width": "30%"}),
                              dcc.Download(id="download-report", type="application/pdf")],
                         className="text-center"),html.Br(),
                    general_section(),
                    trust_section(),
                    html.Div([
                    # html.H2("Pillar Sections", className="text-center"),
                    dbc.Row(
                    [
                    dbc.Col(dbc.Button("FAIRNESS", id='fairness_s',className="mt-3", color="primary", style={"background-color": FAIRNESS_COLOR})),
                    dbc.Col(dbc.Button("EXPLAINABILITY", id='explainability_s',className="mt-3", color="primary", style={"background-color": EXPLAINABILITY_COLOR})),
                    dbc.Col(dbc.Button("ROBUSTNESS", id='robustness_s',className="mt-3", color="primary" , style={"background-color": ROBUSTNESS_COLOR})),
                    dbc.Col(dbc.Button("METHODOLOGY", id='methodology_s',className="mt-3", color="primary" , style={"background-color": METHODOLOGY_COLOR})),
                    ]
                    ), html.Br()],className="text-center", style={"margin-left":"10%","margin-right":"10%"}),
                    html.Div([
                        dbc.Col(dbc.Button("Show ALL", id='all_s',className="mt-3", color="primary" , style={"background-color": TRUST_COLOR, "border-radius":"80%"}))
                        ],className="text-center"),
                    pillar_section("fairness", fairness_metrics),
                    pillar_section("explainability", explainability_metrics),
                    pillar_section("robustness", robustness_metrics),
                    pillar_section("methodology", methodology_metrics),
                    dcc.Store(id='training_data'),
                    dcc.Store(id='test_data'),
                    # html.Div([dbc.Button("Download Report", id='download_report_button', color="primary", className="mt-3"),
                    #           dcc.Download(id="download-report", type="application/pdf")],
                    #      className="text-center"),
                ], id="analysis_section")
            ],
                width=12, 
                className="mt-2 pt-2 pb-2 mb-2",
                style={
                    "border": "1px solid #d8d8d8",
                    "borderRadius": "6px",
                    "backgroundColor": SECONDARY_COLOR
                }   
            ),
        ], no_gutters=False)
    ])
])
