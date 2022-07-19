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



# === SECTIONS ===
def general_section_1():
    return html.Div([
                html.Div([html.H2("• General Information")]),
                html.Div([], id="general_description-1"),
                dbc.Row([
                    dbc.Col(html.Div(["Performance Metrics Section"], id="performance_metrics_section-1")),
                    dbc.Col(html.Div(["Properties Section"], id="properties_section-1"))
                ])
                ])

def general_section_2():
    return html.Div([
                html.Div([html.H2("• General Information")]),
                html.Div([], id="general_description-2"),
                dbc.Row([
                    dbc.Col(html.Div(["Performance Metrics Section"], id="performance_metrics_section-2")),
                    dbc.Col(html.Div(["Properties Section"], id="properties_section-2"))
                ])
                ])

def trust_section_1():
    return html.Div([
        html.Div(id='boolean-switch-output'),
        html.Div([daq.BooleanSwitch(id='toggle_charts-1',
                                    on=False,
                                    label='Alternative Style',
                                    labelPosition="top",
                                    color="green",
                                    style={"float": "right"}
                                    )], className="mt-2"),
        html.H2("Trustworthiness", className="text-center"),

        html.Div([], id="trust_overview-1"),
        html.H3("Overall Score", className="text-center"),
        html.Div([], id="trust_star_rating-1", className="star_rating, text-center"),
        dcc.Graph(id='spider-1', style={'display': 'none'}),
        dcc.Graph(id='bar-1', style={'display': 'none'}),
        html.Div([], id="trust_details-1"),
        html.Hr()
    ], id="trust_section-1", style={"display": "None"})

def trust_section_2():
    return html.Div([
        html.Div(id='boolean-switch-output'),
        html.Div([daq.BooleanSwitch(id='toggle_charts-2',
                                    on=False,
                                    label='Alternative Style',
                                    color="green",
                                    style={"float": "right"}
                                    )], className="mt-2"),
        html.H2("Trustworthiness", className="text-center"),

        html.Div([], id="trust_overview-2"),
        html.H3("Overall Score", className="text-center"),
        html.Div([], id="trust_star_rating-2", className="star_rating, text-center"),
        dcc.Graph(id='spider-2', style={'display': 'none'}),
        dcc.Graph(id='bar-2', style={'display': 'block'}),
        html.Div([], id="trust_details-2"),
        html.Hr()
    ], id="trust_section-2", style={"display": "None"})


def pillar_section_1(pillar):
    return html.Div([
        html.Div([], id="{}_overview-1".format(pillar)),
        html.H3("{0} Score".format(pillar), className="text-center"),
        html.Div([], id="{}_star_rating-1".format(pillar), className="star_rating, text-center"),
        dcc.Graph(id='{}_spider-1'.format(pillar), style={'display': 'none'}),
        dcc.Graph(id='{}_bar-1'.format(pillar), style={'display': 'block'}),
        dbc.Collapse(
            html.P("{} Details".format(pillar)),
            id="{}_details-1".format(pillar),
            is_open=False,
        ),
        html.Hr(style={"size": "10"}),

    ], id="{}_section-1".format(pillar), style={"display": "None"})

def pillar_section_2(pillar):
    return html.Div([
        html.Div([], id="{}_overview-2".format(pillar)),
        html.H3("{0} Score".format(pillar), className="text-center"),
        html.Div([], id="{}_star_rating-2".format(pillar), className="star_rating, text-center"),
        dcc.Graph(id='{}_spider-2'.format(pillar), style={'display': 'none'}),
        dcc.Graph(id='{}_bar-2'.format(pillar), style={'display': 'block'}),
        dbc.Collapse(
            html.P("{} Details".format(pillar)),
            id="{}_details-2".format(pillar),
            is_open=False,
        ),
        html.Hr(style={"size": "10"}),

    ], id="{}_section-2".format(pillar), style={"display": "None"})

pillars = ['fairness', 'explainability', 'robustness', 'methodology']

def map_dropdown(pillar):
    return html.Div(
        [html.Br(),
        html.Div(html.Label("{} mapping".format(pillar)), style={ 'display': 'inline-block',"vertical-align": "top",'margin-left': "40%"}),
        html.Br(),
        html.Div(dcc.Dropdown(
                    id='{}-dropdown-compare'.format(pillar),
                    options=list(map(lambda name:{'label': name[:-5], 'value': "configs/supervised/mappings/{}/{}".format(pillar,name)} ,os.listdir("configs/supervised/mappings/{}".format(pillar)))),
                    value='configs/supervised/mappings/{}/default.json'.format(pillar)
                ), 
                 style={'width': "20%", 'display': 'inline-block',"vertical-align": "top",'margin-left': "40%"})
        ]
        )

@app.callback(
    [Output("solution_set_dropdown-1", 'options'),Output("solution_set_dropdown-2", 'options')],
    [Input('scenario_dropdown_compare', 'value'), Input('toggle_supervised_unsupervised_compare', 'on')], prevent_initial_call=False)
def load_solution_sets(scenario_id, unsupervised):
    if scenario_id and not unsupervised:
        return get_scenario_solutions_options(scenario_id), get_scenario_solutions_options(scenario_id)
    elif scenario_id and unsupervised:
        print("hello")
        return get_scenario_solutions_options_unsupervised(scenario_id), get_scenario_solutions_options_unsupervised(scenario_id)
    else:
        return [], []

@app.callback(
    Output("scenario_dropdown_compare", "options"),
    Input('toggle_supervised_unsupervised_compare', 'on'), prevent_initial_call = True
)
def toggle_mode_compare(unsupervised):

    print("toggle on compare site")
    print(get_scenario_options(unsupervised))
    return get_scenario_options(unsupervised)

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Compare", className="text-center"), width=12, className="mb-5"),
            daq.BooleanSwitch(id='toggle_config_compare',
                      on=False,
                      label="Show Configuration",
                      labelPosition="top",
                      color = TRUST_COLOR,
                      style={"float": "right",'margin-left': "44%"}
                    ),
            daq.BooleanSwitch(id='toggle_supervised_unsupervised_compare',
                      on=False,
                      label="enable Unsupervised",
                      labelPosition="top",
                      color = TRUST_COLOR,
                      style={"float": "right",'margin-left': "44%"}
                    ),
            html.Br(),
            dbc.Col(html.Div(
                [html.Br(),
                html.H4("Configurations", style = {"text-align":"center"}),
                html.Div(html.Label("Weight Configuration"), style={ 'display': 'inline-block',"vertical-align": "top",'margin-left': "40%"}),
                html.Br(),
                html.Div(dcc.Dropdown(
                            id='config-dropdown-compare',
                            options=list(map(lambda name:{'label': name[:-5], 'value': 'configs/supervised/weights/{}'.format(name)} ,os.listdir("configs/supervised/weights"))),
                            value='configs/supervised/weights/default.json'
                        ), 
                         style={'width': "20%", 'display': 'inline-block',"vertical-align": "top",'margin-left': "40%"}),
                html.Div(list(map(lambda pillar: map_dropdown(pillar) , pillars))),
                html.Button('Apply', id='apply-config-compare', style={"background-color": "gold","margin-top":15,"margin-left":"40%","width":"20%"})
                    ],style={"background-color": "rgba(255,228,181,0.5)",'padding-bottom': 20," margin-left": "auto", "margin-right": "auto"}), 
                width=12,style={'display': 'none'},id="compare-config"),
            dcc.Store(id='result-1'),
            dbc.Col([dcc.Dropdown(
                    id='scenario_dropdown_compare',
                    options= get_scenario_options(),
                    value = None,

                    placeholder='Select Scenario'
                )], width=12, style={"marginLeft": "0 px", "marginRight": "0 px"}, className="mb-1 mt-1"
            ),
            dbc.Col([dcc.Dropdown(
                id='solution_set_dropdown-1',
                options=get_scenario_solutions_options('it_sec_incident_classification'),
                placeholder='Select Solution A',
                value=None,
            )], width=6, style={"marginLeft": "0 px", "marginRight": "0 px"}, className="mb-1 mt-1"
            ),
            dcc.Store(id='result-2'),
            dbc.Col([dcc.Dropdown(
                id='solution_set_dropdown-2',
                options=get_scenario_solutions_options('it_sec_incident_classification'),
                placeholder='Select Solution B',
                value=None,
            )], width=6, className="mb-1 mt-1"
            ),
            dbc.Col([html.Div([], id="performance_div-1")], width=6, style={"width":"40%","marginLeft": "0 px", "marginRight": "0 px"},),
            dbc.Col([html.Div([], id="performance_div-2")], width=6, style={"width":"40%","marginLeft": "0 px", "marginRight": "0 px"}),

            dbc.Col([
                html.Div([], id="toggle_charts_section-1"),
                html.Div([
                    general_section_1(),
                    trust_section_1(),
                    pillar_section_1("fairness"),
                    pillar_section_1("explainability"),
                    pillar_section_1("robustness"),
                    pillar_section_1("methodology"),
                    dcc.Store(id='training_data-1'),
                    dcc.Store(id='test_data-1')
                ], id="analysis_section-1")
            ],
                width=6,
                className="mt-2 pt-2 pb-2 mb-2",
                style={
                    "border": "1px solid #d8d8d8",
                    "borderRadius": "6px",
                    "backgroundColor": SECONDARY_COLOR
                }
            ),
            dbc.Col([
                html.Div([], id="toggle_charts_section-2"),
                html.Div([
                    general_section_2(),
                    trust_section_2(),
                    pillar_section_2("fairness"),
                    pillar_section_2("explainability"),
                    pillar_section_2("robustness"),
                    pillar_section_2("methodology"),
                    dcc.Store(id='training_data-2'),
                    dcc.Store(id='test_data-2')
                ], id="analysis_section-2")
            ],
                width=6,
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

@app.callback(
    Output("compare-config", "style"),
    [Input("toggle_config_compare", "on")],
    prevent_initial_call=True
)
def toggle_config(on):
    print(on)
    #app.logger.info("toggle {0} detail section".format(s))
    if not on:
        return {'display': 'None'}
    else:
        return {'display': 'Block'}

@app.callback(
    Output(component_id="bar-1", component_property='style'),
    Output(component_id="fairness_bar-1", component_property='style'),
    Output(component_id="explainability_bar-1", component_property='style'),
    Output(component_id="robustness_bar-1", component_property='style'),
    Output(component_id="methodology_bar-1", component_property='style'),
    Output(component_id="spider-1", component_property='style'),
    Output(component_id="fairness_spider-1", component_property='style'),
    Output(component_id="explainability_spider-1", component_property='style'),
    Output(component_id="robustness_spider-1", component_property='style'),
    Output(component_id="methodology_spider-1", component_property='style'),
    [Input('toggle_charts-1', 'on'), Input('solution_set_dropdown-1', 'value')]
)
def toggle_charts_1(visibility_state, solution_set):
    if solution_set is None:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {
            'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {
                   'display': 'none'}, {'display': 'none'}
    if visibility_state == True:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {
            'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {
                   'display': 'block'}, {'display': 'block'}
    if visibility_state == False:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {
            'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {
                   'display': 'none'}

@app.callback(
    Output(component_id="bar-2", component_property='style'),
    Output(component_id="fairness_bar-2", component_property='style'),
    Output(component_id="explainability_bar-2", component_property='style'),
    Output(component_id="robustness_bar-2", component_property='style'),
    Output(component_id="methodology_bar-2", component_property='style'),
    Output(component_id="spider-2", component_property='style'),
    Output(component_id="fairness_spider-2", component_property='style'),
    Output(component_id="explainability_spider-2", component_property='style'),
    Output(component_id="robustness_spider-2", component_property='style'),
    Output(component_id="methodology_spider-2", component_property='style'),
    [Input('toggle_charts-2', 'on'), Input('solution_set_dropdown-2', 'value')]
)
def toggle_charts_2(visibility_state, solution_set):
    if solution_set is None:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {
            'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {
                   'display': 'none'}, {'display': 'none'}
    if visibility_state == True:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {
            'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {
                   'display': 'block'}, {'display': 'block'}
    if visibility_state == False:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {
            'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {
                   'display': 'none'}

# @app.callback(Output('solution_set_dropdown-1', 'options'),
#               Input('solution_set_dropdown-1', 'nclicks'))
# def update_solution_set_dropdown_1(n_clicks):
#     return get_solution_options()

# @app.callback(Output('solution_set_dropdown-2', 'options'),
#               Input('solution_set_dropdown-2', 'nclicks'))
# def update_solution_set_dropdown_2(n_clicks):
#     return get_solution_options()


@app.callback(
    [Output('training_data-1', 'data'),
     Output('test_data-1', 'data')],
    [Input('solution_set_dropdown-1', 'value')], prevent_initial_call=True)
def load_data_1(solution_set_path):
    if solution_set_path != None:
        training_data = read_train(solution_set_path)
        test_data = read_test(solution_set_path)
        return training_data.to_json(date_format='iso', orient='split'), test_data.to_json(date_format='iso', orient='split'),
    else:
        return None, None

@app.callback(
    [Output('training_data-2', 'data'),
     Output('test_data-2', 'data')],
    [Input('solution_set_dropdown-2', 'value')], prevent_initial_call=True)
def load_data_2(solution_set_path):
    if solution_set_path != None:
        training_data = read_train(solution_set_path)
        test_data = read_test(solution_set_path)
        return training_data.to_json(date_format='iso', orient='split'), test_data.to_json(date_format='iso', orient='split'),
    else:
        return None, None


@app.callback(
    Output(component_id="trust_section-1", component_property='style'),
    Output(component_id="fairness_section-1", component_property='style'),
    Output(component_id="explainability_section-1", component_property='style'),
    Output(component_id="robustness_section-1", component_property='style'),
    Output(component_id="methodology_section-1", component_property='style'),
    [Input("bar-1", 'figure')], prevent_initial_call=True)
def toggle_pillar_section_visibility_1(path):
    if path is not None:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {
            'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

@app.callback(
    Output(component_id="trust_section-2", component_property='style'),
    Output(component_id="fairness_section-2", component_property='style'),
    Output(component_id="explainability_section-2", component_property='style'),
    Output(component_id="robustness_section-2", component_property='style'),
    Output(component_id="methodology_section-2", component_property='style'),
    [Input("bar-2", 'figure')], prevent_initial_call=True)
def toggle_pillar_section_visibility_2(path):
    if path is not None:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {
            'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

#@app.callback(Output('performance_div-1', 'children'),
#              Input('solution_set_dropdown-1', 'value'))
#def get_performance_1(solution_set_dropdown):
#    if not solution_set_dropdown:
#        return html.P()
#    test, train, model, factsheet = read_scenario(solution_set_dropdown)
#    target_column = factsheet["general"].get("target_column")
#    performance = get_performance_table(model, test, target_column).transpose()
#    performance_table = dash_table.DataTable(
#        id='table-1',
#        columns=[{"name": i, "id": i} for i in performance.columns],
#        data=performance.to_dict('records'),
#        style_table={'overflowX': 'auto'},

#    )
#    performance_div = html.Div([html.H5("Performance metrics", style={"width": "100%", "text-align": "center", "margin-right": "auto",
#                                               "margin-left": "auto"}),
#                                performance_table],  style={ "width":"100%","marginLeft": "0 px", "marginRight": "0 px"})
#    return performance_div
#
#@app.callback(Output('performance_div-2', 'children'),
#              Input('solution_set_dropdown-2', 'value'))
#def get_performance_2(solution_set_dropdown):
#    if not solution_set_dropdown:
#        return html.P()
#    test, train, model, factsheet = read_scenario(solution_set_dropdown)
#    target_column = factsheet["general"].get("target_column")
#    performance = get_performance_table(model, test, target_column).transpose()
#    performance_table = dash_table.DataTable(
#        id='table-2',
#        columns=[{"name": i, "id": i} for i in performance.columns],
#        data=performance.to_dict('records'),
#        style_table={'overflowX': 'auto'},
#    )
#    performance_div = html.Div([html.H5("Performance metrics", style={"width": "100%", "text-align": "center", "margin-right": "auto",
#                                               "margin-left": "auto"}),
#                                performance_table],  style={ "width":"100%","marginLeft": "0 px", "marginRight": "0 px"})
#    return performance_div


@app.callback(Output('result-1', 'data'),
              [Input('solution_set_dropdown-1', 'value'),
              Input("apply-config-compare", "n_clicks")],
              [State('config-dropdown-compare', 'value')] +  [State('{}-dropdown-compare'.format(pillar),"value") for pillar in pillars],  State('toggle_supervised_unsupervised_compare', 'on'))
def store_result_1(solution_set_dropdown, n, weight, map_fairness, map_explainability, map_robustness, map_methodology, unsupervised):
    if not solution_set_dropdown:
        return None
    if unsupervised:
        weight = "configs/unsupervised/weights/default.json"
        map_fairness = "configs/unsupervised/mappings/fairness/default.json"
        map_explainability = "configs/unsupervised/mappings/explainability/default.json"
        map_robustness = "configs/unsupervised/mappings/robustness/default.json"
        map_methodology = "configs/unsupervised/mappings/methodology/default.json"

        with open(weight, 'r') as f:
            weights_config = json.loads(f.read())

        mappings_config = dict()

        for pillar, map_conf in zip(pillars, [map_fairness, map_explainability, map_robustness, map_methodology]):
            with open(map_conf, 'r') as f:
                mappings_config[pillar] = json.loads(f.read())

        test, train, outliers, model, factsheet = read_solution_unsupervised(solution_set_dropdown)
        final_score, results, properties = get_final_score_unsupervised(model, train, test, outliers, weights_config, mappings_config,
                                                           factsheet, solution_set_dropdown)
        trust_score = get_trust_score(final_score, weights_config["pillars"])

        def convert(o):
            if isinstance(o, np.int64): return int(o)

        data = {"final_score": final_score,
                "results": results,
                "trust_score": trust_score,
                "properties": properties}
        return json.dumps(data, default=convert)
    else:
        with open(weight, 'r') as f:
            weights_config = json.loads(f.read())

        mappings_config = dict()

        for pillar, map_conf in zip(pillars,[map_fairness, map_explainability, map_robustness, map_methodology]):
            with open(map_conf, 'r') as f:
                mappings_config[pillar] = json.loads(f.read())

        test, train, model, factsheet = read_solution(solution_set_dropdown)
        final_score, results, properties = get_final_score(model, train, test, weights_config, mappings_config, factsheet, solution_set_dropdown)
        trust_score = get_trust_score(final_score, weights_config["pillars"])
        def convert(o):
            if isinstance(o, np.int64): return int(o)
        data = {"final_score": final_score,
                "results": results,
                "trust_score": trust_score,
                "properties": properties}
        return json.dumps(data, default=convert)

@app.callback(Output('result-2', 'data'),
              [Input('solution_set_dropdown-2', 'value'),
               Input("apply-config-compare", "n_clicks")],
              [State('config-dropdown-compare', 'value')] +  [State('{}-dropdown-compare'.format(pillar),"value") for pillar in pillars],
              State('toggle_supervised_unsupervised_compare', 'on'))
def store_result_2(solution_set_dropdown, n, weight, map_fairness, map_explainability, map_robustness, map_methodology, unsupervised):
    if not solution_set_dropdown:
        return None
    if unsupervised:
        weight = "configs/unsupervised/weights/default.json"
        map_fairness = "configs/unsupervised/mappings/fairness/default.json"
        map_explainability = "configs/unsupervised/mappings/explainability/default.json"
        map_robustness = "configs/unsupervised/mappings/robustness/default.json"
        map_methodology = "configs/unsupervised/mappings/methodology/default.json"

        with open(weight, 'r') as f:
            weights_config = json.loads(f.read())

        mappings_config = dict()

        for pillar, map_conf in zip(pillars, [map_fairness, map_explainability, map_robustness, map_methodology]):
            with open(map_conf, 'r') as f:
                mappings_config[pillar] = json.loads(f.read())

        # mappings_config = dict(fairness=config_fairness["parameters"], explainability=config_explainability["parameters"],
        #                robustness=config_robustness["parameters"], methodology=config_methodology["parameters"])
        print(map_explainability)
        test, train, model, factsheet = read_solution(solution_set_dropdown)
        final_score, results, properties = get_final_score_unsupervised(model, train, test, weights_config, mappings_config,
                                                           factsheet, solution_set_dropdown)
        trust_score = get_trust_score(final_score, weights_config["pillars"])

        def convert(o):
            if isinstance(o, np.int64): return int(o)

        data = {"final_score": final_score,
                "results": results,
                "trust_score": trust_score,
                "properties": properties}
        return json.dumps(data, default=convert)
    else:
        with open(weight, 'r') as f:
            weights_config = json.loads(f.read())

        mappings_config = dict()

        for pillar, map_conf in zip(pillars,[map_fairness, map_explainability, map_robustness, map_methodology]):
            with open(map_conf, 'r') as f:
                mappings_config[pillar] = json.loads(f.read())

        # mappings_config = dict(fairness=config_fairness["parameters"], explainability=config_explainability["parameters"],
        #                robustness=config_robustness["parameters"], methodology=config_methodology["parameters"])
        print(map_explainability)
        test, train, model, factsheet = read_solution(solution_set_dropdown)
        final_score, results, properties = get_final_score(model, train, test, weights_config, mappings_config, factsheet, solution_set_dropdown)
        trust_score = get_trust_score(final_score, weights_config["pillars"])
        def convert(o):
            if isinstance(o, np.int64): return int(o)
        data = {"final_score": final_score,
                "results": results,
                "trust_score": trust_score,
                "properties": properties}
        return json.dumps(data, default=convert)


@app.callback(Output('general_description-1', 'children'),
              [Input('scenario_dropdown_compare', 'value'),
               Input('solution_set_dropdown-1', 'value')], prevent_initial_call=True)
def show_general_description_1(scenario_id, solution_set_path):
    description = []
    if scenario_id and solution_set_path:
        factsheet = read_factsheet(solution_set_path)

        solution_description_header = html.H5("Model Information")
        solution_description = get_solution_description(factsheet)
        solution_description_table = dash_table.DataTable(
            id='solution_description_table-1',
            columns=[{"name": i, "id": i} for i in solution_description.columns],
            data=solution_description.to_dict('records'),
            style_table={
                # "table-layout": "fixed",
                "width": "100%",
                'overflowX': 'hidden',
                'textAlign': 'left'
            },
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                # 'lineHeight': '15px'
            },
            style_header={
                'backgroundColor': SECONDARY_COLOR,
                # "display": "none",
                # "visibility": "hidden"
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

@app.callback(Output('general_description-2', 'children'),
              [Input('scenario_dropdown_compare', 'value'),
               Input('solution_set_dropdown-2', 'value')], prevent_initial_call=True)
def show_general_description_2(scenario_id, solution_set_path):
    description = []
    if scenario_id and solution_set_path:
        factsheet = read_factsheet(solution_set_path)

        solution_description_header = html.H5("Model Information")
        solution_description = get_solution_description(factsheet)
        solution_description_table = dash_table.DataTable(
            id='solution_description_table-2',
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

@app.callback(Output('performance_metrics_section-1', 'children'),
          Input('solution_set_dropdown-1', 'value'),
              State('toggle_supervised_unsupervised_compare', 'on'), prevent_initial_call=True)
def show_performance_metrics_1(solution_set_path, unsupervised):
    if unsupervised:
        return []
    if not solution_set_path:
        return []
    else:
        test_data, training_data, model, factsheet = read_solution(solution_set_path)
        target_column = factsheet.get("general", {}).get("target_column", "")

        performance_metrics = get_performance_metrics(model, test_data, target_column)
        performance_metrics_table = dash_table.DataTable(
                                id='performance_metrics_table-1',
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
        return html.Div([html.H5("Performance Metrics"), performance_metrics_table], className="mt-4 mb-4", style={"border": "4px solid {}".format(TRUST_COLOR)})

@app.callback(Output('performance_metrics_section-2', 'children'),
          Input('solution_set_dropdown-2', 'value'),
              State('toggle_supervised_unsupervised_compare', 'on'), prevent_initial_call=True)
def show_performance_metrics_2(solution_set_path, unsupervised):
    if unsupervised:
        return []
    if not solution_set_path:
        return []
    else:
        test_data, training_data, model, factsheet = read_solution(solution_set_path)
        target_column = factsheet.get("general", {}).get("target_column", "")

        performance_metrics =  get_performance_metrics(model, test_data, target_column)
        performance_metrics_table = dash_table.DataTable(
                                id='performance_metrics_table-2',
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
        return html.Div([html.H5("Performance Metrics"), performance_metrics_table], className="mt-4 mb-4", style={"border": "4px solid {}".format(TRUST_COLOR)})

@app.callback(Output('properties_section-1', 'children'),
              Input('solution_set_dropdown-1', 'value'),
              State('toggle_supervised_unsupervised_compare', 'on'), prevent_initial_call=True)
def show_properties_1(solution_set_path, unsupervised):
    if not solution_set_path:
        return []
    else:
        test_data, training_data, model, factsheet = read_solution(solution_set_path)
        if not unsupervised:
            properties = get_properties_section(training_data, test_data, factsheet)
        else:
            properties = get_properties_section_unsupervised(training_data, test_data, factsheet)

        properties_table = dash_table.DataTable(
            id='properties_table-1',
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

@app.callback(Output('properties_section-2', 'children'),
              Input('solution_set_dropdown-2', 'value'),
              State('toggle_supervised_unsupervised_compare', 'on'), prevent_initial_call=True)
def show_properties_2(solution_set_path, unsupervised):
    if not solution_set_path:
        return []
    else:
        test_data, training_data, model, factsheet = read_solution(solution_set_path)
        if not unsupervised:
            properties = get_properties_section(training_data, test_data, factsheet)
        else:
            properties = get_properties_section_unsupervised(training_data, test_data, factsheet)

        properties_table = dash_table.DataTable(
            id='properties_table-2',
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

@app.callback(
    [Output('bar-1', 'figure'),
     Output('spider-1', 'figure'),
     Output('fairness_bar-1', 'figure'),
     Output('explainability_bar-1', 'figure'),
     Output('robustness_bar-1', 'figure'),
     Output('methodology_bar-1', 'figure'),
     Output('fairness_spider-1', 'figure'),
     Output('explainability_spider-1', 'figure'),
     Output('robustness_spider-1', 'figure'),
     Output('methodology_spider-1', 'figure'),
     Output('trust_star_rating-1', 'children'),
     Output('fairness_star_rating-1', 'children'),
     Output('explainability_star_rating-1', 'children'),
     Output('robustness_star_rating-1', 'children'),
     Output('methodology_star_rating-1', 'children')],
    Input('result-1', 'data'))
def update_figure_1(data):
    if data is None:
        return [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    result = json.loads(data)
    final_score, results = result["final_score"], result["results"]
    trust_score = result["trust_score"]
    pillars = list(map(lambda x: x.upper(),list(final_score.keys())))
    values = list(final_score.values())

    colors = [FAIRNESS_COLOR, EXPLAINABILITY_COLOR, ROBUSTNESS_COLOR, METHODOLOGY_COLOR]

    # barchart
    chart_list = []
    bar_chart = go.Figure(data=[go.Bar(
        x=pillars,
        y=values,
        marker_color=colors
    )])
    bar_chart.update_layout(title_text='<b style="font-size: 48px;">{}/5</b>'.format(trust_score), title_x=0.5)
    bar_chart.update_yaxes(range=[0, 5], fixedrange=True)
    chart_list.append(bar_chart)

    # spider
    spider_plt = px.line_polar(r=values, theta=pillars, line_close=True,
                               title='<b style="font-size:42px;">{}/5</b>'.format(trust_score))
    spider_plt.update_layout(title_x=0.5)
    spider_plt.update_traces(fill='toself', fillcolor=TRUST_COLOR, marker_color=TRUST_COLOR, marker_line_width=1.5,
                             opacity=0.6)
    chart_list.append(spider_plt)

    # barcharts
    for n, (pillar, sub_scores) in enumerate(results.items()):
        title = "<b style='font-size:32px;''>{}/5</b>".format(final_score[pillar])
        categories = list(map(lambda x: x.replace("_", ' ').title(), sub_scores.keys()))
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
        bar_chart_pillar = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colors[n])])
        bar_chart_pillar.update_layout(title_text=title, title_x=0.5, xaxis_tickangle=XAXIS_TICKANGLE)
        #bar_chart_pillar.update_layout(title_text='', title_x=0.5, xaxis_tickangle=XAXIS_TICKANGLE, paper_bgcolor='#FFFFFF', plot_bgcolor=SECONDARY_COLOR)
        bar_chart_pillar.update_yaxes(range=[0, 5], fixedrange=True)
        chart_list.append(bar_chart_pillar)

    # spider charts
    for n, (pillar, sub_scores) in enumerate(results.items()):
        title = "<b style='font-size:32px;''>{}/5</b>".format(final_score[pillar])
        categories = list(map(lambda x: x.replace("_", ' ').title(), sub_scores.keys()))
        val = list(map(float, sub_scores.values()))
        if np.isnan(values).any():
            nonNanCategories = list()
            nonNanValues = list()
            for c, v in zip(categories, values):
                if not np.isnan(v):
                    nonNanCategories.append(c)
                    nonNanValues.append(v)
            categories = nonNanCategories
            val = nonNanValues
        spider_plt_pillar = px.line_polar(r=val, theta=categories, line_close=True, title=title)
        spider_plt_pillar.update_traces(fill='toself', fillcolor=colors[n], marker_color=colors[n],
                                        marker_line_width=1.5, opacity=0.6)
        spider_plt_pillar.update_layout(title_x=0.5)
        chart_list.append(spider_plt_pillar)

    star_ratings = []
    star_ratings.append(show_star_rating(trust_score))
    star_ratings.append(show_star_rating(final_score["fairness"]))
    star_ratings.append(show_star_rating(final_score["explainability"]))
    star_ratings.append(show_star_rating(final_score["robustness"]))
    star_ratings.append(show_star_rating(final_score["methodology"]))
    return chart_list + star_ratings

@app.callback(
    [Output('bar-2', 'figure'),
     Output('spider-2', 'figure'),
     Output('fairness_bar-2', 'figure'),
     Output('explainability_bar-2', 'figure'),
     Output('robustness_bar-2', 'figure'),
     Output('methodology_bar-2', 'figure'),
     Output('fairness_spider-2', 'figure'),
     Output('explainability_spider-2', 'figure'),
     Output('robustness_spider-2', 'figure'),
     Output('methodology_spider-2', 'figure'),
     Output('trust_star_rating-2', 'children'),
     Output('fairness_star_rating-2', 'children'),
     Output('explainability_star_rating-2', 'children'),
     Output('robustness_star_rating-2', 'children'),
     Output('methodology_star_rating-2', 'children')],
    Input('result-2', 'data'))
def update_figure_2(data):
    if data is None:
        return [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    result = json.loads(data)
    final_score, results = result["final_score"], result["results"]
    trust_score = result["trust_score"]
    pillars = list(map(lambda x: x.upper(),list(final_score.keys())))
    values = list(final_score.values())

    colors = [FAIRNESS_COLOR, EXPLAINABILITY_COLOR, ROBUSTNESS_COLOR, METHODOLOGY_COLOR]

    # barchart
    chart_list = []
    bar_chart = go.Figure(data=[go.Bar(
        x=pillars,
        y=values,
        marker_color=colors
    )])
    bar_chart.update_layout(title_text='<b style="font-size: 48px;">{}/5</b>'.format(trust_score), title_x=0.5)
    bar_chart.update_yaxes(range=[0, 5], fixedrange=True)
    chart_list.append(bar_chart)

    # spider
    spider_plt = px.line_polar(r=values, theta=pillars, line_close=True,
                               title='<b style="font-size:42px;">{}/5</b>'.format(trust_score))
    spider_plt.update_layout(title_x=0.5)
    spider_plt.update_traces(fill='toself', fillcolor=TRUST_COLOR, marker_color=TRUST_COLOR, marker_line_width=1.5,
                             opacity=0.6)
    chart_list.append(spider_plt)

    # barcharts
    for n, (pillar, sub_scores) in enumerate(results.items()):
        title = "<b style='font-size:32px;''>{}/5</b>".format(final_score[pillar])
        categories = list(map(lambda x: x.replace("_", ' ').title(), sub_scores.keys()))
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
        bar_chart_pillar = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colors[n])])
        bar_chart_pillar.update_layout(title_text=title, title_x=0.5, xaxis_tickangle=XAXIS_TICKANGLE)
        bar_chart_pillar.update_yaxes(range=[0, 5], fixedrange=True)
        chart_list.append(bar_chart_pillar)

    # spider charts
    for n, (pillar, sub_scores) in enumerate(results.items()):
        title = "<b style='font-size:32px;''>{}/5</b>".format(final_score[pillar])
        categories = list(map(lambda x: x.replace("_", ' ').title(), sub_scores.keys()))
        val = list(map(float, sub_scores.values()))
        if np.isnan(values).any():
            nonNanCategories = list()
            nonNanValues = list()
            for c, v in zip(categories, values):
                if not np.isnan(v):
                    nonNanCategories.append(c)
                    nonNanValues.append(v)
            categories = nonNanCategories
            val = nonNanValues
        spider_plt_pillar = px.line_polar(r=val, theta=categories, line_close=True, title=title)
        spider_plt_pillar.update_traces(fill='toself', fillcolor=colors[n], marker_color=colors[n],
                                        marker_line_width=1.5, opacity=0.6)
        spider_plt_pillar.update_layout(title_x=0.5)
        chart_list.append(spider_plt_pillar)

    star_ratings = []
    star_ratings.append(show_star_rating(trust_score))
    star_ratings.append(show_star_rating(final_score["fairness"]))
    star_ratings.append(show_star_rating(final_score["explainability"]))
    star_ratings.append(show_star_rating(final_score["robustness"]))
    star_ratings.append(show_star_rating(final_score["methodology"]))
    return chart_list + star_ratings

@app.callback(
        [Output("config-dropdown-compare", "options")] + list(map(lambda pillar: Output('{}-dropdown-compare'.format(pillar), "options") , pillars)),
        Input("toggle_config_compare","on"))
def update_options(trig):
    output = []
    output.append(list(map(lambda name:{'label': name[:-5], 'value': 'configs/supervised/weights/{}'.format(name)} ,os.listdir("configs/supervised/weights"))))
    output = output + list(map(lambda pillar: list(map(lambda name:{'label': name[:-5], 'value': "configs/supervised/mappings/{}/{}".format(pillar,name)} ,
                                                       os.listdir("configs/supervised/mappings/{}".format(pillar)))), pillars))
    return output