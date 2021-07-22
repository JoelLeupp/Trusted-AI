import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import base64
import os
import pandas as pd
import io
import datetime
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_table
import json
import pickle



# must add this line in order for the app to be deployed successfully on Heroku
from app import server
from app import app
# import all pages in the app
from apps import home, upload_train_data, visualisation, test



navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(dbc.NavbarBrand("Trusted AI Algorithm", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/home",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
        ]
    ),
    color="dark",
    dark=True,
    className="mb-4",
)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
        elif 'pkl' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_pickle(io.BytesIO(decoded))
        df = df.describe().reset_index()
        
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    return html.Div([
        html.H5("Statistics regarding "+filename, className="text-center", style={"color":"DarkBlue"}),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'scroll'},
        ),
        html.Hr(),
    ])

def save_file(name, content):
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(os.getcwd(), "uploaded_files", name), "wb") as fp:
        fp.write(base64.decodebytes(data))

def save_model(name, content):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_pickle(io.BytesIO(decoded))
    pickle.dump(df, open(os.path.join(os.getcwd(), "uploaded_files", name), 'wb'))

def save_factsheet(regularization):
    factsheet = { 'regularization': regularization}
    with open(os.path.join(os.getcwd(), "uploaded_files","factsheet.json"), "w",  encoding="utf8") as fp:
        json.dump(factsheet, fp, indent=4)
    
@app.callback(Output('hidden-div', 'children'),
              [Input('trustscore-button', 'n_clicks'),
               Input('upload-model', 'contents'),
               Input('upload-train-data', 'contents'),
               Input('upload-test-data', 'contents'),
               # State('regularization', 'value'),
               State('upload-train-data', 'contents'),
               State('upload-test-data', 'contents'),
               State('upload-model', 'contents')
               ])
def calculate_trust_score(n_clicks, modeltrigger, traintrigger, testtrigger, train, test, model):
    trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if trigger == "trustscore-button":
        if n_clicks is None:
            return ""
        else:
            # save_factsheet(regularization)
            if train is None:
                return html.H4("Please upload the train data.", style={"color":"Red"},  className="text-center")
            elif test is None:
                return html.H4("Please upload the test data.", style={"color":"Red"},  className="text-center")
            elif model is None:
                return html.H4("Please upload the model.", style={"color":"Red"},  className="text-center")
            return dcc.Location(pathname="/visualisation", id="someid_doesnt_matter")
    else:
        return ""

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/upload_train_data':
        return upload_train_data.layout
    elif pathname == '/visualisation':
        return visualisation.layout
    elif pathname == '/test':
        return test.layout
    else:
        return home.layout
    
@app.callback(
   Output(component_id='panel', component_property='style'),
   [Input(component_id="toggle-hide", component_property='on')])

def show_hide_element(visibility_state):
    if visibility_state == True:
        return {'display': 'block'}
    if visibility_state == False:
        return {'display': 'none'}

@app.callback([Output('model-uploaded-div', 'children'),
               Output('upload-model', 'children')],
              [Input('upload-model', 'contents'),
              State('upload-model', 'filename')])
def upload_model_callback(content, name):
    if content is not None:
        save_model(name, content)
        return [html.H4("Model is uploaded.", style={"color":"Green"}),  html.Div(['Drag and Drop or Select a Different File (Overwrites the Previous One)'])]
    return [None,  html.Div(['Drag and Drop or Select Files'])]
    
@app.callback([Output('output-train-data-upload', 'children'),
              Output('upload-train-data', 'children')],
              [Input('upload-train-data', 'contents'),
              State('upload-train-data', 'filename')])
def upload_train_callback(content, name):
    if content is not None:
        children = [parse_contents(content, name)]
        save_file(name, content)
        return [children, html.Div(['Drag and Drop or Select a Different File (Overwrites the Previous One)'])]
    return [None, html.Div(['Drag and Drop or Select Files'])]

@app.callback([Output('output-test-data-upload', 'children'),
              Output('upload-test-data', 'children')],
              [Input('upload-test-data', 'contents'),
              State('upload-test-data', 'filename')])
def upload_test_callback(content, name):
    if content is not None:
        children = [parse_contents(content, name)]
        save_file(name, content)
        return [children, html.Div(['Drag and Drop or Select a Different File (Overwrites the Previous One)'])]
    return [None, html.Div(['Drag and Drop or Select Files'])]

@app.callback([Output('spider', 'style'),
              Output('spider_pillars', 'style'),
              Output('bar', 'style'),
              Output('bar_pillars', 'style')],
              [Input('plot_type', 'value')])
def show_the_graphs(value):
    if value == "spider":
        return [{'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}]
    else:
        return [{'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}]


if __name__ == '__main__':
    app.run_server(debug=False)


#debug
# with open("Validation/case1/X_test.pkl", "rb") as file:
#     encoded_string = base64.b64encode(file.read()).decode()
    
# encoded_string.encode("utf8")

# encoded_string.encode("utf8").split(b";base64,")[1]
# encoded_string.encode("utf8").split(b";base64,")

# data = encoded_string.encode("utf8")
# with open(os.path.join(os.getcwd(), "uploaded_files", "test.pkl"), "wb") as fp:
#       fp.write(base64.decodebytes(data))
