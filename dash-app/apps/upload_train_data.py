import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

layout = html.Div([
    dbc.Col(html.H1("Please upload the train data (csv and pickle files are accepted).", className="text-center")
                    , className="mb-4"),
    html.Div([dbc.Button("Next",id='hidden-button-train', href="/upload_test_data", className="mt-3", style ={'display':'none'})],
             style ={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-train-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-train-data-upload'),
])








if __name__ == '__main__':
    app.run_server(debug=True)
