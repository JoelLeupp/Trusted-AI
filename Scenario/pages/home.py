import dash_html_components as html
import dash_bootstrap_components as dbc

# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Quantifying the Trustworthiness Level of Artificial Intelligence Models and Decisions", className="text-center")
                    , className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. '
                                     )
                    , className="mb-4")
            ]),
        dbc.Row([
         dbc.Col(dbc.Card(children=[html.H3(children='Try the prototype',
                                               className="text-center"),
                                       dbc.Button("Prototype",
                                                  href="/upload_train_data",
                                                  color="primary",
                                                  className="mt-3"),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=6, className="mb-4"),
        dbc.Col(dbc.Card(children=[html.H3(children='Access to the github repo',
                                               className="text-center"),
                                       dbc.Button("Github",
                                                  href="https://github.com/JoelLeupp/Trusted-AI",
                                                  color="primary",
                                                  className="mt-3"),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=6, className="mb-4")],
         className="mb-5"),
    ])

])
