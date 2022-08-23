# === IMPORTS ===
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import json
import os
import shutil
from config import *
from helpers import *
from app import server
from app import app
# === FUNCTIONS ===

def scenario_dropdown_options():
    scenario_ids = get_scenario_ids()
    options = []
    for scenario_id in scenario_ids:
        scenario_name = id_to_name(scenario_id)
        options.append({"label": scenario_name, "value": scenario_id})
    return options

def load_scenario(scenario_id, unsupervised = False):
    """Example Google style docstrings.

    This module demonstrates documentation as specified by the `Google Python
    Style Guide`_. Docstrings may extend over multiple lines. Sections are created
    with a section header and a colon followed by a block of indented text.

    Example:
        Examples can be given using either the ``Example`` or ``Examples``
        sections. Sections support any reStructuredText formatting, including
        literal blocks::

            $ python example_google.py

    Section breaks are created by resuming unindented text. Section breaks
    are also implicitly created anytime a new section starts.

    Attributes:
        module_level_variable1 (int): Module level variables may be documented in
            either the ``Attributes`` section of the module docstring, or in an
            inline docstring immediately following the variable.

            Either form is acceptable, but the two should not be mixed. Choose
            one convention to document module level variables and be consistent
            with it.

    Todo:
        * For module TODOs
        * You have to also use ``sphinx.ext.todo`` extension

    .. _Google Python Style Guide:
       http://google.github.io/styleguide/pyguide.html

    """
    scenario_path = get_scenario_path(scenario_id, unsupervised)
    scenario_factsheet = read_scenario_factsheet(scenario_id)
    scenario_description = scenario_factsheet.get("description", "")
    scenario_link = scenario_factsheet.get("link", "")
    #scenario_link = load_scenario_link(scenario_path)
    #scenario_description = load_scenario_description(scenario_path)

    all_solutions = []

    if os.path.isdir(os.path.join(scenario_path, SOLUTIONS_FOLDER)):
        all_solutions = all_solutions + [f.name for f in os.scandir(os.path.join(scenario_path, SOLUTIONS_FOLDER)) if f.is_dir() and not f.name.startswith('.')]

    return scenario_link, scenario_description, all_solutions
  

def display_scenario(scenario_id, scenario_name, scenario_link, scenario_description, scenario_solutions):
    """This function open and closes the dialog window
    where the user can create new scenarios

    Args:
        n1: number of clicks on the open button.
        n2: number of clicks on the submit button.
        is_open: current state of the modal.

    Returns:
        Returns false if the dialog was previously open and
        returns true if the dialog was previously closed.

    """
    sections = [
                dcc.ConfirmDialog(
                    id='delete_{}_confirm'.format(scenario_id),
                    message='Are you sure that you want to delete the "{}" scenario?'.format(scenario_name),
                ),
                dbc.Button(
                    html.I(className="fas fa-backspace"),
                    id="delete_{}_button".format(scenario_id), 
                    n_clicks=0,
                    style={"float": "right", "backgroundColor": SECONDARY_COLOR}
                ),
        html.H3("▶ {}".format(scenario_name), style={"text-transform": "none"}),
               html.Div(scenario_description, className="mt-4 mb-1", style={"font-style": "italic"}),
               html.A("Link To Dataset", href=scenario_link, className="mt-1 mb-4", style={"display": "block", "font-style": "italic", "color": PRIMARY_COLOR}),
               html.H4("• Solutions", style={"text-transform": "none"})]
    for i in range(len(scenario_solutions)):
        sections.append(html.H6("-" + scenario_solutions[i], style={"text-transform": "none"}))
    sections.append(html.Hr())
    return html.Div(sections, id="{}_scenario".format(scenario_id))

def display_scenarios(unsupervised = False):
    """This function open and closes the dialog window
        where the user can create new scenarios

    Args:
        n1: number of clicks on the open button.
        n2: number of clicks on the submit button.
        is_open: current state of the modal.

    Returns:
        Returns false if the dialog was previously open and
        returns true if the dialog was previously closed.

    """
    scenario_ids = get_scenario_ids(unsupervised)
    sections = []
    for scenario_id in scenario_ids:
        scenario_name = id_to_name(scenario_id)
        scenario_link, scenario_description, scenario_solutions = load_scenario(scenario_id, unsupervised)
        sections.append(display_scenario(scenario_id, scenario_name, scenario_link, scenario_description, scenario_solutions))
    return sections  
  
    
    
#=== CALLBACKS ===
@app.callback(
    Output("create_scenario_dialog", "is_open"),
    [Input("open_create_scenario_dialog", "n_clicks"), Input("submit_create_scenario_dialog", "n_clicks")],
    [State("create_scenario_dialog", "is_open")],
)
def toggle_create_scenario_modal(n1, n2, is_open):
    """This function open and closes the dialog window
        where the user can create new scenarios

    Args:
        n1: number of clicks on the open button.
        n2: number of clicks on the submit button.
        is_open: current state of the modal.

    Returns:
        Returns false if the dialog was previously open and
        returns true if the dialog was previously closed.

    """
    if n1 or n2:
        return not is_open
    return is_open

for scenario_id in get_scenario_ids():
    #@app.callback(
    #    Output("{}_scenario".format(scenario_id), "style"),
    #    Input("delete_{}_button".format(scenario_id), "n_clicks"),
    #    prevent_initial_call=True
    #)
    #def toggle_detail_section(n_clicks):
    #    print("DELETE {} SCENARIO".format(scenario_id))
    #    if n_clicks:
    #        return {'display': 'None'}
        
    @app.callback(Output('delete_{}_confirm'.format(scenario_id), 'displayed'),
              Input('delete_{}_button'.format(scenario_id), 'n_clicks'), prevent_initial_call=True)
    def display_delete_scenario_confirm(n_clicks):
        if n_clicks:
            return True
        else:
            return False
        
    @app.callback(
        Output("{}_scenario".format(scenario_id), "style"),
        Input("delete_{}_button".format(scenario_id), "n_clicks"),
        prevent_initial_call=True
    )
    def delete_scenario(submit_n_clicks):
        if submit_n_clicks:
            print("Trying to delete the {} scenario".format(scenario_id))
            scenario_path = get_scenario_path(scenario_id)
            try:
                shutil.rmtree(scenario_path, ignore_errors=False)
                return {"display": "none"}
            except Exception as e:
                print(e)
                raise
        else:
            return {"display": "block"}

@app.callback(
    [Output("scenario_display", "children"),
     Output("scenario_name", "value"), 
     Output("scenario_link", "value"), 
     Output("scenario_description", "value")],
    [Input('submit_create_scenario_dialog', 'n_clicks')],
    [State('scenario_display', 'children'),
     State('scenario_name', 'value'), 
     State('scenario_link', 'value'), 
     State('scenario_description', 'value')], prevent_initial_call=True)
def create_scenario(n_clicks, scenario_display, scenario_name, scenario_link, scenario_description):
    if scenario_name:
        # Create folder to contain all solutions
        scenario_id = name_to_id(scenario_name)
        res = os.makedirs(os.path.join(SCENARIOS_FOLDER_PATH, scenario_id, "solutions"))
        f = open(os.path.join(SCENARIOS_FOLDER_PATH, scenario_id, SCENARIO_DESCRIPTION_FILE), "w")
        f.write(scenario_description)
        f.close()
        f = open(os.path.join(SCENARIOS_FOLDER_PATH, scenario_id, SCENARIO_LINK_FILE), "w")
        f.write(scenario_link)
        f.close()
        scenario_display = scenario_display + [display_scenario(scenario_id, scenario_name, scenario_link, scenario_description, [])]
        
    return scenario_display, "", "", ""

# === LAYOUT ===
create_scenario_dialog = html.Div(
    [
        dbc.Button(
            html.I(className="fas fa-plus-circle"),
            id="open_create_scenario_dialog", 
            n_clicks=0,
            style={"float": "right"}
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Create a scenario"),
                dbc.ModalBody(
                dbc.Form([
    dbc.FormGroup([
        dbc.FormText(
            "A scenario acts as a container for multiple different solutions.",
            color="secondary",
        ),
        dbc.Label("Name", html_for="scenario_name"),
        dbc.Input(type="text", id="scenario_name", placeholder="", debounce=True),
        dbc.Label("Link", html_for="scenario_link"),
        dbc.Input(type="url", id="scenario_link", placeholder="", debounce=True),
        dbc.Label("Description", html_for="scenario_name"),
        dcc.Textarea(
            id='scenario_description',
            value='',
            style={'width': '100%', 'height': 100},
        ),
        dbc.Button(
            "Create", id="submit_create_scenario_dialog", className="ml-auto", n_clicks=0, style={"float": "right"}
        )
    ])
])),
            ],
            id="create_scenario_dialog",
            is_open=False,
        ),
    ]
)

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                create_scenario_dialog,
                html.H1("Scenarios Supervised", className="text-center", style={"text-transform": "uppercase"}),
            ], width=12),
            dbc.Col(
                html.Div(
                    children=[ 
                        html.Div(children=display_scenarios(), id="scenario_display", style={"backgroundColor": SECONDARY_COLOR}),              
                   ]
                ),
                className="mb-5 mt-5",
                width=12, 
                style={
                    "border": "1px solid #d8d8d8",
                    "borderRadius": "6px",
                    "backgroundColor": SECONDARY_COLOR
                }   
            ),
        ]),
        dbc.Row([
            dbc.Col([
                create_scenario_dialog,
                html.H1("Scenarios Unsupervised", className="text-center", style={"text-transform": "uppercase"}),
            ], width=12),
            dbc.Col(
                html.Div(
                    children=[
                        html.Div(children=display_scenarios(unsupervised=True), id="scenario_display_unsupervised", style={"backgroundColor": SECONDARY_COLOR}),
                   ]
                ),
                className="mb-5 mt-5",
                width=12,
                style={
                    "border": "1px solid #d8d8d8",
                    "borderRadius": "6px",
                    "backgroundColor": SECONDARY_COLOR
                }
            ),
        ])
    ])
])