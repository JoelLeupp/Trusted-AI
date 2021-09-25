import os
import dash_daq as daq
from config import SCENARIOS_FOLDER_PATH
import glob
import pickle
import pandas as pd
import json
import base64
import io
import numpy as np
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from config import *
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import random
import pickle
import matplotlib.pyplot as plt
from math import pi
import sklearn.metrics as metrics
import collections
import reportlab
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.rl_config import defaultPageSize
from reportlab.lib.units import inch
from reportlab.graphics.shapes import *
from reportlab.lib.colors import *
from base64 import b64encode
from textwrap import wrap
import timeit

PAGE_HEIGHT=defaultPageSize[1]; PAGE_WIDTH=defaultPageSize[0]

result = collections.namedtuple('result', 'score properties')

def draw_bar_plot(categories, values, ax, color='lightblue', title='Trusting AI Final Score',size=12):
    
    # drop top and right spine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #plt.grid(True,axis='y',zorder=0)
    
    # create barplot
    x_pos = np.arange(len(categories))
    plt.bar(x_pos, values,zorder=4,color=color)

    for i, v in enumerate(values):
        plt.text(i , v+0.1 , str(v),color='dimgray', fontweight='bold',ha='center')

    # Create names on the x-axis
    plt.xticks(x_pos, categories,size=size,wrap=True)

    plt.yticks([1,2,3,4, 5], ["1","2","3","4","5"], size=12)
    plt.ylim(0,5)
    
    if len(categories) > 4:
        plt.xticks(rotation=15)
    
    if isinstance(color, list):
        plt.title(title, size=11, y=1.1)
    else:
        # plt.title(title, size=11, y=1.1,
        #      bbox=dict(facecolor=color, edgecolor=color, pad=2.0))
        plt.title(title, size=11, y=1.1)
    return plt
   

def get_performance_metrics(model, test_data, target_column):

    if target_column:
        X_test = test_data.drop(target_column, axis=1)
        y_test = test_data[target_column]
    else:
        X_test = test_data.iloc[:,:DEFAULT_TARGET_COLUMN_INDEX]
        y_test = test_data.reset_index(drop=True).iloc[:,DEFAULT_TARGET_COLUMN_INDEX:]

    y_true = y_test.values.flatten()
    if (isinstance(model, tf.keras.Sequential)):
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test).flatten()
    #y_pred_proba = model.predict_proba(X_test)
    print("y_true.shape: {}".format(y_true.shape))
    print("y_pred.shape: {}".format(y_pred.shape))
    #print("y_pred_proba.shape: {}".format(y_pred_proba.shape))
    labels = np.unique(np.array([y_pred,y_true]).flatten())

    performance_metrics = pd.DataFrame({
        "accuracy" :  [metrics.accuracy_score(y_true, y_pred)],
        "global recall" :  [metrics.recall_score(y_true, y_pred, labels=labels, average="micro")],
        "class weighted recall" : [metrics.recall_score(y_true, y_pred,average="weighted")],
        "global precision" : [metrics.precision_score(y_true, y_pred, labels=labels, average="micro")],
        "class weighted precision" : [metrics.precision_score(y_true, y_pred,average="weighted")],
        "global f1 score" :  [metrics.f1_score(y_true, y_pred,average="micro")],
        "class weighted f1 score" :  [metrics.f1_score(y_true, y_pred,average="weighted")],
        #"cross-entropy loss" : [metrics.log_loss(y_true, y_pred_proba)],
        #"ROC AUC" : [metrics.roc_auc_score(y_true, y_pred_proba,average="weighted", multi_class='ovr')]
    }).round(decimals=2)
    
    #performance_metrics = pd.Data
    #print("------------------ {}".format(performance_metrics.transpose().columns))
    performance_metrics = performance_metrics.transpose()
    performance_metrics = performance_metrics.reset_index()
    performance_metrics['index'] = performance_metrics['index'].str.title()
    performance_metrics.rename(columns={"index":"key", 0:"value"}, inplace=True)
    return performance_metrics


def get_scenario_description(scenario_id):
    scenario_factsheet = read_scenario_factsheet(scenario_id)
    data = []
    data.insert(0, {'key': 'name', 'value': id_to_name(scenario_id)})
    scenario_description = pd.DataFrame.from_dict(scenario_factsheet, orient="index", columns=["value"])
    #scenario_description.index.set_names(['Scenario'])
    scenario_description = scenario_description.reset_index()
    print(scenario_description.columns)
    #scenario_description.index.rename('Scenario', inplace=True)
    print(scenario_description.columns)
    scenario_description.rename(columns={'index': 'key'}, inplace=True)
    scenario_description = pd.concat([pd.DataFrame(data), scenario_description], ignore_index=True)
    scenario_description['key'] = scenario_description['key'].str.capitalize()
    return scenario_description

def get_properties_section(train_data, test_data, factsheet):
    if "properties" in factsheet:
        factsheet = factsheet["properties"]

        properties = pd.DataFrame({
            "Model Type": [factsheet["explainability"]["algorithm_class"]["clf_name"][1]],
            "Train Test Split": [factsheet["methodology"]["train_test_split"]["train_test_split"][1]],
            "Train / Test Data Size": str(train_data.shape[0])+ " samples / "+ str(test_data.shape[0])+ " samples",
            "Regularization Technique": [factsheet["methodology"]["regularization"]["regularization_technique"][1]],
            "Normalization Technique": [factsheet["methodology"]["normalization"]["normalization"][1]],
            "Number of Features": [factsheet["explainability"]["model_size"]["n_features"][1]],
        })
        properties = properties.transpose()
        properties = properties.reset_index()
        properties['index'] = properties['index'].str.title()
        properties.rename(columns={"index": "key", 0: "value"}, inplace=True)
        return properties
    return None


def get_solution_description(factsheet):
    description = {}
    for e in GENERAL_INPUTS:
        if e == "target_column":
            continue
        
        description[id_to_name(e)] = factsheet.get("general", {}).get(e, " ")
    #if "general" in factsheet:
        #if "model_name" in factsheet["general"]:
           
            #description["Model Name"]= factsheet["general"]["model_name"]
           
        #if "purpose_description" in factsheet["general"]:
        #    description["Purpose of the Model"] = factsheet["general"]["purpose_description"]
        #if "training_data_description" in factsheet["general"]:
        #    description["Training Data Description"] = factsheet["general"]["training_data_description"]
    description = pd.DataFrame(description, index=[0])
    description = description.transpose()
    description = description.reset_index()
    description['index'] = description['index'].str.title()
    description.rename(columns={"index": "key", 0: "value"}, inplace=True)
    return description

def show_star_rating(rating):
    stars = []
    for i in range(0,5):
        if i+0.99 <= rating:
            stars.append(html.I(className="fas fa-star"))
        elif i+0.49 < rating:
            stars.append(html.I(className="fas fa-star-half-alt"))
        else:
            stars.append(html.I(className="far fa-star"))
    return stars

def id_to_name(identifier):
    """This function converts scenario and solution ids into the matching names

    Args:
        n1: number of clicks on the open button.

    Returns:
        Returns false if the dialog was previously open and
        returns true if the dialog was previously closed.

    """
    return identifier.replace("_", " ").title()

def name_to_id(name):
    """This function converts scenario and solution names into a valid ids

    Args:
        n1: number of clicks on the open button.

    Returns:
        Returns false if the dialog was previously open and

    """
    return name.replace(" ", "_").lower()

# === SCENARIOS ===
def get_scenario_ids():
    scenario_ids = [f.name for f in os.scandir(SCENARIOS_FOLDER_PATH) if f.is_dir() and not f.name.startswith('.')]
    #sort scenario ids in place
    scenario_ids.sort()
    return scenario_ids

def get_scenario_options():
    scenario_ids = get_scenario_ids()
    options = [{"label": id_to_name(scenario_id), "value": scenario_id} for scenario_id in scenario_ids]
    return options

def get_scenario_path(scenario_id):
    return os.path.join(SCENARIOS_FOLDER_PATH, scenario_id)

def get_solution_ids(scenario_id):
    solution_ids = [(f.name, f.path) for f in os.scandir(get_solution_path(scenario_id, "")) if f.is_dir() and not f.name.startswith('.')]
    solution_ids = sorted(solution_ids, key=lambda x: x[0])
    return solution_ids
    
def get_solution_options():
    scenario_ids = get_scenario_ids()
    options = []
    for scenario_id in scenario_ids:
        scenario_name = id_to_name(scenario_id)
        solutions = get_solution_ids(scenario_id)
        for solution_id, solution_path in solutions:
            solution_name = id_to_name(solution_id)
            options.append({"label": scenario_name + " > " + solution_name, "value": solution_path})
    return options

def get_scenario_solutions_options(scenario_id):
    options = []  
    solutions = get_solution_ids(scenario_id)
    for solution_id, solution_path in solutions:
            solution_name = id_to_name(solution_id)
            options.append({"label": solution_name, "value": solution_path})
    return options

def get_solution_path(scenario_id, solution_id):
    return os.path.join(SCENARIOS_FOLDER_PATH, scenario_id, SOLUTIONS_FOLDER, solution_id)

def read_test(solution_set_path):
    #test data
    test_file = glob.glob(os.path.join(solution_set_path, TEST_DATA_FILE_NAME_REGEX))[0]
    ext = os.path.splitext(test_file)[1]
    if ext == ".pkl":
        with open(test_file,'rb') as file:
            test = pickle.load(file)
    elif ext == ".csv":
        test = pd.read_csv(test_file)
    else:
        test = None
    
    return test
        
def read_train(solution_set_path):
    if solution_set_path is None:
        return
    train_file = glob.glob(os.path.join(solution_set_path, TRAINING_DATA_FILE_NAME_REGEX))[0]
    print("--- {}".format(glob.glob(os.path.join(solution_set_path,"train.*"))[0]))
    ext = os.path.splitext(train_file)[1]
    if ext == ".pkl":
        with open(train_file,'rb') as file:
            train = pickle.load(file)
    elif ext == ".csv":
        train = pd.read_csv(train_file)
    else:
        train = None
    
    return train

# Load .joblib or .pickle model
def read_model(solution_set_path):
    model_file = glob.glob(os.path.join(solution_set_path, MODEL_REGEX))[0]
    print("model_file: {}".format(model_file))
    file_extension = os.path.splitext(model_file)[1]
    print("file extension of model to load {0}".format(file_extension))
    pickle_file_extensions = [".sav", ".pkl", ".pickle"]
    if file_extension in pickle_file_extensions:
        with open(model_file,'rb') as file:
            model = pickle.load(file)
        return model
    if file_extension == ".h5":
        tf.compat.v1.disable_eager_execution()
        model = load_model(model_file)
        return model
    if file_extension == ".joblib":
        print("loading joblib model")
        return joblib.load(model_file)

# === FACTSHEET ===
'''
    This function reads the factsheet into a dictionary
'''
def read_factsheet(solution_set_path):
    factsheet_path = os.path.join(solution_set_path, FACTSHEET_NAME)
    if os.path.isfile(factsheet_path):
        with open(factsheet_path,'rb') as f:
            factsheet = json.loads(f.read())
        return factsheet
    else:
        return {}
    
'''
    This function reads the factsheet into a dictionary
'''
def read_scenario_factsheet(scenario_id):
    factsheet_path = os.path.join(SCENARIOS_FOLDER_PATH, scenario_id, FACTSHEET_NAME)
    if os.path.isfile(factsheet_path):
        with open(factsheet_path,'rb') as f:
            factsheet = json.loads(f.read())
        return factsheet
    else:
        return {}
    
'''
    This function reads the factsheet into a dictionary
'''
def write_scenario_factsheet(scenario_id):
    factsheet_path = os.path.join(SCENARIOS_FOLDER_PATH, scenario_id, FACTSHEET_NAME)
    if os.path.isfile(factsheet_path):
        with open(factsheet_path,'rb') as f:
            factsheet = json.loads(f.read())
        return factsheet
    else:
        return {}
    
    
def write_into_factsheet(new_factsheet, solution_set_path):
    factsheet_path = os.path.join(solution_set_path, FACTSHEET_NAME)
    with open(factsheet_path, 'w') as outfile:
        json.dump(new_factsheet, outfile, indent=4)
    return

def title_style(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Bold',16)
    canvas.setFillColor('#000080')
    canvas.drawString(50, 800, "TRUSTED AI")
    canvas.drawCentredString(PAGE_WIDTH/2.0, PAGE_HEIGHT-88, "Report")
    canvas.setFont('Times-Roman',9)
    canvas.restoreState()


def report_section(Story, title , keys, values, sizex, sizey):
    it = iter(zip(keys, values))

    Story.append(Spacer(1, 0.2 * inch))

    p = Paragraph(id_to_name(title))

    Story.append(p)
    Story.append(Spacer(1, 0.1 * inch))
    d = Drawing(PAGE_WIDTH, 1)
    d.add(Line(0, 0, PAGE_WIDTH-130, 0, strokeColor='#000080', strokeWidth=.8))
    Story.append(d)
    Story.append(Spacer(1, 0.1 * inch))

    data = []

    for x in it:
        p1 = Paragraph('{}:'.format(id_to_name(x[0])))
        p2 = Paragraph('{}'.format(x[1]))
        item = next(it, None)
        if item is None:
            p3 = ''
            p4 = ''
        else:
            l = item[0]
            v = item[1]
            p3 = Paragraph('{}:'.format(id_to_name(l)))
            p4 = Paragraph('{}'.format(v))
        m = [p1, p2, p3, p4]
        data.append(m)
    print(data)
    t = Table(data,sizex, sizey, style = [('VALIGN',(0,0),(-1,-1),'MIDDLE')])
    Story.append(t)
    Story.append(Spacer(1, 0.2 * inch))

    return Story

def report_performance_metrics_section(Story, title , keys, values):
    Story.append(Spacer(1, 0.2 * inch))
    p = Paragraph(id_to_name(title))
    Story.append(p)
    Story.append(Spacer(1, 0.1 * inch))
    d = Drawing(PAGE_WIDTH, 1)
    d.add(Line(0, 0, PAGE_WIDTH-130, 0, strokeColor='#000080', strokeWidth=.8))
    Story.append(d)
    Story.append(Spacer(1, 0.1 * inch))

    it = iter(zip(keys, values))
    data = []
    l,v = next(it)
    m = ['{}:'.format(id_to_name(l)), '{}'.format(v), '', '']
    data.append(m)
    for x in it:
        l, v = next(it)
        m = ['{}:'.format(id_to_name(x[0])), '{}'.format(x[1]), '{}:'.format(id_to_name(l)), '{}'.format(v)]
        data.append(m)

    t = Table(data, [2 * inch, 0.7*inch, 2 * inch, 0.7*inch], 4 * [0.3 * inch],
              style=[('SPAN', (1, 0), (-1, 0))])
    Story.append(t)
    Story.append(Spacer(1, 0.2 * inch))
    return Story


def add_matplotlib_to_report(Story, fig, sizex, sizey):
    imgdata = io.BytesIO()
    fig.savefig(imgdata, format='png',dpi=300,bbox_inches='tight')
    imgdata.seek(0)  # rewind the data
    im = reportlab.platypus.Image(imgdata)
    im._restrictSize(sizex, sizey)
    Story.append(im)
    return Story


def save_report_as_pdf(result, model, test_data, target_column, factsheet, charts, configs):
      
    start = timeit.timeit()
    print("creating report")
    weight, map_f, map_e, map_r, map_m = configs
    doc = SimpleDocTemplate("report.pdf")
    Story = [Spacer(1, 0.1 * inch)]

    sizex = [1.3 * inch, 2.1*inch, 1.3 * inch, 2.1*inch]
    sizey = 4 * [0.4 * inch]
    Story = report_section(Story, "Model Information", factsheet["general"].keys(),factsheet["general"].values(), sizex, sizey)
    perf = get_performance_metrics(model, test_data, target_column)
    keys = []
    values = []
    for dic in perf.to_dict('records'):
        keys.append(dic["key"])
        values.append(dic["value"])

    Story = report_performance_metrics_section(Story,  "Performance of the Model", keys, values)


    #overall score chart
    final_score = result["final_score"]
    pillars = list(final_score.keys())
    values = list(final_score.values())
    pillar_colors = ['#06d6a0', '#ffd166', '#ef476f', '#118ab2']

    plt.switch_backend('Agg')
    my_dpi = 96
    fig = plt.figure(figsize=(600 / my_dpi, 400 / my_dpi), dpi=my_dpi)
    ax = plt.subplot(111)
    trust_score = result["trust_score"]
    draw_bar_plot(pillars, values, ax, color=pillar_colors,
                  title="Overall Trust Score {}/5 \n(config: {})".format(trust_score, weight.split("/")[-1][:-5]))
    Story = add_matplotlib_to_report(Story, fig, 7 * inch, 5 * inch)

    results = result["results"]

    plots= []
    for n, (pillar, sub_scores) in enumerate(results.items()):
        my_dpi = 96
        fig = plt.figure(figsize=(600 / my_dpi, 400 / my_dpi), dpi=my_dpi)
        title = "{} \n(config: {})".format(pillar, configs[n + 1].split("/")[-1][:-5])
        categories = list(map(lambda x: x.replace("_", ' '), sub_scores.keys()))
        categories = ['\n'.join(wrap(l, 12, break_long_words=False)) for l in categories]
        values = list(sub_scores.values())
        nan_val = np.isnan(values)
        values = np.array(values)[~nan_val]
        categories = np.array(categories)[~nan_val]
        ax = plt.subplot(2, 2, n + 1)
        draw_bar_plot(categories, values, ax, color=pillar_colors[n], title=id_to_name(title), size=6)
        plots.append(fig)

    Story = add_matplotlib_to_report(Story, plots[0], 7 * inch, 5 * inch)
    fairness_properties = [p for k, p in result["properties"]["fairness"].items()]
    k = []
    v = []
    for l in fairness_properties:
        if l!= {}:
            for i,m in l.items():
                if type(m) == list:
                    k.append(m[0])
                    v.append(m[1])
                else: 
                    k.append(i)
                    v.append(m)

    sizex = 4* [1.6 * inch]
    sizey = math.ceil(len(k)/2) * [0.4 * inch]
    Story = report_section(Story, "Fairness Properties",  k, v, sizex, sizey)

    Story = add_matplotlib_to_report(Story, plots[1], 7 * inch, 5 * inch)
    explainability_properties = [p for k, p in result["properties"]["explainability"].items()] 
    k = []
    v = []
    for l in explainability_properties:
        if l!= {}:
            for i,m in l.items():
             if i !="importance":
                if type(m) == list:
                    k.append(m[0])
                    v.append(m[1])
                else: 
                    k.append(i)
                    v.append(m)

    sizex = [2.3 * inch, 1.4 * inch, 2.3 * inch, 0.8 * inch]
    sizey = math.ceil(len(k)/2) * [0.45 * inch]
    Story = report_section(Story, "Explainability Properties", k, v, sizex, sizey)

    Story = add_matplotlib_to_report(Story, plots[2], 7 * inch, 5 * inch)
    robustness_properties = [p for k, p in result["properties"]["robustness"].items()]
    k = []
    v = []
    for l in robustness_properties:
        if l!= {}:
            for i,m in l.items():
                if type(m) == list:
                    k.append(m[0])
                    v.append(m[1])
                else: 
                    k.append(i)
                    v.append(m)
    sizex = [2.4 * inch, 0.8 * inch, 2.4 * inch, 0.8 * inch]
    sizey = math.ceil(len(k)/2) * [0.4 * inch]
    Story = report_section(Story, "Robustness Properties", k, v, sizex, sizey)

    Story = add_matplotlib_to_report(Story, plots[3], 7 * inch, 5 * inch)
    methodology_properties = [p for k, p in result["properties"]["methodology"].items()]
    k = []
    v = []
    for l in methodology_properties:
        if l!= {}:
            for i,m in l.items():
                if type(m) == list:
                    k.append(m[0])
                    v.append(m[1])
                else: 
                    k.append(i)
                    v.append(m)

    sizex = [2.2 * inch, 1.2 * inch, 2.4 * inch, 1 * inch]
    sizey = math.ceil(len(k)/2) * [0.4 * inch]
    Story = report_section(Story, "Methodology Properties", k, v, sizex, sizey)


    doc.build(Story, onFirstPage=title_style)
    end = timeit.timeit()
    print(end - start)  
    

def read_solution(solution_set_path):
    test = read_test(solution_set_path)
    train = read_train(solution_set_path)
    model = read_model(solution_set_path)
    factsheet = read_factsheet(solution_set_path)
                
    return test, train, model, factsheet
   
def create_info_modal(module_id, name, content, example):
    modal = html.Div(
    [
        dbc.Button(
            html.I(className="fas fa-info-circle"),
            id="{}_info_button".format(module_id), 
            n_clicks=0,
            style={"float": "right", "backgroundColor": SECONDARY_COLOR}
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(name),
                dbc.ModalBody([content, html.Br(), html.Br(), dcc.Markdown(example) ]),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="{}_close".format(module_id), className="ml-auto", n_clicks=0
                    )
                ),
            ],
            id="{}_info_modal".format(module_id),
            is_open=False,
        ),
    ]
)
    return modal

#def load_scenario_description(scenario_path):
#    scenario_description = ""
#    path = os.path.join(scenario_path, SCENARIO_DESCRIPTION_FILE)
#    if os.path.exists(path):
#        file = open(path, mode='r')
#        scenario_description = file.read()
#        file.close()
#    return scenario_description
 
#def load_scenario_link(scenario_path):
#    scenario_link = ""
#    path = os.path.join(scenario_path, SCENARIO_LINK_FILE)
#    if os.path.exists(path):
#        file = open(path, mode='r')
#        scenario_link = file.read()
#        file.close()
#    return scenario_link
    
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
        df = df[:8]
        
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ]), []
    
    table = html.Div([
        html.H5("Preview of "+filename, className="text-center", style={"color":"DarkBlue"}),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'scroll'},
            style_header={"textTransform": "none"},
        ),
        html.Hr(),
    ])
    columns = df.columns.values
    return table, columns

def list_of_metrics(pillar):
    metrics = []
    with open(os.path.join(METRICS_CONFIG_PATH, "config_{}.json".format(pillar))) as file:
        config_file = json.load(file)
        for metric_name in config_file["weights"]:
            metrics.append(metric_name.lower())
    return metrics

def create_metric_details_section(metric_id, i, section_n = 1, is_open=False, score="X"):
    metric_name = metric_id.replace("_", " ")
    return html.Div([

        html.Div([
            html.I(className="fas fa-chevron-down ml-4", id="toggle_{}_details".format(metric_id), style={"float": "right"}),
            html.H4("({}/5)".format(score),id="{}_score".format(metric_id), style={"float": "right"}), 
        html.H4("{2}.{0} {1}".format(i+1, metric_name, section_n)),
        ]),
            dbc.Collapse(
            html.Div([NO_DETAILS]),
            id="{}_details".format(metric_id),
            is_open=is_open,          
        ),
        ], id="{}_section".format(metric_id), className="mb-5 mt-5")

def show_metric_details_section(metric_id, metric_score=None, metric_properties = None, metric_index = 1, section_index = 1):
    metric_name = metric_id.replace("_", " ")
    sections = []
    if not math.isnan(metric_score):
        sections.append(html.I(className="fas fa-chevron-down ml-4", id="toggle_{}_details".format(metric_id), style={"float": "right"}))
        sections.append(html.H4("({}/5)".format(metric_score),id="{}_score".format(metric_id), style={"float": "right"})),
        sections.append(html.H4("{0}.{1} {2}".format(section_index, metric_index, metric_name)))
    else: 
        sections.append(html.H4("- {}".format(metric_name)))
    
    if metric_properties:
        sections.append(dbc.Collapse(show_metric_properties(metric_properties), id="{}_details".format(metric_id)))
        
    return html.Div(sections, id="{}_section".format(metric_id), className="mb-5 mt-5")

def metric_detail_div(properties):
    prop = []
    for k, v in properties.items():
        prop.append(html.Div("{}: {}".format(v[0], v[1])))
    return html.Div(prop)

def show_metric_properties(metric_properties):
    sections = []
    for k, v in metric_properties.items():
        sections.append(html.Div("{}: {}".format(k, v)))
    return html.Div(sections)

def pillar_section(pillar, metrics):
        #configuration_section = []
        #configuration_section.append(html.H3("▶ {} Configuration".format(pillar)))
        #configuration_section.append()
        #metric_detail_sections = [html.Div([], id="{}_configuration".format(pillar))]
        metric_detail_sections = []
        for i in range(len(metrics)):
            metric_id = metrics[i].lower()
            metric_detail_sections.append(create_metric_details_section(metric_id, i))

        return html.Div([
                # html.H2("{}".format(pillar.upper()), className="text-center"),
                dbc.Row(
                    [
                    dbc.Col(html.Div([daq.BooleanSwitch(id='toggle_{}_details'.format(pillar),
                      on=False,
                      label='Show Details',
                      labelPosition="top",
                      color = TRUST_COLOR,
                    )], className="text-center")),
                    dbc.Col(html.Div([daq.BooleanSwitch(id='toggle_{}_mapping'.format(pillar),
                      on=False,
                      label='Show Mappings',
                      labelPosition="top",
                      color = TRUST_COLOR,
                    )], className="text-center"))
                    ]),
                # html.Div([
                #     daq.BooleanSwitch(id='toggle_{}_details'.format(pillar),
                #       on=False,
                #       label='Show Details',
                #       labelPosition="right",
                #       color = TRUST_COLOR,
                #       style={"float": "right","margin-right":"3%"}
                #     ),
                #     daq.BooleanSwitch(id='toggle_{}_mapping'.format(pillar),
                #       on=False,
                #       label='Show Mappings',
                #       labelPosition="right",
                #       color = TRUST_COLOR,
                #       style={"float": "right","margin-right":"3%"}
                #     ),
                #     html.H2("{}".format(pillar.upper()), className="mb-5"),
                #     ], id="{}_section_heading".format(pillar.lower())),
                    dbc.Collapse(html.Div(mapping_panel(pillar)[0]),
                        id="{}_mapping".format(pillar),
                        is_open=False,
                        style={"background-color": "rgba(255,228,181,0.5)",'padding-bottom': 20, 'display': 'none'}
                    ),
                    html.Br(),
                    html.Div([], id="{}_overview".format(pillar)),
                    html.H3("{0}-Score".format(pillar), className="text-center"),
                    html.Div([], id="{}_star_rating".format(pillar), className="star_rating, text-center"),
                    html.B(["X/5"], id="{}_score".format(pillar), className="text-center", style={"display": "block","font-size":"32px"}),
                    dcc.Graph(id='{}_spider'.format(pillar), style={'display': 'none'}),
                    dcc.Graph(id='{}_bar'.format(pillar), style={'display': 'block'}),    
                    html.Div([], id="{}_available_metrics".format(pillar),className = pillar),
                    dbc.Collapse(["{}_configuration".format(pillar)],
                        id="{}_configuration".format(pillar),
                        is_open=False,
                    ),
                    dbc.Collapse(metric_detail_sections,
                        id="{}_details".format(pillar),
                        is_open=False,
                    ),
                    html.Hr(style={"size": "10"}),
                    dbc.Modal(
                    [   
                        dbc.ModalHeader("Save {} Mapping".format(pillar)),
                        dbc.ModalBody([
                                       html.Div([
                                        html.Div(html.Label("Please enter a name:"), style={'width': '40%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
                                        html.Div(dcc.Input(id="mapping-name-{}".format(pillar), type='text', placeholder="Alias for mapping", style={"width":"200px"}), 
                                                 style={'width': '40%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
                                        ])
                                      ]),
                        dbc.ModalFooter(
                                    dbc.Button(
                                        "Save", id="save-{}-mapping".format(pillar), className="ml-auto", n_clicks=0, 
                                        style={'background-color': 'green','font-weight': 'bold'}
                                    )
                                ),
                    ],
                    id="modal-{}-mapping".format(pillar),
                    is_open=False,
                    backdrop=True
                    ),
                    dbc.Modal(
                    [
                        dbc.ModalHeader("Success"),
                        dbc.ModalBody([dbc.Alert(id="alert-success",children ="You successfully saved the Mapping", color="success"),
                                      ]),
                    ],
                    id="modal-saved-{}".format(pillar),
                    is_open=False,
                    backdrop=True
                    )                   

                ], id="{}_section".format(pillar), style={"display": "None"}, hidden = True)
    
def mapping_panel(pillar):
    
    with open('configs/mappings/{}/default.json'.format(pillar), 'r') as f:
                mapping  = json.loads(f.read())
    
    map_panel = []
    input_ids = []
    
    #weight panel
    map_panel.append(html.H4("Mappings",style={'text-align':'center'}))
    # map_panel.append(dcc.Store(id='{}-mapping'.format(pillar)))
    for metric, param in mapping.items():
        map_panel.append(html.H5(metric.replace("_",' '),style={'text-align':'center'}))
        for p, v in param.items():
            input_id = "{}-{}".format(metric,p)
            
            input_ids.append(input_id)
           
            map_panel.append(html.Div(html.Label(v.get("label", p).replace("_",' '), title=v.get("description","")), style={"margin-left":"30%"})),
            if p== "clf_type_score":
                map_panel.append(html.Div(dcc.Textarea(id=input_id, name=pillar,value=str(v.get("value" "")), style={"width":300, "height":150}), style={"margin-left":"30%"}))
            else:
                map_panel.append(html.Div(dcc.Input(id=input_id, name=pillar,value=str(v.get("value" "")), type='text', style={"width":200}), style={"margin-left":"30%"}))
            map_panel.append(html.Br())
    map_panel.append(html.Hr())      
    map_panel.append(html.Div([html.Label("Load saved mappings",style={"margin-left":10}),
                            dcc.Dropdown(
                                id='mapping-dropdown-{}'.format(pillar),
                                options=list(map(lambda name:{'label': name[:-5], 'value': "configs/mappings/{}/{}".format(pillar,name)} ,os.listdir("configs/mappings/{}".format(pillar)))),
                                value='configs/mappings/{}/default.json'.format(pillar),
                                style={'width': 200},
                                className = pillar
                            )],style={"margin-left":"30%","margin-bottom":15}))
                            
                        
    map_panel.append(html.Div(html.Button('Apply', id='apply-mapping-{}'.format(pillar), style={"background-color": "gold","margin-left":"30%","margin-bottom":15,"width":200})
                     , style={'width': '100%', 'display': 'inline-block'}))
    map_panel.append(html.Div(html.Button('Save', id='save-mapping-{}'.format(pillar), style={"background-color": "green","margin-left":"30%","width":200})
                     , style={'width': '100%', 'display': 'inline-block'}))
    return map_panel , input_ids

def metrics_list(metrics):
    elements = []
    for metric_id in metrics:
        metric_name = id_to_name(metric_id)
        elements.append(html.Li(metric_name, className="text-left"))
    return html.Ul(elements)
        
