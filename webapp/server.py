import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
import time
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = './files'
ALLOWED_DATASET_EXTENSIONS = {'csv'}
ALLOWED_MODEL_EXTENSIONS = {'k5'}
ALLOWED_FACTSHEET_EXTENSIONS = {'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# def allowed_file(filename):
#    return '.' in filename and \
#           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_new_directory_name():
    return str(time.time()).replace('.', '')


def allowed_dataset(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_DATASET_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print(get_new_directory_name())

    # print ("The current working directory is %s" % path)

    if request.method == 'POST':
        # check if the post request has the file part
        if 'dataset' not in request.files:
            flash('No dataset')
            return redirect(request.url)

        if 'model' not in request.files:
            flash('No model')
            return redirect(request.url)

        if 'factsheet' not in request.files:
            flash('No factsheet')
            return redirect(request.url)

        dataset = request.files['dataset']
        model = request.files['model']
        factsheet = request.files['factsheet']

        if dataset.filename == '':
            flash('No selected file')
            return redirect(request.url)

        print(dataset.filename, flush=True)
        if dataset and allowed_dataset(dataset.filename):
            base_path = '{0}/{1}'.format(app.config['UPLOAD_FOLDER'], get_new_directory_name())

            # create the directory to hold all the files
            os.makedirs(base_path)

            filename = secure_filename(dataset.filename)

            dataset.save(os.path.join(base_path, 'dataset.csv'))
            model.save(os.path.join(base_path, 'model.k5'))
            factsheet.save(os.path.join(base_path, 'factsheet.json'))

            return redirect(url_for('uploaded_file',
                                    filename="dataset.csv"))

    return render_template("index.html", content="Testing")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route("/analysis/<analysis_id>")
def analysis(analysis_id):
    base_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_directory, 'files/' + analysis_id)
    files = os.listdir(path)
    t = pd.read_csv(os.path.join(path, 'dataset.csv'))
    features = t.columns.values.tolist()
    print(features)
    print(type(t))
    print(t.head(10), flush=True)
    #table = t.reset_index(drop=True, inplace=True)
    table = t
    #print(table.)
    return render_template('analysis.html', title='Analysis', files=files, table=table, features=features)


@app.route("/analyses", methods=['GET'])
def analyses():
    path = "files"

    base_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_directory, 'files')
    directories = os.listdir(path)
    return render_template('analyses.html', title='Analyses', analyses=directories)


if __name__ == "__main__":
    app.run(debug=True)