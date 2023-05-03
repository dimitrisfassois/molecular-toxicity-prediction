import deepchem as dc
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import fsspec
import torch
from lime import lime_tabular
from rdkit import Chem

from util.constants import CONST
from data_loaders.data_loaders import get_disk_dataset

app = dash.Dash(__name__)
# Expose Flask instance
server = app.server

# Load the model
project_id = "molecular-toxicity-prediction"
fs = fsspec.filesystem('gs', project=project_id)
n_features = 1024
n_tasks = len(CONST.TASKS)
dc_model = dc.models.MultitaskClassifier(n_tasks=n_tasks, n_features=n_features, layer_sizes=[1000])

checkpoint = "gs://molecular-toxicity-prediction/callback_checkpoints/multi_task_classifier/checkpoint.pt"
with fs.open(checkpoint, mode='rb') as f:
    data = torch.load(f, encoding='ascii')
dc_model.model.load_state_dict(data['model_state_dict'])

# Load the test dataset
project_id = "molecular-toxicity-prediction"
fs = fsspec.filesystem('gs', project=project_id)
test_data_dir = "gs://molecular-toxicity-prediction/data/test_dataset"
test_dataset = get_disk_dataset(fs, test_data_dir)
# test_dataset = dc.data.DiskDataset('/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/data/test_dataset')

# Set up Lime
feature_names = ["fp_%s" % x for x in range(n_features)]
explainer = lime_tabular.LimeTabularExplainer(test_dataset.X[:, :n_features],
                                              feature_names=feature_names,
                                              categorical_features=feature_names,
                                              class_names=['not toxic', 'toxic'],
                                              discretize_continuous=True)


def eval_model(dc_model, task_no, n_tasks):
    def eval_closure(x):
        ds = dc.data.NumpyDataset(x, n_tasks=n_tasks)
        predictions = dc_model.predict(ds)[:, task_no]
        return predictions
    return eval_closure


def smiles_to_svg(smiles):
    molecule = MolFromSmiles(smiles)
    Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DSVG(250, 250)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def fp_mol(mol, fp_length=1024):
    """
    returns: dict of <int:list of string>
        dictionary mapping fingerprint index
        to list of SMILES strings that activated that fingerprint
    """
    d = {}
    feat = dc.feat.CircularFingerprint(sparse=True, smiles=True, size=1024)
    retval = feat._featurize(mol)
    for k, v in retval.items():
        index = k % fp_length
        if index not in d:
            d[index] = set()
        d[index].add(v['smiles'])
    return d


# App layout
app.layout = html.Div([
    html.Label('Select a toxicity test:'),
    dcc.Dropdown(
        id='task-dropdown',
        options=[{k: v  for k, v in zip(('label', 'value'), [CONST.TASKS[i], i])} for i in range(n_tasks)],
        value=None
    ),
    html.Br(),
    html.Label('Select a molecule that is toxic for this test:'),
    dcc.Dropdown(
        id='row-dropdown',
        value=None
    ),
    html.Br(),
    html.Div(id='svg-container'),
    html.Br(),
    html.Div(id='smiles-string'),
    html.Br(),
    html.Div(id='predicted-prob-str'),
    html.Br(),
    html.Div(id='explain-str'),
    html.Br(),
])


# Provide options for update_svg callback function
@app.callback([Output('row-dropdown', 'options'), Output('row-dropdown', 'value')],
              Input('task-dropdown', 'value'))
def update_svg_dropdowns(first_dropdown_value):
    options = []
    if first_dropdown_value:
        ds = dc.data.NumpyDataset(test_dataset.X[:, :n_features].astype(float), n_tasks=n_tasks)
        preds = dc_model.predict(ds)[:, first_dropdown_value, 1]
        options_vals = list(np.where((test_dataset.y[:, first_dropdown_value] == 1) * (preds > 0.5))[0])
        options = [{'label': i, 'value': i} for i in options_vals]
    return options, None


# Callback to update the displayed SVG file
@app.callback([Output('svg-container', 'children'),
               Output('smiles-string', 'children'),
               Output('predicted-prob-str', 'children'),
               Output('explain-str', 'children')],
              [Input('row-dropdown', 'value'),
               Input('task-dropdown', 'value')])
def update_svg(value, first_dropdown_value):
    if value:
        smiles = test_dataset.ids[value]
        svg_data = smiles_to_svg(smiles)
        iframe = html.Iframe(srcDoc=svg_data, style={'width': '100%', 'height': '400px'})
        smiles_string = 'Molecule Smiles representation: ' + smiles

        model_fn = eval_model(dc_model, first_dropdown_value, n_tasks)
        exp = explainer.explain_instance(test_dataset.X[value, :n_features], model_fn, num_features=100, top_labels=1)

        predicted_prob = round(exp.predict_proba[1], 2)
        predicted_prob_str = f"Predicted probability for the molecule to be toxic: {predicted_prob}"

        activated_fragments = fp_mol(Chem.MolFromSmiles(test_dataset.ids[value]))
        fragment_weights = dict(exp.as_map()[1])

        explain_str = "Fragments within the molecule that contributed the most to the prediction, and their weights:\n"
        for index in activated_fragments:
            if index in fragment_weights:
                explain_str += "\n"
                explain_str += f"Fragment: {activated_fragments[index]}, "
                explain_str += f"Prediction weight: {fragment_weights[index]})"

        return iframe, smiles_string, predicted_prob_str, explain_str
    else:
        return html.Div(), None, None, None


if __name__ == '__main__':
    app.run_server(debug=True)
