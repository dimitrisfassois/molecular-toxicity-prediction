import deepchem as dc
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np

app = dash.Dash(__name__)

# Load the dataframe
test_dataset = dc.data.DiskDataset('/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/data/test_dataset')

def smiles_to_svg(smiles):
    molecule = MolFromSmiles(smiles)
    Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DSVG(250, 250)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

# App layout
app.layout = html.Div([
    html.Label('Select a toxicity test:'),
    dcc.Dropdown(
        id='task-dropdown',
        options=[
            {'label': 'NR-AR', 'value': 0},
            {'label': 'NR-AR-LBD', 'value': 1},
            {'label': 'NR-AhR', 'value': 2}
        ],
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
])


# Provide options for update_svg callback function
@app.callback([Output('row-dropdown', 'options'), Output('row-dropdown', 'value')],
              Input('task-dropdown', 'value'))
def update_svg_dropdowns(first_dropdown_value):
    options_vals = list(np.where(test_dataset.y[:, first_dropdown_value] == 1)[0])
    options = [{'label': i, 'value': i} for i in options_vals]
    return options, None


# Callback to update the displayed SVG file
@app.callback([Output('svg-container', 'children'), Output('smiles-string', 'children')], [Input('row-dropdown', 'value')])
def update_svg(value):
    if value:
        smiles = test_dataset.ids[value]
        svg_data = smiles_to_svg(smiles)
        iframe = html.Iframe(srcDoc=svg_data, style={'width': '100%', 'height': '400px'})
        smiles_string = 'Molecule Smiles representation: ' + smiles
        return iframe, smiles_string
    else:
        return html.Div(), None


if __name__ == '__main__':
    app.run_server(debug=True)
