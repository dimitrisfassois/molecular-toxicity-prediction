import deepchem as dc
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.Draw import rdMolDraw2D
from dash.development.base_component import Component

app = dash.Dash(__name__)

# Load the dataframe
test_dataset = dc.data.DiskDataset('/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/data/test_dataset')

options = [{'label': i, 'value': i} for i in range(5)]

def smiles_to_svg(smiles):
    molecule = MolFromSmiles(smiles)
    Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DSVG(250, 250)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

# Define the layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='row-dropdown',
        options=options,
    ),
    html.Div(id='svg-container'),
    html.Div(id='smiles-string')
])

# Define the callback to update the displayed SVG file
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
