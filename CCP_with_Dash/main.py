import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from scipy.spatial import distance as dist
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
from datasets import load_data
from tsom import TSOM


X, labels_animal, labels_feature = load_data(retlabel_animal=True, retlabel_feature=True)
tsom = TSOM(L1=2, L2=2, K1=10, K2=10,
            sigma1_max=2.2, sigma2_max=2.2,
            sigma1_min=0.2, sigma2_min=0.2,
            tau1=50, tau2=50)
history, Zeta1, Zeta2 = tsom.fit(X, num_epoch=50)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([
    html.H1('Conditional Component Plane'),
    html.Div(
        dcc.Graph(id='mode1'),
        className='four columns'
    ),
    html.Div(
        dcc.Graph(id='mode2'),
        className='four columns'
    ),
], className='row')


def get_figure(Zeta, cp):
    fig = px.imshow(cp)
    return fig


@app.callback(
    Output('mode1', 'figure'),
    [Input(component_id='mode2', component_property='clickData')]
)
def update_mode1_latent(clickData):
    clicked_unit = get_bmu(Zeta1, clickData)
    tmp = history.Y[-1, :, clicked_unit, :]
    component_plane = np.sqrt(np.sum(tmp * tmp, axis=1)).reshape(10, 10)
    return get_figure(Zeta1, cp=component_plane)


@app.callback(
    Output('mode2', 'figure'),
    [Input(component_id='mode1', component_property='clickData')]
)
def update_mode2_latent(clickData):
    print(clickData)
    clicked_unit = get_bmu(Zeta2, clickData)
    tmp = history.Y[-1, clicked_unit, :, :]
    component_plane = np.sqrt(np.sum(tmp * tmp, axis=1)).reshape(10, 10)
    return get_figure(Zeta2, cp=component_plane)


def get_bmu(Zeta, clickData):
    clicked_point = [[clickData['points'][0]['x'], clickData['points'][0]['y']]] if clickData else [[0, 0]]
    clicked_point = np.array(clicked_point)
    dists = dist.cdist(Zeta, clicked_point)
    unit = np.argmin(dists, axis=0)
    return unit[0]


if __name__ == '__main__':
    app.run_server(debug=True)
