import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
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
    html.Div(
        dcc.Input(id='tmp-input', type='text')
    )
], className='row')

def get_figure(Zeta):
    #  if selectedpoints_local and selectedpoints_local['range']:
        #  ranges = selectedpoints_local['range']
        #  selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            #  'y0': ranges['y'][0], 'y1': ranges['y'][1]}
    #  else:
        #  selection_bounds = {'x0': np.min(df[x_col]), 'x1': np.max(df[x_col]),
                            #  'y0': np.min(df[y_col]), 'y1': np.max(df[y_col])}

    fig = px.scatter(x=Zeta[:, 0], y=Zeta[:, 1])

    #  fig.update_traces(selectedpoints=selectedpoints,
                      #  customdata=df.index,
                      #  mode='markers+text', marker={ 'color': 'rgba(0, 116, 217, 0.7)', 'size': 20 }, unselected={'marker': { 'opacity': 0.3 }, 'textfont': { 'color': 'rgba(0, 0, 0, 0)' }})

    #  fig.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5}, dragmode='select', hovermode=False)

    #  fig.add_shape(dict({'type': 'rect',
                        #  'line': { 'width': 1, 'dash': 'dot', 'color': 'darkgrey' }},
                       #  **selection_bounds))
    return fig


@app.callback(
    Output('mode1', 'figure'),
    [Input(component_id='tmp-input', component_property='value')]
)
def update_mode1_latent(value):
    #  selectedpoints = df.index
    #  for selected_data in [selection1, selection2, selection3]:
        #  if selected_data and selected_data['points']:
            #  selectedpoints = np.intersect1d(selectedpoints,
                #  [p['customdata'] for p in selected_data['points']])

    #  return [get_figure(df, "Col 1", "Col 2", selectedpoints, selection1),
            #  get_figure(df, "Col 3", "Col 4", selectedpoints, selection2),
            #  get_figure(df, "Col 5", "Col 6", selectedpoints, selection3)]
    return get_figure(Zeta1)


@app.callback(
    Output('mode2', 'figure'),
    [Input(component_id='tmp-input', component_property='value')]
)
def update_mode2_latent(value):
    return get_figure(Zeta2)


if __name__ == '__main__':
    app.run_server(debug=True)
