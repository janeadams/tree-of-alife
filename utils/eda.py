import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
#import nx_cugraph as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Set default template:
pio.templates.default = "plotly_dark"
config = {'displayModeBar': False}

quantVars = ["lifeSpan", "speed", "maxEnergy", "kidEnergy", "sensors", "nkids", "pgmDeath", "successorCount", "predecessorCount"]#, "deathLuck"]
colors = px.colors.qualitative.Plotly
quantVarColors = dict(zip(quantVars, colors))

bias_presets = {
    'lifeSpan': 0.3,
    'deathLuck': 5,
    'pgmDeath': 0.5,
}

def load_sample(size='all', cache=True):
    """
    Load a sample of the data. If the sample file does not exist, create it.
    """
    print(f"Loading sample of size {size}...")
    source_path = os.path.join('data','SummaryIndividuals.csv')
    if size == 'all':
        return pd.read_csv(os.path.join('data','SummaryIndividuals.csv'))
    sample_path = os.path.join('data',f"sample_{size}.csv")
    # Check if sample file exists:
    if (not os.path.exists(sample_path)) or (not cache):
        print("Creating sample file...")
        df = pd.read_csv(source_path)
        sample = df.sample(size)
        sample.to_csv(sample_path, index=False)
    else:
        print("Sample file found. Loading sample file...")
        sample = pd.read_csv(sample_path)
    return sample

def compute_deathLuck(df, showFig=True):
    """
    Compute the deathLuck variable, which is the ratio of the programmed death age to the actual lifespan of the creature.
    """
    print(f"Computing deathLuck for {df.shape[0]} observations...")
    df.loc[:, 'deathLuck'] = df['lifeSpan'] / df['pgmDeath']
    if showFig:
        fig = px.scatter(df, x='pgmDeath', y='lifeSpan', color='deathLuck')
        fig.to_html(os.path.join('figs', 'deathLuck.html'))
        fig.show(config=config)
    return df
    

def check_bias(df, biased_df, biasStr, var='lifeSpan'):
    """
    Compare the distribution of a variable in the original and biased samples.
    """
    print(f"Visualizing bias effect of {biasStr} on {var}...")
    compare(df[var],
            biased_df[var],
            f'Original {var}',
            f'{var} Biased by {biasStr}',
            var=var)

def compare(arr1, arr2, title1, title2, add_title="", var=None):
    """
    Compare two arrays with a histogram plot.
    """
    print(f"Comparing {title1} and {title2}...")
    arr1 = arr1.sample(min(10_000, len(arr1)))
    arr2 = arr2.sample(min(10_000, len(arr2)))
    hist1 = px.histogram(arr1, nbins=50, title=title1)
    hist2 = px.histogram(arr2, nbins=50, title=title2)
    fig = make_subplots(rows=1, cols=2, subplot_titles=[title1, title2])
    fig.add_trace(hist1['data'][0], row=1, col=1)
    fig.add_trace(hist2['data'][0], row=1, col=2)
    fig.update_layout(title_text=f"Biased Sample {add_title}", showlegend=False)
    color = quantVarColors[var] if var in quantVarColors else colors[0]
    fig.update_traces(marker=dict(color=color))
    path = f"compare_{title1}_{title2}_{add_title}".lower().replace(" ", "_").replace(":"," -")
    local_path = os.path.join('figs', path+'.html')
    fig.write_html(local_path)
    global_path = os.path.join(os.getcwd(), local_path)
    print(f"Saved figure to {global_path}")
    fig.show(config=config)

def bias(df, cache=False, alpha=0.2, var='lifeSpan', makeFig=True, size=None):
    """
    Create a biased sample of the data based on the exponential weight of a variable. We can think of this as individuals being more 'memorable' if they lived longer (lifeSpan), or unusually long (deathLuck), for example.
    """
    df = df.copy()
    if (size is None) or (size > df.shape[0]):
        size = min(100_000, df.shape[0])
        print("Sampling 100,000 observations for biasing...")

    print(f"Creating biased sample for {size} observations..."
          f" biasing by {var} @ alpha = {alpha}...")
    path = f"biased_sample_{var}_{size}_{alpha}"
    data_path = os.path.join('data', path+'.csv')
    if (not os.path.exists(data_path)) or (not cache):
        def exponential_weight(x, alpha=alpha):
            return np.exp(alpha * x)

        # Apply the weights to the observations
        weights = exponential_weight(df[var])
        normalized_weights = weights / weights.sum()

        # Sample according to the weighted probabilities
        sampled_indices = np.random.choice(len(df[var]), size=size, p=normalized_weights)
        biased_df = df.iloc[sampled_indices]
        biased_df.to_csv(data_path, index=False)
    else:
        biased_df = pd.read_csv(data_path)
    if makeFig:
            compare(df[var], biased_df[var], "Original", "Biased", f"({var} @ a={alpha})", var=var)
    return biased_df

def run_biasing_workflow(df, bias_by, alpha=None, size=None, cache=False):
    """
    Run the biasing workflow for a given variable.
    """
    if alpha is None:
        alpha = bias_presets.get(bias_by, 0.5)
    print(f"Running biasing workflow for {df.shape[0]} observations, biasing by {bias_by} @ alpha = {alpha}...")
    biased_df = bias(df,var=bias_by,alpha=alpha, size=size, cache=cache)
    for var in bias_presets.keys():
        if (var != bias_by) and (var in df.columns):
            # Cross-check biasing effect on other variables
            print(f"Cross-checking biasing effect on {var}...")
            check_bias(df, biased_df, f'{bias_by} @ {alpha}', var)
            print()
    print(f"Finished biasing workflow for {df.shape[0]} observations.")
    print(f"Biased dataframe size is {biased_df.shape[0]}.")
    return biased_df

def ridgeline(df, quantVars=quantVars, cache=True):
    """
    Create a ridgeline plot for the quantitative variables in the dataframe.
    """
    print(f"Creating ridgeline plot for {df.shape[0]} observations...")
    size = df.shape[0]
    if size > 10_000:
        print("Sampling 10,000 observations for ridgeline plot...")
        size = 10_000
        df = df.sample(size)
    filename = f"ridgeline_{size}"
    pickle_path = os.path.join('pickles', filename+'.pkl')
    if cache and os.path.exists(pickle_path):
        return pickle.load(open(pickle_path, 'rb'))
    if quantVars is None:
        quantVars = df.select_dtypes(include='number').columns
    # Create subplots for each quantitative variable:
    fig = go.Figure()
    for i, q in enumerate(quantVars):
        fig.add_trace(go.Violin(
            x=df[q],
            opacity=0.6,
            name=q,
            side='positive',
            hoverinfo='text',
            text=df[q],
            marker=dict(size=2, color=colors[i]),
            meanline_visible=True,
            hoveron='points',
            ))
        # Add annotation at median:
        fig.add_annotation(
            x=df[q].median(),
            y=i,
            text=f"{q}<br>{df[q].median():0.0f}",
            yshift=20,
            xshift=50,
            align='left',
            showarrow=False,
            font=dict(color=colors[i]),
        )
    fig.update_layout(
        height=300, width=400,
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
        violingap=0, violinmode='overlay'
    )
    fig.update_traces(
        orientation='h', side='positive',
        width=3
        )
    fig.update_yaxes(
        showticklabels=False
    )
    fig_path = os.path.join('figs', filename+'.html')
    fig.write_html(fig_path)
    global_fig_path = os.path.join(os.getcwd(), fig_path)
    print(f"Saved figure to {global_fig_path}")
    pickle.dump(fig, open(pickle_path, 'wb'))
    fig.show(config=config)

def scatter_3d(df, quantVars=quantVars, x=None, y=None, z=None, color=None):
    """
    Create a 3D scatter plot for the quantitative variables in the dataframe.
    """
    print(f"Creating 3D scatter plot for {df.shape[0]} observations...")
    if quantVars is None:
        quantVars = df.select_dtypes(include='number').columns
    
    varMap = {'x': x, 'y': y, 'z': z, 'color': color}
    
    # Initialize the axes with default values if not provided
    for i, axis in enumerate(varMap.keys()):
        if varMap[axis] is None:
            varMap[axis] = quantVars[i]
    
    # Create the initial 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=df[varMap['x']],
        y=df[varMap['y']],
        z=df[varMap['z']],
        mode='markers',
        marker=dict(
            size=2,
            color=df[varMap['color']],
        )
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title=varMap['x'],
            yaxis_title=varMap['y'],
            zaxis_title=varMap['z']
        )
    )
    size = df.shape[0]
    path = os.path.join('figs', f"scatter3d_{size}")
    for axis in varMap.keys():
        path += f"_{varMap[axis]}"
    path += ".html"
    fig.write_html(path)
    global_path = os.path.join(os.getcwd(), path)
    print(f"Saved figure to {global_path}")
    fig.show(config=config)


def make_edgelist(df, source='parent'):
    """
    Create an edge list from the dataframe.
    """
    print(f'Making edge list for {df.shape[0]} nodes...')
    edge_df = df[[source, 'ID']].dropna().rename(columns={ source:'source', 'ID':'target'})
    print(f'Edge list has {edge_df.shape[0]} edges.')
    return edge_df

def compute_positions(df, biased_df, x='created', y='maxEnergy', z='kidEnergy', color='sensors', source='parent'):
    """
    Compute the positions of the nodes and edges in the graph.
    """
    print(f'Computing positions for {df.shape[0]} nodes...')
    
    edge_df = make_edgelist(biased_df, source=source)

    # Filter node_df to only include nodes with edges:
    relevant = list(set(edge_df['source']).union(set(edge_df['target'])))
    # Filter the node_df down to indices that are in the relevant list:
    node_df = df[df['ID'].isin(relevant)].copy().set_index('ID')
    # Create a dictionary of node positions:
    node_dict = node_df.to_dict(orient='index')

    prev = edge_df.shape
    # For now, we just drop the orphaned targets
    edge_df = edge_df[edge_df['source'].isin(node_df.index)]

    print(f'Dropped {prev[0] - edge_df.shape[0]} orphaned targets; {edge_df.shape[0]} edges remain.')
    
    g = nx.from_pandas_edgelist(edge_df, source='source', target='target')

    # set edge positions:
    def get_loc(node_id, var):
        node_info = node_dict.get(node_id, None)
        if node_info is None:
            print(f'Node {node_id} not found in node_dict.')
            return None
        var_val = node_info.get(var, None)
        if var_val is None:
            print(f'Variable {var} not found in node_dict for node {node_id}.')
            return None
        return var_val

    edge_df['Xs'] = [[get_loc(s,x), get_loc(t,x)] for s, t in zip(edge_df['source'], edge_df['target'])]
    edge_df['Ys'] = [[get_loc(s,y), get_loc(t,y)] for s, t in zip(edge_df['source'], edge_df['target'])]
    edge_df['Zs'] = [[get_loc(s,z), get_loc(s,z)] for s, t in zip(edge_df['source'], edge_df['target'])]

    print(f'Computed positions for {edge_df.shape[0]} edges.')

    return node_df, edge_df

def visualize_graph(df, biased_df, x='created', y='maxEnergy', z='kidEnergy', color='sensors', source='parent'):
    """
    Visualize the graph of the data.
    """
    print(f'Visualizing graph for {df.shape[0]} nodes...')
    node_df, edge_df = compute_positions(df, biased_df, x=x, y=y, z=z, color=color, source=source)

    # Catch categorical colors:
    if color in ['ancestor', 'parent']:
        node_df['color'] = pd.Categorical([str(c) for c in node_df[color]])
        # Use plotly categorical colors:
        node_df['color'] = node_df['color'].cat.codes
        # Get a colormap by subdiving the color wheel:
        def get_colormap(i):
            return f'hsla({360*i//node_df["color"].nunique()}, 50%, 50%, 0.05)'
        node_df['color'] = [get_colormap(i) for i in node_df['color']]
    else:
        node_df['color'] = node_df[color]

    def flatten(l):
        # add a 0 between each pair of coordinates
        #return [x for pair in l for x in pair + [0]]
        return [x for pair in l for x in pair]

    # create edge trace
    edge_trace = go.Scatter3d(x=flatten(edge_df['Xs']), y=flatten(edge_df['Ys']), z=flatten(edge_df['Zs']),
                              mode='lines', line=dict(color='#fff', width=0.1), hoverinfo='none')
    # create node trace
    node_trace = go.Scatter3d(x=node_df[x], y=node_df[y], z=node_df[z],
                            mode='markers', name='nodes',
                            text=[f'{x}: {nx}<br>{y}: {ny}<br>{z}: {nz}<br>{color}: {ncolor}' for nx, ny, nz, ncolor in zip(node_df[x], node_df[y], node_df[z], node_df[color])],
                            hoverinfo='text',
                            marker=dict(
                                symbol='circle', size=2, color=node_df['color']))
    # create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        width=1000,
                        height=1000,
                        title=f'Graph, edges based on {source}, colored by {color}',
                        showlegend=True,
                        scene=dict(
                            xaxis=dict(title=x, showgrid=False, zeroline=False, showline=False, showticklabels=False),
                            yaxis=dict(title=y, showgrid=False, zeroline=False, showline=False, showticklabels=False),
                            zaxis=dict(title=z, showgrid=False, zeroline=False, showline=False, showticklabels=False)),
                        margin=dict(t=100), hovermode='closest'))
    fig.write_html(os.path.join('figs', f'graph_{x}_{y}_{z}_{color}_{source}.html'))
    fig.show(config=config)

