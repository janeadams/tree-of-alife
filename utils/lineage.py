import os
import pickle
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from utils.eda import make_edgelist
import plotly.io as pio

# Set default template:
pio.templates.default = "plotly_dark"
config = {'displayModeBar': False}

def compute_network(df, cache=True):
    """
    Compute the successor and predecessor counts.
    """
    path = f'lineage_graph_{df.shape[0]}'
    lineage_pickle = os.path.join('pickles', path+'.pkl')

    if cache and os.path.exists(lineage_pickle):
        print('Loading cached lineage pickle.')
        g = pickle.load(open(lineage_pickle, 'rb'))
        return g

    edge_df = make_edgelist(df)
    g = nx.from_pandas_edgelist(edge_df, source='source', target='target', create_using=nx.DiGraph)

    pickle.dump(g, open(lineage_pickle, 'wb'))

    return g

def count_lineages(df, cache=True):
    """
    Count the number of successors for each node.
    """
    g = compute_network(df)

    lineage_csv = os.path.join('csv', f'lineage_{df.shape[0]}.csv')
    
    if cache and os.path.exists(lineage_csv):
        print('Loading cached lineage pickle.')
        df = pd.read_csv(lineage_csv)
        return df

    # Find the longest path in the graph
    longest_path = nx.dag_longest_path(g)
    print(f'The longest path is {len(longest_path)} nodes long.')

    df['successors'] = df['ID'].apply(lambda x: list(g.successors(x)))
    df['predecessors'] = df['ID'].apply(lambda x: list(g.predecessors(x)))

    df['successorCount'] = [len(s) for s in df['successors']]
    df['predecessorCount'] = [len(p) for p in df['predecessors']]

    print(f'The average number of successors is {df["successorCount"].mean()}')
    print(f'The average number of predecessors is {df["predecessorCount"].mean()}')
    print(f'The maximum number of successors is {df["successorCount"].max()}')
    print(f'The maximum number of predecessors is {df["predecessorCount"].max()}')
    print(f'There are {df[df["successorCount"] > 1].shape[0]} nodes with more than 1 successor.')
    print(f'There are {df[df["predecessorCount"] > 1].shape[0]} nodes with more than 1 predecessor.')

    print('Filtering to nodes with more than 1 successor or predecessor.')
    df = df[(df['successorCount'] > 1) or (df['predecessorCount'] > 1)]

    hist = go.Figure()
    hist.add_trace(go.Histogram(x=df['successorCount'], name='Successor Count (> 1)'))
    hist.add_trace(go.Histogram(x=df['predecessorCount'], name='Predecessor Count (> 1)'))
    # Overlay both histograms
    hist.update_layout(barmode='overlay')
    hist.write_html(os.path.join('figs', f'successorCount_{df.shape[0]}.html'))
    hist.show(config=config)

    df[['ID', 'successorCount', 'predecessorCount']].to_csv(lineage_csv, index=False)
    return df

def filter_by_lineage_length(df, cache=True):
    """
    Filter the dataframe by the length of the lineages.
    """
    lineage_df, g = compute_lineages(df, cache=cache)
    # Get the top 20% of the successorCount nodes
    top_successors = df[df['successorCount'] > df['successorCount'].quantile(0.8)]
    # Get the top 20% of the predecessorCount nodes
    top_predecessors = df[df['predecessorCount'] > df['predecessorCount'].quantile(0.8)]

    df = pd.concat([top_successors, top_predecessors]).drop_duplicates()
    return df