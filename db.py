from tqdm import tqdm
import os
import pickle
import plotly.express as px
from utils.eda import *
from utils.lineage import *
from tinydb import table, TinyDB, Query, where

# %%
def find_longest_paths(df):
    print('Finding longest paths...')

    pkl_path = 'pickles/longest_paths.pkl'

    #if os.path.exists(pkl_path):
        #print('Loading from pickle...')
        #with open(pkl_path, 'rb') as f:
            #longest_path = pickle.load(f)
        #return longest_path

    G = compute_network(df)

    # Perform topological sort
    topo_order = list(nx.topological_sort(G))
    
    # Initialize a dictionary to store the longest path length for each node
    longest_paths = {node: (0, []) for node in G.nodes}

    os.makedirs('data/paths', exist_ok=True)
    
    # Process nodes in topological order
    for node in tqdm(topo_order):
        current_length, current_path = longest_paths[node]
        for successor in G.successors(node):
            successor_length, successor_path = longest_paths[successor]
            new_length = current_length + 1
            if new_length > successor_length:
                longest_paths[successor] = (new_length, current_path + [successor])

        node_path_filepath = f'data/paths/{node}.pkl'
        if os.path.exists(node_path_filepath):
            continue
        with open(node_path_filepath, 'wb') as f:
            pickle.dump(current_path + [successor], f)
            f.close()
        

    #with open('longest_paths.pkl', 'wb') as f:
        #pickle.dump(longest_paths, f)
    return #longest_paths

# %%
df_filepath = 'data/longest_paths.csv'
if os.path.exists(df_filepath):
    df = pd.read_csv(df_filepath)
else:
    df = load_sample('all')
    df.head(5)
    print('Computing network...')
    longest_paths = find_longest_paths(df)
    df['path_length'] = df['ID'].map(lambda x: longest_paths[x][0])
    df.to_csv('data/longest_paths.csv', index=False)
    
longest_paths = find_longest_paths(df)
df.sort_values('path_length', ascending=False).head(10)
px.histogram(df, x='path_length', nbins=100, log_y=True, title='Distribution of Path Lengths').show()

# %%
def assign_coordinates(df):

    # Get the top 20% of the df by path length
    df = df.head(int(0.2 * len(df)))

    # Find the central path (longest path)
    def lookup_path(node):
        node_path_filepath = f'data/paths/{node}.pkl'
        if os.path.exists(node_path_filepath):
            with open(node_path_filepath, 'rb') as f:
                loaded = pickle.load(f)
                f.close()
                return loaded
        else:
            with open('error.log', 'a') as f:
                f.write(f'Error: No pickle for {node}\n')
                f.close()

    _, central_path = lookup_path(df.iloc[0]['ID'])
    num_nodes = len(central_path)
    
    # Assign coordinates to the central path
    coordinates = {}
    x_center = 0
    z_center = 0
    for i, node in enumerate(central_path):
        coordinates[node] = (x_center, df.loc[df['ID'] == node, 'created'].values[0], z_center)
    
    # Splay out the branches using an L-system inspired approach
    angle_increment = np.pi / 4
    for _, path in longest_paths[1:]:
        angle = 0
        for node in tqdm(path):
            if node not in coordinates:
                parent = next(pred for pred in G.predecessors(node) if pred in coordinates)
                px, py, pz = coordinates[parent]
                distance = 1  # Distance between nodes (can be adjusted)
                x = px + distance * np.cos(angle)
                z = pz + distance * np.sin(angle)
                y = df.loc[df['ID'] == node, 'created'].values[0]
                coordinates[node] = (x, y, z)
                angle += angle_increment
    
    # Put coordinates back into the DataFrame
    df['x'] = df['ID'].map(lambda x: coordinates[x][0])
    df['y'] = df['ID'].map(lambda x: coordinates[x][1])
    df['z'] = df['ID'].map(lambda x: coordinates[x][2])

    df.to_csv('data/coordinates.csv', index=False)
    
    return df

# %%
assign_coordinates(df)


