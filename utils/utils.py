import os
import re
import json
import pandas as pd
import numpy as np
import tqdm
import pickle
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pymongo import MongoClient
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors as mcolors


#db = MongoClient("mongodb://localhost:27017")['tree']
#animalsDB = db.animals

db = MongoClient("mongodb://localhost:27017")['tree2']
animalsDB = db.summary

def setup_run():

    #pull in settings from settings.json:
    with open('settings.json') as f:
        settings = json.load(f)

    run_path = os.path.join('tree',
    "n="+str(settings['sample_size']) + ' ' + settings['sample_type'] + " k="+str(settings['k']) + " prefetch="+str(settings['prefetch_pool_size']) + " prek="+str(settings['prek']),
    "min_d="+ str(settings['min_lineage_length']) + " max_d="+str(settings['max_lineage_length']) + " max_l="+str(settings['max_lineages']) + " skip="+str(settings['sample_n_ancestors']))

    # Make sure those directories exist
    os.makedirs(run_path, exist_ok=True)

    print(f'Run path: {run_path}')

    settings['run_path'] = run_path

    settings['hover_data'] = ['_id'].extend(settings['stat_cols'])

    # Save settings to the run path
    with open(os.path.join(run_path, 'settings.json'), 'w') as f:
        json.dump(settings, f)

    return settings


def get_parent(animal):
    if animal["_id"] == animal["parent"]:
        return None
    parent = animalsDB.find_one({'_id': animal['parent']})
    if parent:
        return parent
    else:
        return None


def simplify_lineage(ancestors, sample_n_ancestors):
    # choose every nth ancestor
    return ancestors[::sample_n_ancestors]


def deduplicate_edges(run_path, edgeDF):

    ddEdgeDF_path = os.path.join(run_path, 'deduped_edges.csv')

    if os.path.exists(ddEdgeDF_path):
        ddEdgeDF = pd.read_csv(ddEdgeDF_path)

    else:
        # Deduplicate and save the lineage as an array of lineages
        ddEdgeDF = edgeDF.drop_duplicates(subset=['source', 'target']).copy()
        ddEdgeDF.to_csv(os.path.join(run_path, 'deduped_edges.csv'), index=False)
        print(f'Original shape: {edgeDF.shape}, Deduplicated shape: {ddEdgeDF.shape}')

    return ddEdgeDF



def normalize_series(series):
    min = series.min()
    max = series.max()
    std = series.std()

    def normalize(x):
        try:
            return (x - min) / (max - min)
        except ZeroDivisionError:
            return 0
    
    return series.apply(normalize)


def prefetch_by_kmeans(run_path, prek, cluster_cols, prefetch_pool_size=34_000_000):

    save_path = os.path.join(run_path, 'sampled_animals.pkl')

    if os.path.exists(save_path):
        print('Loading cached prefetch from file')
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    # Get 1M animals from the database
    print(f'Prefetching {prefetch_pool_size} animals from the database')
    animals = list(animalsDB.aggregate([{'$sample': {'size': prefetch_pool_size}}]))
    
    # Store in a dataframe
    df = pd.DataFrame(animals)
    print(f'Prefetched {df.shape[0]} animals with {df.shape[1]} columns')
    
    # Normalize the columns specified for clustering
    print(f'Normalizing columns: {cluster_cols}')
    for col in cluster_cols:
        df[col] = normalize_series(df[col])
    
    # Cluster the animals based on their normalized stats
    print(f'Running K-means clustering with k={prek} on prefetched animals')
    X = np.array(df[cluster_cols])
    kmeans = KMeans(n_clusters=prek, random_state=0).fit(X)

    # What is the smallest cluster size?
    cluster_sizes = np.bincount(kmeans.labels_)
    print(f'Smallest cluster size: {cluster_sizes.min()}')

    # Sample 10_000 animals from each k cluster
    print(f'Sampling 10% of the smallest cluster size from each cluster')
    sampled_animals = []
    for i in range(prek):
        cluster = df[kmeans.labels_ == i]
        sampled_cluster = cluster.sample(int(cluster_sizes.min() / 10))  # Make sure it's less than the smallest cluster
        sampled_animals.extend(sampled_cluster.to_dict(orient='records'))  # Convert back to list of dicts

    # Save the sampled animals to a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(sampled_animals, f)

    # Return in the same format as list(animalsDB.aggregate())
    return sampled_animals



def get_edge_df(run_path, k, prek, cluster_cols, prefetch_pool_size, sample_size, sample_type, sample_n_ancestors, max_lineages, max_lineage_length, min_lineage_length, dedupe=True):
    
    edgeDF_path = os.path.join(run_path, 'edges.csv')

    if os.path.exists(edgeDF_path):
        print('Loading cached edgeDF from file')
        edgeDF = pd.read_csv(edgeDF_path)
        if k == 'lineage_count':
            k = len(edgeDF['lineage'].unique())

    else:
        if sample_type == 'random':
            # Find N random animals sampled from the database
            sample = list(animalsDB.aggregate([{'$sample': {'size': sample_size}}]))

        elif sample_type == 'recent':
            # Find N most recent animals
            sample = list(animalsDB.find().sort('created_at', -1).limit(sample_size))

        elif sample_type == 'oldest':
            # Find N oldest animals
            sample = list(animalsDB.find().sort('created_at', 1).limit(sample_size))

        elif sample_type == 'kmeans':
            sample = prefetch_by_kmeans(run_path, prek, cluster_cols, prefetch_pool_size)

        else:
            raise ValueError('Inval_id sample type')

        # Recurse through the animal's parents to create a list of all ancestors
        seen_nodes = []
        leaf_nodes = []
        start_nodes = []
        series_source = []
        series_target = []
        series_lineage_id = []
        long_lineage_count = 0

        for animal in tqdm.tqdm(sample):

            if max_lineages and long_lineage_count >= max_lineages:
                break

            ancestors = []
            parent = get_parent(animal)
            if parent is None: # skip animals with no parent
                continue

            # Recurse through the animals' parents to create a list of all ancestors
            while parent and (len(ancestors) < max_lineage_length) and (parent not in seen_nodes): # don't save the whole lineage, just the most recent max_lineage_length ancestors
                seen_nodes.append(parent)
                ancestors.append(parent)
                parent = get_parent(parent)

            if len(ancestors) > min_lineage_length: # only retain lineages longer than the minimum threshold
                long_lineage_count += 1

                # Take every nth ancestor
                simplified_ancestors = simplify_lineage(ancestors, sample_n_ancestors=1)

                # Add the start animal to the list of leaf nodes
                leaf_nodes.append(animal['_id'])

                # Add the lineage to the series
                series_source.append(animal['_id'])
                series_target.append(simplified_ancestors[0]['_id'])
                series_lineage_id.append(animal['_id'])

                # Add the rest of the lineage to the series
                for i in range(1, len(simplified_ancestors)):
                    series_source.append(simplified_ancestors[i-1]['_id'])
                    series_target.append(simplified_ancestors[i]['_id'])
                    series_lineage_id.append(animal['_id'])

                # Add the oldest ancestor to the list of start nodes
                start_nodes.append(simplified_ancestors[-1]['_id'])

        print(f'Found {long_lineage_count} long lineages')

        edgeDF = pd.DataFrame({'source': series_source, 'target': series_target, 'lineage': series_lineage_id})

        def get_all_lineages(source):
            return set(edgeDF[edgeDF['source'] == source]['lineage'])

        edgeDF['lineages'] = [get_all_lineages(s) for s in edgeDF['source']]

        edgeDF.to_csv(os.path.join(run_path, 'edges.csv'), index=False)

    return deduplicate_edges(run_path, edgeDF) if dedupe else edgeDF



def get_nodeDF(run_path, edgeDF, k, stat_cols, cluster_cols, pca):

    if k == 'lineage_count':
        k = edgeDF['lineage'].nunique()
    
    nodeDF_path = os.path.join(run_path, 'nodes.csv')

    if os.path.exists(nodeDF_path):
        print('Loading cached nodeDF from file')
        nodeDF = pd.read_csv(nodeDF_path)

    else:
        nodeDF = pd.DataFrame({'_id': list(set(edgeDF['source'].values) | set(edgeDF['target'].values))})

        def isLeaf(_id):
            return (_id in edgeDF['target'].values) and not (_id in edgeDF['source'].values)

        def isStart(_id):
            return (_id in edgeDF['source'].values) and not (_id in edgeDF['target'].values)

        nodeDF['is_leaf'] = nodeDF['_id'].apply(lambda x: isLeaf(x))
        nodeDF['is_start'] = nodeDF['_id'].apply(lambda x: isStart(x))

        nodeDF['lineages'] = nodeDF['_id'].apply(lambda x: edgeDF[edgeDF['source'] == x]['lineages'].values[0] if x in edgeDF['source'].values else edgeDF[edgeDF['target'] == x]['lineages'].values[0])
        nodeDF['lineage'] = nodeDF['_id'].apply(lambda x: edgeDF[edgeDF['source'] == x]['lineage'].values[0] if x in edgeDF['source'].values else edgeDF[edgeDF['target'] == x]['lineage'].values[0])

        def get_stat(_id, col):
            animal = animalsDB.find_one({'_id': _id})
            if animal:
                return animal[col]
            else:
                return None

        for col in stat_cols:
            nodeDF[col] = nodeDF['_id'].apply(lambda x: get_stat(x, col))

        # change created to numeric for normalization
        nodeDF['created'] = pd.to_numeric(nodeDF['created'])

        # Create normalized columns for each stat
        for stat in stat_cols:
            nodeDF[f'{stat}_normalized'] = normalize_series(nodeDF[stat])

        # K-means cluster the nodes based on their stats
        print(f'Running K-means clustering with k={k}')
        X = np.array(nodeDF[[f'{stat}_normalized' for stat in cluster_cols]])
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        # Assign clusters to the nodes based on k-means labels
        nodeDF['cluster'] = [str(klabel) for klabel in kmeans.labels_]
        nodeDF['is_centroid'] = False

        # Insert dummy nodes for each centroid:
        centroid_rows = []
        for i, centroid in enumerate(kmeans.cluster_centers_):
            centroid_data = {f'{stat}_normalized': value for stat, value in zip(cluster_cols, centroid)}
            centroid_data['cluster'] = str(i)  # Use the index `i` to assign the correct cluster number
            centroid_data['_id'] = f'centroid_{i}'
            centroid_data['is_centroid'] = True
            centroid_rows.append(centroid_data)

        # Create a new DataFrame from the centroid rows and concatenate it with the original DataFrame
        centroidDF = pd.DataFrame(centroid_rows)
        nodeDF = pd.concat([nodeDF, centroidDF])

        # We have to get a new X array with the centroids included
        X = np.array(nodeDF[[f'{stat}_normalized' for stat in cluster_cols]])

        print(f'Running t-sne on {X.shape[0]} nodes')
        tsne = TSNE(n_components=pca, random_state=0)
        reduced_data = tsne.fit_transform(X)

        # Save the tsne results to the nodeDF
        for i in range(pca):
            nodeDF[f'tsne{i+1}'] = reduced_data[:, i]

        # Normalize the reduced data to fit within H and S ranges
        h_values = np.interp(reduced_data[:, 0], (reduced_data[:, 0].min(), reduced_data[:, 0].max()), (0, 360))
        s_values = np.interp(reduced_data[:, 1], (reduced_data[:, 1].min(), reduced_data[:, 1].max()), (0, 100))

        # Format hsl as a string 'hsl(0, 0%, 0%)'
        hsl_colors = [f'hsl({int(h)}, {int(s)}%, 50%)' for h, s in zip(h_values, s_values)]

        # Save the colors to the nodeDF
        nodeDF['hsl_color'] = hsl_colors

        # Perform PCA on the nodes
        print(f'Running PCA with {pca} components')
        PCA_result = PCA(n_components=pca)
        X_pca = PCA_result.fit_transform(X)
        
        for i in range(pca):
            nodeDF[f'pca{i+1}'] = X_pca[:, i]

        # Re-extract the centroids from the nodeDF
        centroidDF = nodeDF[nodeDF['is_centroid'] == True]
        centroidDF.to_csv(os.path.join(run_path, 'centroids.csv'), index=False)
        nodeDF = nodeDF[nodeDF['is_centroid'] == False]

        # Convert centroidDF into a lookup where keys are the cluster and values are the hsl_color
        centroidDF = centroidDF.set_index('cluster')
        centroid_color_lookup = centroidDF['hsl_color'].to_dict()

        # Look up the centroid_hsl for each node:
        nodeDF['centroid_hsl'] = nodeDF['cluster'].apply(lambda x: centroid_color_lookup[x])

        nodeDF['lineages'] = [str(lineage_set) for lineage_set in nodeDF['lineages']]
        nodeDF.sort_values('lineages', inplace=True)
        nodeDF['lineages_short'] = [str(lineage_set)[:10]+'...' if len(lineage_set) > 10 else lineage_set for lineage_set in nodeDF['lineages']]

        nodeDF.to_csv(nodeDF_path, index=False)

    return nodeDF

def get_nodes_and_edges(run_path, k, prek, prefetch_pool_size, sample_size, sample_type, sample_n_ancestors, max_lineages, max_lineage_length, min_lineage_length, stat_cols, cluster_cols, pca, dedupe=True):
    
        edgeDF = get_edge_df(run_path, k, prek, cluster_cols, prefetch_pool_size, sample_size, sample_type, sample_n_ancestors, max_lineages, max_lineage_length, min_lineage_length, dedupe=dedupe)
        nodeDF = get_nodeDF(run_path, edgeDF, k, stat_cols, cluster_cols, pca)
        for col in list(nodeDF.columns):
            if col not in edgeDF.columns:
                edgeDF[col] = edgeDF['source'].apply(lambda x: nodeDF[nodeDF['_id'] == x][col].values[0])

        return nodeDF, edgeDF


def format_label(label):

    label = re.sub(r"(?<=\w)([A-Z])", r" \1", label)

    return label.replace('_', ' ').title().replace('Pca', 'PCA ')


def save_fig(run_path, fig, filename):

    for template in ['plotly_dark']:
        fig.update_layout(template=template)

        for ext in ['html', 'svg', 'png']:
            full_filepath = os.path.join(run_path, f'{filename}_{template.split("_")[-1]}.{ext}')

            #if not os.path.exists(full_filepath):
            if ext == 'html':
                fig.write_html(full_filepath)
            elif ext == 'png':
                fig.write_image(full_filepath, scale=4)
            else:
                fig.write_image(full_filepath)

def make_violin(nodeDF, run_path, col):

    if col in ['sensors']:
        box = True
    else:
        box = False

    fig = go.Figure()
    if col == 'created':
        fig.update_xaxes(type='date')

    # Sort by mean lifeSpan of cluster
    cluster_order = nodeDF.groupby('cluster').agg({col:'mean'}).sort_values(col).index

    for i, cluster in enumerate(cluster_order):
        color = nodeDF['centroid_hsl'][nodeDF['cluster']==cluster].iloc[0]
        color_base = (',').join(color.split(',')[:2]).replace('hsl', 'hsla')
        line_color = f'{color_base},40%,1)'
        fill_color = f'{color_base},50%,0.5)'
        x_vals = nodeDF[col][nodeDF['cluster']==cluster]
        if col == 'created':
            x_vals = pd.to_datetime(x_vals)
        args = dict(
            x=x_vals,
            name=str(cluster),
            line_color=line_color,
            fillcolor=fill_color
        )
        if box:
            fig.add_trace(go.Box(**args))
            fig.update_xaxes(showticklabels=False)
        else:
            fig.add_trace(go.Violin(**args, box_visible=False, meanline_visible=True))

    fig.update_traces(orientation='h')

    if not box:
        fig.update_traces(side='positive', width=5, points=False)

    fig.update_layout(template='plotly_dark', xaxis_title=format_label(col), showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=800,
        height=200)
    fig.update_yaxes(showticklabels=False)
    save_fig(run_path, fig, f'violin_{col}')
    return fig

def make_violins(nodeDF, run_path, stat_cols):
    figs = []
    for col in stat_cols:
        fig = make_violin(nodeDF, run_path, col)
        fig.show()
        figs.append(fig)
    return

def make_scatterplot(edgeDF, nodeDF, x, y, size, run_path):

    print(f'Making 2d scatterplot. Node size is {nodeDF.shape[0]}')

    # Create the scatter plot
    fig = go.Figure()

    if size:
        if f'{size}_normalized' in edgeDF.columns:
            # Discretize normalized_column into 10 bins
            edgeDF['bin'] = pd.cut(edgeDF[f'{size}_normalized'], bins=10, labels=False)
        elif size in edgeDF.columns:
            # Discretize size column into 10 bins
            edgeDF['bin'] = pd.cut(edgeDF[size], bins=10, labels=False)
        else:
            print(f'Warning: {size} not in edgeDF columns. Using default size.')
            edgeDF['bin'] = 0
            size = None
    else:
        edgeDF['bin'] = 0

    # Get unique bins
    unique_bins = edgeDF['bin'].unique()

    for bin_value in unique_bins:
        bin_edges = edgeDF[edgeDF['bin'] == bin_value]
        
        xs = []
        ys = []
        
        for i, row in bin_edges.iterrows():
            source = row['source']
            target = row['target']
            source_row = nodeDF[nodeDF['_id'] == source]
            target_row = nodeDF[nodeDF['_id'] == target]
            
            if len(source_row) > 0 and len(target_row) > 0:
                xs.extend([source_row[x].values[0], target_row[x].values[0], None])
                ys.extend([source_row[y].values[0], target_row[y].values[0], None])

        if size:
            if f'{size}_normalized' in edgeDF.columns:
                avg = bin_edges[f'{size}_normalized'].mean()
            elif size in edgeDF.columns:
                avg = bin_edges[size].mean()
            else:
                print(f'Warning: {size} not in edgeDF columns. Using default size.')
                avg = 0.5
        else:
            avg = 0.5
        
        # Assign color based on bin_value or use a custom color map
        bin_color = f"rgba(100, 100, 100, {avg})"  # Example of HSL color assignment

        # Add trace for this bin
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(xs) if x=='created' else xs,
                y=ys,
                mode='lines',
                line=dict(color=bin_color, width=avg)
            )
        )


    # Add the nodes as markers
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(nodeDF[x]) if x=='created' else nodeDF[x],
        y=nodeDF[y],
        mode='markers',
        marker=dict(
            color=[c for c in nodeDF['hsl_color']],
            opacity=0.5,
            line=dict(width=0.1, color='#333')
        ),
        text=nodeDF['_id'],
        hoverinfo='text',
    ))

    if size:
        if f'{size}_normalized' in nodeDF.columns:
            marker_size = [s * 15 for s in nodeDF[f'{size}_normalized']]
        elif size in nodeDF.columns:
            marker_size = [s * 15 for s in nodeDF[size]]
        else:
            print(f'Warning: {size} not in nodeDF columns. Using default size.')
            marker_size = [15 for _ in range(nodeDF.shape[0])]

        fig.update_traces(marker=dict(size=marker_size))

    if 'created' not in [x, y]:
        # Import the centroids and plot those as diamonds
        centroidDF = pd.read_csv(os.path.join(run_path, 'centroids.csv'))
        fig.add_trace(go.Scatter(
            x=centroidDF[x],
            y=centroidDF[y],
            mode='markers',
            marker=dict(
                size=20,
                color=[c for c in centroidDF['hsl_color']],
                symbol='diamond',
                line=dict(width=1, color='White')
            ),
            text=f"Centro_id of Cluster {centroidDF['cluster']}",
            hoverinfo='text',
        ))

    title = f'<b>{format_label(x)} vs {format_label(y)}</b><br>Color: t-SNE'

    if size:
        title += f', Size: {format_label(size)}'

    # Update axes and layout
    fig.update_xaxes(title_text=format_label(x))
    fig.update_yaxes(title_text=format_label(y))
    fig.update_layout(
        template='plotly_dark',
        height=1000,
        width=1200,
        showlegend=False,
        title=title
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    if x=='created':
        fig.update_layout(width=1200)
        fig.update_xaxes(type='date')
    # Save the figure
    save_fig(run_path, fig, f'connected_scatter_x={x}_y={y}_size={size}')

    print(f'Done making 2d scatterplot. Node size is {nodeDF.shape[0]}')
    return fig


def make_3d_scatterplot(edgeDF, nodeDF, x, y, z, size, run_path, hover_data):


    print(f'Making 3d scatterplot. Node size is {nodeDF.shape[0]}')

    # Create the 3D scatter plot for nodes using go.Scatter3d directly
    fig = go.Figure()

    # Set up axis titles
    fig.update_scenes(
        aspectmode='cube',
        xaxis_title=format_label(x),
        yaxis_title=format_label(y),
        zaxis_title=format_label(z),
    )

    """

    # Discretize normalized_column into 10 bins
    edgeDF['bin'] = pd.cut(edgeDF[f'{size}_normalized'], bins=10, labels=False)

    # Get unique bins
    unique_bins = edgeDF['bin'].unique()

    for bin_value in unique_bins:
        bin_edges = edgeDF[edgeDF['bin'] == bin_value]

        xs = []
        ys = []
        zs = []
        
        for i, row in bin_edges.iterrows():
            source = row['source']
            target = row['target']
            source_row = nodeDF[nodeDF['_id'] == source]
            target_row = nodeDF[nodeDF['_id'] == target]

            if len(source_row) > 0 and len(target_row) > 0:
                xs.extend([source_row[x].values[0], target_row[x].values[0], None])
                ys.extend([source_row[y].values[0], target_row[y].values[0], None])
                zs.extend([source_row[z].values[0], target_row[z].values[0], None])
        
        avg = bin_edges[f'{size}_normalized'].mean()

        # Assign color and set width based on the average value in this bin
        bin_color = f"rgba(100, 100, 100, {avg})"
        
        # Add trace for this bin
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='lines',
                line=dict(color=bin_color, width=avg*2)  # Adjust width as needed
            )
        )

    """

    # Create the nodes
    fig.add_trace(
        go.Scatter3d(
            x=nodeDF[x],
            y=nodeDF[y],
            z=nodeDF[z],
            mode='markers',
            marker=dict(
                size=[s * 10 for s in nodeDF[f'{size}_normalized']],
                color=[c for c in nodeDF['hsl_color']],
                opacity=1,
                line=dict(width=0.5, color='darkslategray')
            ),
            hovertext=nodeDF['_id'],
            text=hover_data,
        )
    )

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye={'x': 1.25, 'y': 1.25, 'z': -0.25}
    )

    # Update layout
    fig.update_layout(
        scene_camera=camera,
        template='plotly_dark',
        height=800,
        width=800,
        showlegend=False,
        title=f'<b>{format_label(x)} vs {format_label(y)} vs {format_label(z)}</b><br>Color: t-SNE, Size: {format_label(size)}'
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    # Save the figure
    save_fig(run_path, fig, f'3d_scatter_x={x}_y={y}_z={z}_size={size}')

    print(f'Done making 3d scatterplot. Node size is {nodeDF.shape[0]}')
    return fig
