import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

# Set consistent color scheme for better readability
COLOR_SCHEME = {
    'movie': '#3498db',     # Blue
    'director': '#e74c3c',  # Red
    'star': '#2ecc71',      # Green
    'genre': '#f39c12',     # Orange/Yellow
    'decade': '#9b59b6',    # Purple
    'unknown': '#95a5a6'    # Gray
}

# Define consistent node sizes
NODE_SIZES = {
    'movie': 800,
    'director': 700,
    'star': 700,
    'genre': 700,
    'decade': 600,
    'unknown': 500
}

def visualize_subgraph(G, center_node, depth=1):
    """
    Visualize a subgraph centered on a specific node.
    
    Parameters:
    - G: NetworkX graph
    - center_node: Central node to visualize around
    - depth: Depth of neighbors to include
    """
    # Extract subgraph
    nodes = {center_node}
    current_nodes = {center_node}

    for _ in range(depth):
        next_nodes = set()
        for node in current_nodes:
            neighbors = set(G.successors(node)) | set(G.predecessors(node))
            next_nodes.update(neighbors)
        nodes.update(next_nodes)
        current_nodes = next_nodes

    subgraph = G.subgraph(nodes)

    # Create visualization
    pos = nx.spring_layout(subgraph, seed=42)  # Fixed seed for reproducibility

    # Group nodes by type
    node_groups = {
        'movie': [],
        'director': [],
        'star': [],
        'genre': [],
        'decade': []
    }
    
    for n, d in subgraph.nodes(data=True):
        node_type = d.get('type', 'unknown')
        if node_type in node_groups:
            node_groups[node_type].append(n)
    
    plt.figure(figsize=(14, 12))
    
    # Draw edges first so they appear behind nodes
    nx.draw_networkx_edges(subgraph, pos, alpha=0.4, arrows=True, arrowsize=15, 
                         edge_color='gray', width=1.5)
    
    # Highlight center node
    center_node_type = G.nodes[center_node].get('type', 'unknown')
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[center_node], 
                         node_color='white', 
                         node_size=NODE_SIZES.get(center_node_type, 600) * 1.2,
                         edgecolors=COLOR_SCHEME.get(center_node_type, 'gray'), 
                         linewidths=3)
    
    # Draw each node type with consistent colors
    legend_handles = []
    for node_type, nodes_list in node_groups.items():
        if nodes_list:
            color = COLOR_SCHEME.get(node_type, COLOR_SCHEME['unknown'])
            nx.draw_networkx_nodes(subgraph, pos, nodelist=nodes_list, 
                                 node_color=color, 
                                 node_size=NODE_SIZES.get(node_type, 500), 
                                 alpha=0.9)
            # Add to legend
            legend_handles.append(mpatches.Patch(color=color, label=node_type.capitalize()))
    
    # Add labels to all nodes for clarity
    nx.draw_networkx_labels(subgraph, pos, font_size=10, font_weight='bold')
    
    # Add edge weights but with smaller font to reduce clutter
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in subgraph.edges(data=True)}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)
    
    plt.legend(handles=legend_handles, loc='upper right')
    plt.title(f"Movie Graph Centered on '{center_node}'", fontsize=15, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_multi_paths(G, paths, title="Multiple Recommendation Paths", preferences=None):
    """
    Visualize multiple paths that led to recommendations with enhanced visual distinction.
    
    Parameters:
    - G: NetworkX graph
    - paths: List of paths, where each path is a list of nodes
    - title: Title for the visualization
    - preferences: Dictionary of user preferences to highlight input nodes
    """
    if not paths:
        print("No paths to visualize")
        return
    
    # Collect all unique nodes from all paths
    all_nodes = set()
    for path in paths:
        all_nodes.update(path)
    
    # Create subgraph
    subgraph = G.subgraph(all_nodes)
    
    # Use spring layout with custom parameters for better spacing
    pos = nx.spring_layout(subgraph, seed=42, k=0.5)
    
    plt.figure(figsize=(16, 12))
    
    # Get node types
    node_types = {node: G.nodes[node].get('type', 'unknown') for node in all_nodes}
    
    # Identify input nodes (from preferences)
    input_nodes = set()
    if preferences:
        for pref_type, nodes in preferences.items():
            for node in nodes:
                if node in all_nodes:
                    input_nodes.add(node)
    
    # Identify output nodes (last node of each path)
    output_nodes = set()
    for path in paths:
        if path and len(path) > 0:
            output_nodes.add(path[-1])
    
    # Regular nodes (not input or output)
    regular_nodes = [n for n in all_nodes if n not in input_nodes and n not in output_nodes]
    
    # Draw background edges first for better layering
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, arrows=False, edge_color='#cccccc', width=1)
    
    # Collect and draw path edges with distinct colors
    path_edges = []
    edge_colors = []
    all_edges = []
    
    # Use a visually distinct color palette for paths
    path_colors = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4', 
                   '#42d4f4', '#f032e6', '#fabebe', '#469990', '#e6beff']
    
    # Create path elements for the legend
    path_legend_elements = []
    
    # Process each path
    for i, path in enumerate(paths):
        if len(path) < 2:
            continue
            
        path_color = path_colors[i % len(path_colors)]
        edges = list(zip(path[:-1], path[1:]))
        all_edges.extend(edges)
        
        # Add edges to drawing lists
        for edge in edges:
            path_edges.append(edge)
            edge_colors.append(path_color)
        
        # Add path to legend if it connects input to output
        if path and path[0] in input_nodes and path[-1] in output_nodes:
            path_legend_elements.append(
                plt.Line2D([0], [0], color=path_color, lw=3, 
                        label=f'Path: {path[0]} → {path[-1]}')
            )
    
    # Draw colored path edges
    nx.draw_networkx_edges(subgraph, pos, edgelist=path_edges, 
                          edge_color=edge_colors, width=2.5, 
                          alpha=0.8, arrows=True, arrowsize=15)
    
    # Draw regular nodes (semi-transparent, no border)
    for node_type, color in COLOR_SCHEME.items():
        nodes = [n for n in regular_nodes if node_types.get(n) == node_type]
        if nodes:
            nx.draw_networkx_nodes(subgraph, pos, 
                                 nodelist=nodes, 
                                 node_color=color, 
                                 node_size=NODE_SIZES.get(node_type, 500) * 0.9,  # Slightly smaller
                                 alpha=0.4,  # More transparent
                                 node_shape='o')  # Regular circle shape
    
    # Draw input nodes (preferences) with distinctive styling - Diamond shape
    for node in input_nodes:
        node_type = node_types.get(node, 'unknown')
        color = COLOR_SCHEME.get(node_type, COLOR_SCHEME['unknown'])
        
        # Draw the input node as a diamond with black border
        nx.draw_networkx_nodes(subgraph, pos, 
                             nodelist=[node], 
                             node_color=color, 
                             node_size=NODE_SIZES.get(node_type, 500) * 1.2, 
                             node_shape='o',  # Round shape
                             edgecolors='black',
                             linewidths=3,
                             alpha=1.0)
    
    # Draw output nodes (recommendations) with distinctive styling - Star shape
    for node in output_nodes:
        node_type = node_types.get(node, 'unknown')
        color = COLOR_SCHEME.get(node_type, COLOR_SCHEME['unknown'])
        
        # Draw the output node as a star
        nx.draw_networkx_nodes(subgraph, pos, 
                             nodelist=[node], 
                             node_color=color, 
                             node_size=NODE_SIZES.get(node_type, 500) * 1.3, 
                             node_shape='*',  # Star shape
                             edgecolors='black',
                             linewidths=3,
                             alpha=1.0)
    
    # Add labels to ALL nodes - with better font for readability
    labels = {node: node for node in subgraph.nodes()}
    
    # Make labels for input and output nodes bold and slightly larger
    nx.draw_networkx_labels(subgraph, pos, 
                           labels={n: n for n in regular_nodes},
                           font_size=8, font_color='black')
    
    # Draw labels for input nodes (bold)
    nx.draw_networkx_labels(subgraph, pos, 
                           labels={n: n for n in input_nodes},
                           font_size=10, font_weight='bold', font_color='black')
    
    # Draw labels for output nodes (bold)
    nx.draw_networkx_labels(subgraph, pos, 
                           labels={n: n for n in output_nodes},
                           font_size=10, font_weight='bold', font_color='black')
    
    # Create node type legend
    node_legend_elements = []
    for node_type, color in COLOR_SCHEME.items():
        nodes = [n for n in subgraph.nodes() if node_types.get(n) == node_type]
        if nodes:
            node_legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                        markersize=10, label=node_type.capitalize())
            )
    
    # Add special legend entries for input and output nodes
    node_legend_elements.append(
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', 
                markersize=12, markeredgecolor='black', linewidth=3,
                label='Input (Preference)')
    )
    node_legend_elements.append(
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
                markersize=15, markeredgecolor='black', linewidth=3,
                label='Output (Recommendation)')
    )
    
    # Create legends
    plt.legend(handles=node_legend_elements, loc='upper right', title="Node Types", fontsize=9)
    
    # Add a second legend for paths if available
    if path_legend_elements:
        # Place second legend below the first one
        plt.legend(handles=path_legend_elements, loc='lower right', title="Recommendation Paths", fontsize=9)
        # Add back the first legend
        plt.gca().add_artist(plt.legend(handles=node_legend_elements, loc='upper right', 
                                      title="Node Types", fontsize=9))
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Display the plot without saving
    plt.show()


def visualize_single_movie_paths(G, movie, all_paths, preferences, rating=None, year=None):
    """
    Visualize all paths leading to a specific movie recommendation.
    
    Parameters:
    - G: NetworkX graph
    - movie: The movie node to visualize paths for
    - all_paths: All paths from find_all_related_movies
    - preferences: Dictionary of user preferences to highlight input nodes
    - rating: Optional movie rating to display in title
    - year: Optional movie release year to display in title
    """
    # Find all paths leading to this movie
    movie_paths = [p[4] for p in all_paths if p[0] == movie]  # p[4] contains the path
    
    if not movie_paths:
        print(f"No paths found for movie: {movie}")
        return
    
    # Get movie details for the title
    if rating is None and movie in G.nodes:
        rating = G.nodes[movie].get('rating', 'N/A')
    if year is None and movie in G.nodes:
        year = G.nodes[movie].get('year', 'N/A')
    
    # Create title with movie details
    if rating and year:
        title = f"Why We Recommended: {movie} ({year}, Rating: {rating})"
    else:
        title = f"Why We Recommended: {movie}"
    
    # Visualize these paths
    visualize_multi_paths(G, movie_paths, title=title, preferences=preferences)


def create_recommendation_report(G, preferences, recommendations, all_paths):
    """
    Generate a visual report showing recommendation paths.
    
    Parameters:
    - G: NetworkX graph
    - preferences: Dictionary of user preferences
    - recommendations: List of (movie, score) tuples
    - all_paths: All paths from find_all_related_movies
    """
    # Collect all relevant paths for all recommendations
    relevant_paths = []
    for movie, _ in recommendations:
        movie_paths = [p[4] for p in all_paths if p[0] == movie]  # p[4] contains the path
        relevant_paths.extend(movie_paths)
    
    # Create a single visualization with all paths
    overview_title = "From Your Preferences to Recommendations"
    visualize_multi_paths(G, relevant_paths, title=overview_title, preferences=preferences)
    
    # Create individual visualization for each recommended movie
    for i, (movie, score) in enumerate(recommendations, 1):
        print(f"\nGenerating visualization for recommendation #{i}: {movie}")
        visualize_single_movie_paths(G, movie, all_paths, preferences)