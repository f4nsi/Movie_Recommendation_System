import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# 1. load data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


# 2. build graph
def build_graph(df):
    G = nx.DiGraph()

    # iterate through each row in the DataFrame
    for idx, row in df.iterrows():
        movie_title = row['Name']
        director = row['Director']
        genres = row['Genre'].split(',')
        stars = row['Stars'].split(',')

        # add movie nodes and rating
        G.add_node(movie_title, type='movie',
                   rating=row['Rating'],
                   year=row['Release_Year'], duration=row['Duration'])

        # add director nodes and edges
        if G.has_node(director) is False:
            G.add_node(director, type='director')
        G.add_edge(director, movie_title, weight=1.0)  # director -> movie
        G.add_edge(movie_title, director, weight=0.5)  # movie -> director

        # add genre nodes and edges
        for genre in genres:
            genre = genre.strip()
            if G.has_node(genre) is False:
                G.add_node(genre, type='genre')
            G.add_edge(genre, movie_title, weight=0.7)  # genre -> movie
            G.add_edge(movie_title, genre, weight=0.3)  # movie -> genre

        # add star nodes and edges
        for star in stars:
            star = star.strip()
            if G.has_node(star) is False:
                G.add_node(star, type='star')
            G.add_edge(star, movie_title, weight=0.8)  # star -> movie
            G.add_edge(movie_title, star, weight=0.4)  # movie -> star

    return G


def visualize_subgraph(G, center_node, depth=1):
    """可视化以特定节点为中心的子图"""
    # 提取子图
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

    # 可视化
    pos = nx.spring_layout(subgraph)

    # 按节点类型分组绘制
    movie_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('type') == 'movie']
    director_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('type') == 'director']
    star_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('type') == 'star']
    genre_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('type') == 'genre']

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(subgraph, pos, nodelist=movie_nodes, node_color='blue', node_size=500, alpha=0.8, label='Movies')
    nx.draw_networkx_nodes(subgraph, pos, nodelist=director_nodes, node_color='red', node_size=400, alpha=0.8, label='Directors')
    nx.draw_networkx_nodes(subgraph, pos, nodelist=star_nodes, node_color='green', node_size=300, alpha=0.8, label='Stars')
    nx.draw_networkx_nodes(subgraph, pos, nodelist=genre_nodes, node_color='yellow', node_size=300, alpha=0.8, label='Genres')

    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, arrows=True)
    nx.draw_networkx_labels(subgraph, pos, font_size=8)

    plt.legend()
    plt.title(f"Movie Graph Centered on '{center_node}'")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    df = load_data('./IMDB_Top_250_Movies.csv')

    G = build_graph(df)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    visualize_subgraph(G, 'Christopher Nolan', depth=2)