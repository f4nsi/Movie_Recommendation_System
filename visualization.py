import matplotlib.pyplot as plt
import networkx as nx


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