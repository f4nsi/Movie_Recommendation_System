import pandas as pd
import networkx as nx
from visualization import visualize_subgraph
from collections import defaultdict


# 1. load data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


# 2. build graph
def build_graph(df):
    '''
    Build a directed weighted graph based on the Dataset.
    The graph contains nodes for movies, directors, stars, and genres.
    Edges are created based on the relationships between these entities.
    
    Parameters:
    - df: the dataframe containing the movie data
    
    Returns:
    - G: a directed graph with nodes and edges
    '''
    G = nx.DiGraph()

    # iterate through each row in the csv file
    for idx, row in df.iterrows():
        movie_title = row['Name']
        director = row['Director']
        genres = row['Genre'].split(',')
        stars = row['Stars'].split(',')[0:3]

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


# 3. use BFS to find related movies based on a starting node
def find_related_movies(G, start_node, depth=2, pref_type=None):
    ''''
    BFS to find related movie paths. Used after the graph is generated.

    Parameters:
    - G: the graph
    - start_node: the starting node
    - depth: the depth of BFS

    Returns:
    - visited: a set of visited nodes
    '''
    visited = set()  # to keep track of visited nodes

    # queue for BFS, each element is a tuple (node, current_depth, weight, path)
    queue = [(start_node, 0, 1.0, [start_node])]
    related_movies = []  # to keep track of movies found

    while queue:
        current_node, current_depth, weight, path = queue.pop(0)

        # check if the current node is visited
        if current_node in visited:
            continue
        else:
            visited.add(current_node)

        # if the current node is a movie and not visited, add it to movies
        if G.nodes[current_node]['type'] == 'movie' and current_node != start_node:
            related_movies.append((current_node, current_depth, weight, pref_type, path))

        # if the current depth is less than the specified depth, continue BFS
        if current_depth < depth:
            neighbors = list(G.successors(current_node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    # calculate the new weight based on the edge weight
                    new_weight = weight * G[current_node][neighbor]['weight']
                    queue.append((neighbor, current_depth + 1, new_weight, path + [neighbor]))

    return related_movies


# 4. Find all related movies based on the user input
def find_all_related_movies(G, preferences):
    '''
    Find all related movies based on the user input.

    Parameters:
    - G: the graph
    - preferences: a dictionary of user preferences
        {'movie': ['movie_name'], 'director': ['director_name']}

    Returns:
    - related_movies: a list of related movies
    '''
    all_related_movies = []
    for pref_type, pref_values in preferences.items():
        for preference in pref_values:
            # find related movies for each preference
            if preference not in G.nodes:
                print(f"Node '{preference}' not found in the graph.")
                continue
            movies = find_related_movies(G, preference, depth=2, pref_type=pref_type)
            if movies:
                all_related_movies.extend(movies)
            else:
                print(f"No related movies found for '{preference}'.")

    return all_related_movies


# 5. calculate movie scores based on user preferences
def calculate_movie_scores(G, movie_name, all_related_movies):
    '''
    Calculate movie scores based on user preferences.
    The score is calculated based on the path taken to reach the movie.
    
    final score = base_score + path_scores
    where path_scores = sum(path_weight * pref_weight * length_penalty * 20)
    '''
    base_score = G.nodes[movie_name].get('rating') * 10  # base score from the movie node
    
    movie_paths = [p for p in all_related_movies if p[0] == movie_name]  # filter paths for the movie
    
    # filter the paths based on the user preferences
    # {'genre':[], 'director':[]}
    paths_dict = defaultdict(list)
    for path in movie_paths:
        paths_dict[path[3]].append(path)
    
    # calculate the score for the movie by summing the weights of the paths
    path_scores = 0
    for pref_type, paths in paths_dict.items():
        # find the path with the highest weight in each preference type
        strongest_path = max(paths, key=lambda x: x[2])
        path_length, path_weight = strongest_path[1], strongest_path[2]
        
        # calculate the preference weight based on the preference type
        pref_weight = 1.0
        if pref_type == 'director':
            pref_weight = 1.5
        elif pref_type == 'star':
            pref_weight = 1.2
        elif pref_type == 'genre':
            pref_weight = 0.8
        else:
            pref_weight = 1.0
        
        # length penalty
        length_penalty = 1 - (path_length / 10)
        
        # calculate the score for the path
        path_score = path_weight * pref_weight * length_penalty * 20
        path_scores += path_score
    
    final_score = base_score + path_scores
    return final_score


def recommendation(csv_path, preferences):
    df = load_data(csv_path)

    G = build_graph(df)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # visualize_subgraph(G, 'Anne Hathaway', depth=2)

    all_related_movies = find_all_related_movies(G, preferences)
    print(f"Scores: {calculate_movie_scores(G, 'Inception', all_related_movies)}")


if __name__ == "__main__":
    preferences = {'director': ['Christopher Nolan'], 'genre': ['Sci-Fi']}
    recommendation('./IMDB_Top_250_Movies.csv', preferences)
