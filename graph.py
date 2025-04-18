import pandas as pd
import networkx as nx
from View import create_recommendation_report
from collections import defaultdict
from difflib import get_close_matches


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
        year = row['Release_Year']
        
        # Create decade string (e.g., "1990s")
        decade = f"{(year // 10) * 10}s"

        # add movie nodes and rating
        G.add_node(movie_title, type='movie',
                   rating=row['Rating'],
                   year=year, duration=row['Duration'])

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

        # add decade nodes and edges
        if not G.has_node(decade):
            G.add_node(decade, type='decade')
        G.add_edge(decade, movie_title, weight=0.6)  # decade -> movie
        G.add_edge(movie_title, decade, weight=0.2)  # movie -> decade

    return G


# fuzzy-match user input to movie nodes
def match_movie_node(G, query, cutoff=0.4):
    movies = [n for n, d in G.nodes(data=True) if d.get('type') == 'movie']
    matches = get_close_matches(query, movies, n=1, cutoff=cutoff)
    return matches[0] if matches else None


# 3. use BFS to find related movies based on a starting node
def find_related_movies(G, start_node, depth=3, pref_type=None):
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
        # boost direct movie matches
        if pref_type == 'movie':
            pref_weight = 2.0
        elif pref_type == 'director':
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


# 6. Merge sort for scores
def merge_sort_movies(movie_score_list):
    if len(movie_score_list) <= 1:
        return movie_score_list
    mid = len(movie_score_list) // 2
    left = merge_sort_movies(movie_score_list[:mid])
    right = merge_sort_movies(movie_score_list[mid:])
    return merge(left, right)


def merge(left, right):
    sorted_list = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][1] >= right[j][1]:
            sorted_list.append(left[i])
            i += 1
        else:
            sorted_list.append(right[j])
            j += 1
    sorted_list.extend(left[i:])
    sorted_list.extend(right[j:])
    return sorted_list


# 7. Top-N selection with diversity and novelty
def select_top_n_with_diversity(G, sorted_movies, n=5, penalty=1.0, coverage_weight=0.5):
    selected = []
    seen = {"genres": set(), "stars": set(), "directors": set()}
    remaining = sorted_movies.copy()
    
    while len(selected) < n and remaining:
        # Evaluate all remaining movies with current diversity considerations
        candidates = []
        for movie, score in remaining:
            genre_overlap = sum(g in seen["genres"] for g in G.predecessors(movie) if G.nodes[g]['type'] == 'genre')
            star_overlap = sum(s in seen["stars"] for s in G.predecessors(movie) if G.nodes[s]['type'] == 'star')
            director_overlap = sum(d in seen["directors"] for d in G.predecessors(movie) if G.nodes[d]['type'] == 'director')

            # Identify new coverage
            new_genres = [g for g in G.predecessors(movie) if G.nodes[g]['type'] == 'genre' and g not in seen['genres']]
            new_stars = [s for s in G.predecessors(movie) if G.nodes[s]['type'] == 'star' and s not in seen['stars']]
            new_directors = [d for d in G.predecessors(movie) if G.nodes[d]['type'] == 'director' and d not in seen['directors']]

            novelty_boost = coverage_weight * (len(new_genres) + len(new_stars) + len(new_directors))
            overlap_penalty = penalty * (genre_overlap + star_overlap + director_overlap)
            adjusted_score = score - overlap_penalty + novelty_boost
            
            candidates.append((movie, adjusted_score, new_genres, new_stars, new_directors))
        
        # Find the movie with the highest adjusted score
        if not candidates:
            break
            
        best_movie = max(candidates, key=lambda x: x[1])
        movie, adj_score, new_genres, new_stars, new_directors = best_movie
        
        print(f"\nEvaluating movie: {movie}")
        print(f"  Base Score: {score:.2f}")
        print(f"  Genre Overlap: {genre_overlap}, Star Overlap: {star_overlap}, Director Overlap: {director_overlap}")
        print(f"  Diversity Penalty: {overlap_penalty:.2f}, Novelty Boost: {novelty_boost:.2f}, Adjusted Score: {adj_score:.2f}")
        
        # Add to selected and update seen sets
        selected.append((movie, adj_score))
        seen["genres"].update(new_genres)
        seen["stars"].update(new_stars)
        seen["directors"].update(new_directors)
        
        # Remove the selected movie from remaining
        remaining = [(m, s) for m, s in remaining if m != movie]
    
    return selected


# 8. Main function to run the recommendation system
def recommendation(csv_path, preferences, top_n=5):
    df = load_data(csv_path)

    G = build_graph(df)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # fuzzy-match movie preferences
    if 'movie' in preferences:
        real_movies = []
        for q in preferences['movie']:
            match = match_movie_node(G, q)
            if match:
                real_movies.append(match)
            else:
                print(f"No close match for '{q}'")
        preferences['movie'] = real_movies

    all_related_movies = find_all_related_movies(G, preferences)
    # print(f"Scores: {calculate_movie_scores(G, 'Inception', all_related_movies)}")

    # score all titles
    all_titles = set()
    for m in all_related_movies:
        all_titles.add(m[0])

    scores = []
    for title in all_titles:
        score = calculate_movie_scores(G, title, all_related_movies)
        scores.append((title, score))

    sorted_movies = merge_sort_movies(scores)

    # series boosting if user picked a single movie
    final_recs = []
    if 'movie' in preferences and preferences['movie']:
        start = preferences['movie'][0]
        series_prefix = start.split(':')[0].strip()
        # other in same series
        series = [(m, s) for m, s in sorted_movies if m != start and series_prefix in m]
        series_sorted = merge_sort_movies(series)
        # remaining
        remaining = [(m, s) for m, s in sorted_movies if m != start and series_prefix not in m]
        diverse_rest = select_top_n_with_diversity(G, remaining, n=top_n - len(series_sorted))
        final_recs = series_sorted + diverse_rest
    else:
        final_recs = select_top_n_with_diversity(G, sorted_movies, n=top_n)

    print(f"\nTop {top_n} Personalized Movie Recommendations:")
    for i, (movie, score) in enumerate(final_recs[:top_n], 1):
        print(f"{i}. {movie} (Score: {score:.2f}) ")
        print(f"    Director: {', '.join([d for d in G.predecessors(movie) if G.nodes[d]['type'] == 'director'])}, "
              f"Release Year: {G.nodes[movie]['year']}, ",
              f"Genres: {', '.join([g for g in G.predecessors(movie) if G.nodes[g]['type'] == 'genre'])}, ",
              f"Rating: {G.nodes[movie]['rating']}, ",
              f"Duration: {G.nodes[movie]['duration']} mins")
    create_recommendation_report(G, preferences, final_recs[:top_n], all_related_movies)


if __name__ == "__main__":
    preferences = {'genre': ['Action']}
    recommendation('./IMDB_Top_250_Movies.csv', preferences)
