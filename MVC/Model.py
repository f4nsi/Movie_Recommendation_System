import networkx as nx
from collections import defaultdict


class MovieRecommendationModel:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, df):
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

        self.graph = G
        return G

    def find_related_movies(self, start_node, depth=2, pref_type=None):
        ''''
        BFS to find related movie paths. Used after the graph is generated.

        Parameters:
        - start_node: the starting node
        - depth: the depth of BFS
        - pref_type: the type of preference

        Returns:
        - related_movies: a list of related movies
        '''
        G = self.graph
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

    def find_all_related_movies(self, preferences):
        '''
        Find all related movies based on the user input.

        Parameters:
        - preferences: a dictionary of user preferences
            {'movie': ['movie_name'], 'director': ['director_name']}

        Returns:
        - related_movies: a list of related movies
        '''
        G = self.graph
        all_related_movies = []
        for pref_type, pref_values in preferences.items():
            for preference in pref_values:
                # find related movies for each preference
                if preference not in G.nodes:
                    print(f"Node '{preference}' not found in the graph.")
                    continue
                movies = self.find_related_movies(preference, depth=2, pref_type=pref_type)
                if movies:
                    all_related_movies.extend(movies)
                else:
                    print(f"No related movies found for '{preference}'.")

        return all_related_movies

    def calculate_movie_scores(self, movie_name, all_related_movies):
        '''
        Calculate movie scores based on user preferences.
        The score is calculated based on the path taken to reach the movie.
        
        final score = base_score + path_scores
        where path_scores = sum(path_weight * pref_weight * length_penalty * 20)
        '''
        G = self.graph
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

    def merge_sort_movies(self, movie_score_list):
        """
        Sort movies by their scores using merge sort algorithm.
        
        Parameters:
        - movie_score_list: a list of (movie, score) tuples
        
        Returns:
        - sorted list of (movie, score) tuples
        """
        if len(movie_score_list) <= 1:
            return movie_score_list
        mid = len(movie_score_list) // 2
        left = self.merge_sort_movies(movie_score_list[:mid])
        right = self.merge_sort_movies(movie_score_list[mid:])
        return self._merge(left, right)

    def _merge(self, left, right):
        """Helper method for merge sort"""
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

    def select_top_n_with_diversity(self, sorted_movies, n=5, penalty=0.5, coverage_weight=0.3):
        """
        Select top-N movies with diversity and novelty.
        
        Parameters:
        - sorted_movies: a list of (movie, score) tuples
        - n: number of movies to select
        - penalty: penalty for overlapping attributes
        - coverage_weight: weight for novel attributes
        
        Returns:
        - list of selected (movie, score) tuples
        """
        G = self.graph
        selected = []
        seen = {"genres": set(), "stars": set(), "directors": set()}
        # Iterate through the sorted movies and select top N with diversity
        for movie, score in sorted_movies:
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

            print(f"\nEvaluating movie: {movie}")
            print(f"  Base Score: {score:.2f}")
            print(f"  Genre Overlap: {genre_overlap}, Star Overlap: {star_overlap}, Director Overlap: {director_overlap}")
            print(f"  Diversity Penalty: {overlap_penalty:.2f}, Novelty Boost: {novelty_boost:.2f}, Adjusted Score: {adjusted_score:.2f}")

            if adjusted_score >= 0:
                selected.append((movie, adjusted_score))
                seen["genres"].update(new_genres)
                seen["stars"].update(new_stars)
                seen["directors"].update(new_directors)

            if len(selected) == n:
                break

        return selected

    def get_movie_details(self, movie_name):
        """
        Get detailed information about a movie.
        
        Parameters:
        - movie_name: name of the movie
        
        Returns:
        - dictionary with movie details
        """
        G = self.graph
        if movie_name not in G.nodes:
            return None
            
        details = {
            "title": movie_name,
            "director": [d for d in G.predecessors(movie_name) if G.nodes[d]['type'] == 'director'],
            "year": G.nodes[movie_name]['year'],
            "genres": [g for g in G.predecessors(movie_name) if G.nodes[g]['type'] == 'genre'],
            "stars": [s for s in G.predecessors(movie_name) if G.nodes[s]['type'] == 'star'],
            "rating": G.nodes[movie_name]['rating'],
            "duration": G.nodes[movie_name]['duration']
        }
        
        return details