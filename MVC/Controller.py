import pandas as pd
from MVC.Model import MovieRecommendationModel
from View import visualize_subgraph


class MovieRecommendationController:
    def __init__(self):
        self.model = MovieRecommendationModel()
    
    def load_data(self, csv_path):
        """
        Load movie data from CSV file.
        
        Parameters:
        - csv_path: path to the CSV file
        
        Returns:
        - DataFrame with movie data
        """
        df = pd.read_csv(csv_path)
        return df
    
    def initialize_graph(self, csv_path):
        """
        Load data and build the movie graph.
        
        Parameters:
        - csv_path: path to the CSV file
        
        Returns:
        - Number of nodes and edges in the graph
        """
        df = self.load_data(csv_path)
        G = self.model.build_graph(df)
        return G.number_of_nodes(), G.number_of_edges()
    
    def visualize_node_connections(self, node, depth=2):
        """
        Visualize the subgraph around a specific node.
        
        Parameters:
        - node: center node for visualization
        - depth: depth of exploration
        """
        G = self.model.graph
        if node not in G.nodes:
            print(f"Node '{node}' not found in the graph.")
            return
        
        visualize_subgraph(G, node, depth)
    
    def get_recommendations(self, preferences, top_n=5):
        """
        Get movie recommendations based on user preferences.
        
        Parameters:
        - preferences: dictionary of preferences
        - top_n: number of recommendations to return
        
        Returns:
        - List of recommended movies with details
        """
        # Find related movies
        all_related_movies = self.model.find_all_related_movies(preferences)
        
        # Get unique movie titles
        all_titles = set()
        for m in all_related_movies:
            all_titles.add(m[0])
        
        # Calculate scores for each movie
        scores = []
        for title in all_titles:
            score = self.model.calculate_movie_scores(title, all_related_movies)
            scores.append((title, score))
        
        # Sort movies by score
        sorted_movies = self.model.merge_sort_movies(scores)
        
        # Select diverse recommendations
        diverse_recommendations = self.model.select_top_n_with_diversity(sorted_movies, n=top_n)
        
        # Get detailed information for recommended movies
        recommendations = []
        for movie, score in diverse_recommendations:
            details = self.model.get_movie_details(movie)
            recommendations.append({
                "title": movie,
                "score": score,
                "details": details
            })
        
        return recommendations
    
    def display_recommendations(self, recommendations):
        """
        Display formatted recommendations to the user.
        
        Parameters:
        - recommendations: list of recommended movies with details
        """
        print(f"\nTop {len(recommendations)} Personalized Movie Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            movie = rec["title"]
            score = rec["score"]
            details = rec["details"]
            
            print(f"{i}. {movie} (Score: {score:.2f}) ")
            print(f"    Director: {', '.join(details['director'])}, "
                  f"Release Year: {details['year']}, ",
                  f"Genres: {', '.join(details['genres'])}, ",
                  f"Rating: {details['rating']}, ",
                  f"Duration: {details['duration']} mins")
    
    def run_recommendation(self, csv_path, preferences, top_n=5):
        """
        Run the complete recommendation process.
        
        Parameters:
        - csv_path: path to the CSV file
        - preferences: dictionary of preferences
        - top_n: number of recommendations to return
        """
        # Initialize graph
        nodes, edges = self.initialize_graph(csv_path)
        print(f"Graph has {nodes} nodes and {edges} edges")
        
        # Get recommendations
        recommendations = self.get_recommendations(preferences, top_n)
        
        # Display recommendations
        self.display_recommendations(recommendations)


# Example usage
if __name__ == "__main__":
    controller = MovieRecommendationController()
    preferences = {'movie': ['Star Wars: Episode IV - A New Hope', 'Alien'], 'genre': ['Sci-Fi']}
    controller.run_recommendation('./IMDB_Top_250_Movies.csv', preferences)