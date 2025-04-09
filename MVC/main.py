from MVC.Controller import MovieRecommendationController


def main():
    # Create controller instance
    controller = MovieRecommendationController()
    
    # Define user preferences
    preferences = {
        'director': ['Christopher Nolan'],
        'genre': ['Sci-Fi']
    }
    
    # Run recommendation process
    controller.run_recommendation('./IMDB_Top_250_Movies.csv', preferences, top_n=5)
    
    # Optionally visualize some relationships
    print("\nVisualizing graph for Christopher Nolan...")
    controller.visualize_node_connections('Christopher Nolan', depth=2)


if __name__ == "__main__":
    main()
