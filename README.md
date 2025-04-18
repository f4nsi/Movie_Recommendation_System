# Movie Recommendation System

A graph-based movie recommendation system that leverages complex relationships between movies, directors, actors, and genres to provide personalized and diverse movie recommendations.

## Features

- **Graph-Based Approach**: Models the movie domain as a weighted directed graph to capture multidimensional relationships
- **Personalized Recommendations**: Generates recommendations based on user preferences for movies, directors, genres, or actors
- **Series Detection**: Automatically identifies and prioritizes movies in the same series or franchise
- **Diversity-Aware Selection**: Ensures recommendation diversity using a novel penalty and novelty boost mechanism
- **Visualization**: Provides intuitive visualizations of recommendation paths and relationships

## Dataset

This system uses the IMDb Top 250 Movies dataset, which contains information about highly-rated films including titles, directors, stars, genres, ratings, and release years.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip for package installation

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/f4nsi/Movie_Recommendation_System.git
   cd Movie_Recommendation_System
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```python
from graph import recommendation

# Example 1: Get recommendations based on favorite movies
preferences = {'movie': ['The Lord of the Rings: The Fellowship of the Ring']}
recommendation('./IMDB_Top_250_Movies.csv', preferences)

# Example 2: Get recommendations based on multiple preference types
preferences = {
    'director': ['Christopher Nolan', 'Quentin Tarantino'],
    'genre': ['Drama', 'Thriller'],
    'star': ['Leonardo DiCaprio']
}
recommendation('./IMDB_Top_250_Movies.csv', preferences)
```

### Visualization

The system automatically generates visualizations to explain the recommendation paths:

1. **Overview Visualization**: Shows the relationship between user preferences and recommendations
2. **Individual Movie Visualizations**: Explains why each specific movie was recommended

## How It Works

1. **Graph Construction**: Builds a directed weighted graph from the movie dataset
2. **Path Finding**: Uses Breadth-First Search to find connections between user preferences and potential recommendations
3. **Scoring**: Calculates scores based on movie ratings, path weights, and preference types
4. **Series Matching**: Identifies and prioritizes movies in the same series (when applicable)
5. **Diversity Enhancement**: Applies a greedy selection algorithm that balances score with diversity

## Project Structure

- `graph.py`: Core implementation of the recommendation algorithm
- `View.py`: Visualization functionality for recommendation paths
- `IMDB_Top_250_Movies.csv`: Dataset containing movie information
- `requirements.txt`: List of dependencies needed to run the project

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IMDb for the movie dataset
- NetworkX library for graph implementation
- Matplotlib for visualization capabilities
