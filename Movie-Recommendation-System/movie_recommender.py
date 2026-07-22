import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

def download_and_extract_dataset():
    """Downloads the MovieLens dataset if it doesn't already exist."""
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = "ml-latest-small.zip"
    extract_dir = "ml-latest-small"

    if not os.path.exists(extract_dir):
        print(f"Downloading MovieLens dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_path) # Clean up zip file
            print("Dataset downloaded and extracted successfully.")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            print("Please manually download the ml-latest-small dataset from GroupLens.")
            exit()
    else:
        print("Dataset already exists locally.")

def recommend_movies(predictions_df, user_id, movies_df, original_ratings_df, num_recommendations=5):
    """
    Recommends movies for a specific user based on SVD predictions.
    """
    # Get and sort the user's predictions
    user_row_number = user_id - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information
    user_data = original_ratings_df[original_ratings_df.userId == (user_id)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False))

    print(f"\nUser {user_id} has already rated {user_full.shape[0]} movies.")
    print("Top 5 movies highly rated by this user:")
    print(user_full[['title', 'rating']].head(5).to_string(index=False))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet
    unseen_movies = movies_df[~movies_df['movieId'].isin(user_full['movieId'])]
    
    # Merge unseen movies with their predicted ratings
    recommendations = (unseen_movies.merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns={user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
         iloc[:num_recommendations, :-1]
                      )

    print(f"\nTop {num_recommendations} Movie Recommendations for User {user_id}:")
    print(recommendations[['title', 'genres']].to_string(index=False))


def main():
    print("--------------------------------------------------")
    print("Movie Recommendation System (Collaborative Filtering)")
    print("--------------------------------------------------")

    download_and_extract_dataset()

    # 1. Load Data
    print("\nLoading data into Pandas DataFrames...")
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    
    print(f"Total Movies: {len(movies)}")
    print(f"Total Ratings: {len(ratings)}")

    # 2. Preprocess Data to create User-Item Matrix
    print("\nBuilding User-Item Utility Matrix...")
    # Rows = Users, Columns = MovieIDs, Values = Ratings
    utility_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    print(f"Utility Matrix shape: {utility_matrix.shape}")

    # 3. Apply Singular Value Decomposition (SVD)
    print("\nApplying Truncated SVD for Matrix Factorization...")
    # We use n_components=50 latent features
    svd = TruncatedSVD(n_components=50, random_state=42)
    matrix_factorized = svd.fit_transform(utility_matrix)
    
    print("Reconstructing predicted ratings matrix...")
    # Reconstruct the matrix by dot product of transformed matrix and components
    predicted_ratings = np.dot(matrix_factorized, svd.components_)
    
    # Convert reconstructed matrix back to a DataFrame
    preds_df = pd.DataFrame(predicted_ratings, columns=utility_matrix.columns)

    # 4. Generate Recommendations
    target_user_id = 10
    recommend_movies(preds_df, target_user_id, movies, ratings, num_recommendations=5)

if __name__ == "__main__":
    main()
