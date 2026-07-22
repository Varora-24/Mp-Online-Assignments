# Movie Recommendation System (Collaborative Filtering)

## Objective
To build a Collaborative Filtering recommendation engine capable of predicting a user's movie preferences and suggesting highly relevant, unseen movies.

## Dataset
The project utilizes the **MovieLens (ml-latest-small)** dataset provided by GroupLens Research.
- **Movies**: ~9,700 movies with genres.
- **Ratings**: ~100,000 ratings across 600 users.
- The Python script automatically downloads and extracts this dataset when executed.

## Libraries Used
- **Pandas & NumPy**: For efficient data manipulation, merging, and constructing the User-Item utility matrix.
- **Scikit-Learn (`TruncatedSVD`)**: Used to perform Singular Value Decomposition (SVD) for Matrix Factorization.

## Methodology
1. **Utility Matrix Construction**: 
   - A pivot table is created from the ratings dataset where rows represent `userId` and columns represent `movieId`.
   - Missing ratings (movies a user hasn't seen) are imputed with `0`.
2. **Matrix Factorization (SVD)**:
   - Collaborative Filtering often relies on Matrix Factorization to discover latent (hidden) features in user preferences and movie attributes.
   - `TruncatedSVD` is applied to reduce the matrix down to 50 latent components.
   - By calculating the dot product of the factorized matrix and the components, we reconstruct a dense matrix where the previously missing `0` values are replaced with **predicted ratings**.
3. **Recommendation Engine**:
   - For a specific user, the engine identifies the movies they have highly rated to understand their preferences.
   - It filters out all movies the user has already seen.
   - It sorts the remaining unseen movies by their predicted rating and outputs the top 5 highest-predicted recommendations.

## How to Run
Ensure you have Pandas and Scikit-Learn installed:
```bash
pip install pandas numpy scikit-learn
```

Run the recommendation script:
```bash
python movie_recommender.py
```
*(The script will automatically download the required dataset, perform SVD, and print out 5 targeted movie recommendations for User #10).*
