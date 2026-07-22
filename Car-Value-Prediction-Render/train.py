import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    print("Training dummy Car Value Prediction model...")
    
    # 1. Create a tiny mock dataset
    # Features: [Year, Mileage, Engine Size (L)]
    # Target: Price ($)
    data = {
        'year': [2010, 2012, 2015, 2018, 2020, 2022],
        'mileage': [120000, 95000, 70000, 45000, 20000, 5000],
        'engine_size': [1.5, 2.0, 1.8, 2.5, 2.0, 3.0],
        'price': [5000, 7500, 12000, 18000, 25000, 35000]
    }
    df = pd.DataFrame(data)
    
    X = df[['year', 'mileage', 'engine_size']]
    y = df['price']
    
    # 2. Train a simple Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    
    print("Model training complete. R^2 Score:", model.score(X, y))
    
    # 3. Save the model to a pickle file
    with open('car_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("Model saved as 'car_model.pkl'.")

if __name__ == "__main__":
    main()
