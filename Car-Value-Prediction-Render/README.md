# End to End Render Deployment Project (Car Value Prediction)

## Objective
To build and deploy an end-to-end Machine Learning web application using Flask and host it on Render. The API takes car attributes (Year, Mileage, Engine Size) and returns a predicted resale price.

## Project Structure
- `train.py`: A script that trains a Scikit-Learn `LinearRegression` model on dummy car data and saves it as `car_model.pkl`.
- `app.py`: A Flask web server that loads the trained model and exposes a POST `/predict` endpoint.
- `requirements.txt`: Python dependencies (Flask, Scikit-Learn, Pandas, Gunicorn).
- `render.yaml`: Blueprint configuration used by Render to automate the deployment process.

## How to Test Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model (generates `car_model.pkl`):
   ```bash
   python train.py
   ```
3. Run the Flask server:
   ```bash
   python app.py
   ```
4. Open `http://localhost:5000` in your browser, or test the API endpoint using `curl`:
   ```bash
   curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"year\": 2018, \"mileage\": 45000, \"engine_size\": 2.5}"
   ```

## How to Deploy on Render
1. Push this repository to GitHub.
2. Log in to [Render](https://render.com/).
3. Click **New** > **Blueprint**.
4. Connect your GitHub repository and select the `render.yaml` file.
5. Render will automatically build the environment, run the `train.py` script to generate the model, and start the Gunicorn web server!
