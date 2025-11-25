
# Streamlit App for Lifestyle, Nutrition & Fitness Analysis

This folder contains a Streamlit application (app.py) that reproduces key steps from the hackathon notebook:
- data cleaning and preprocessing
- feature engineering and EDA
- KMeans clustering for user segmentation
- meal and workout recommendation examples

## Files
- app.py : Streamlit application
- requirements.txt : Python dependencies

## How to run locally
1. Ensure `Final_data.csv` and `meal_metadata.csv` are in the same folder (root).
2. Create a Python virtual environment and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes for Streamlit Cloud
- Push this folder to a GitHub repo and connect to Streamlit Cloud.
- Make sure `Final_data.csv` and `meal_metadata.csv` are included in the repo (or fetch from a public URL).
