
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
import io

st.set_page_config(page_title="Lifestyle, Nutrition & Fitness Analysis", layout="wide")

# --- Helper functions ---
@st.cache_data
def load_data(final_path, meal_path):
    df = pd.read_csv(final_path)
    meals = pd.read_csv(meal_path)
    return df, meals

def clean_data(df):
    df = df.copy()
    # Basic sanitization
    df = df.drop_duplicates().reset_index(drop=True)
    # Simple numeric imputation for numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # Simple encoding for known categorical fields if present
    if 'Experience_Level' in df.columns:
        if df['Experience_Level'].dtype == object:
            df['Experience_Level_code'] = pd.Categorical(df['Experience_Level']).codes
        else:
            df['Experience_Level_code'] = df['Experience_Level']
    return df

def feature_engineering(df, carb_col='carbs_g', prot_col='proteins_g', fat_col='fats_g'):
    df = df.copy()
    if all(c in df.columns for c in [carb_col, prot_col, fat_col]):
        df['total_macro_g'] = df[[carb_col, prot_col, fat_col]].sum(axis=1).replace(0, np.nan)
        df['pct_carbs'] = df[carb_col] / df['total_macro_g']
        df['pct_protein'] = df[prot_col] / df['total_macro_g']
        df['pct_fat'] = df[fat_col] / df['total_macro_g']
    # BMI computation if missing and weight/height present
    if 'BMI' not in df.columns and ('Weight (kg)' in df.columns and 'Height (m)' in df.columns):
        df['BMI'] = df['Weight (kg)'] / (df['Height (m)']**2)
    return df

def run_clustering(df, features, k=3):
    sub = df[features].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(sub.values)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    sub['cluster'] = labels
    # PCA for plotting
    pca = PCA(2)
    comps = pca.fit_transform(X)
    sub['PC1'] = comps[:,0]; sub['PC2'] = comps[:,1]
    return sub, kmeans, scaler, pca

def recommend_meals(meals_df, user_cal, meal_cal_col='Calories', tol=0.25):
    if meal_cal_col not in meals_df.columns:
        # try to find a column with 'cal' in its name
        for c in meals_df.columns:
            if 'cal' in c.lower():
                meal_cal_col = c; break
    if meal_cal_col not in meals_df.columns:
        return pd.DataFrame()  # no calories column
    low = user_cal * (1 - tol)
    high = user_cal * (1 + tol)
    matches = meals_df[(meals_df[meal_cal_col] >= low) & (meals_df[meal_cal_col] <= high)].copy()
    return matches.sort_values(by=meal_cal_col).head(10)

def recommend_workout(bmi, exp):
    try:
        if np.isnan(bmi):
            return ["General fitness: mixed cardio & strength"]
    except:
        pass
    level = 'beginner'
    if isinstance(exp, str):
        s = exp.lower()
        if 'adv' in s or 'expert' in s:
            level = 'advanced'
        elif 'inter' in s:
            level = 'intermediate'
        else:
            level = 'beginner'
    else:
        try:
            if exp > 1: level='advanced'
            elif exp == 1: level='intermediate'
        except:
            level='beginner'
    if bmi < 18.5:
        return ["Hypertrophy 3x/week", "Compound lifts", "High protein meal plan"]
    elif bmi < 25:
        return ["HIIT 2x/week", "Strength 2x/week", "Mobility sessions"]
    else:
        if level == 'beginner':
            return ["Low-impact cardio (walking, cycling)", "Circuit training - low impact"]
        else:
            return ["Cardio steady-state 30-45 min", "Strength circuit", "Interval training"]

# --- UI ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Data Preview", "EDA", "Clustering", "Recommendations", "Deployment"])

# Paths to CSVs (these are expected to be uploaded to the app folder on Streamlit)
FINAL_PATH = "Final_data.csv"
MEAL_PATH = "meal_metadata.csv"

st.sidebar.markdown("**Data files**: Final_data.csv, meal_metadata.csv (place in app root)")

df, meals = load_data(FINAL_PATH, MEAL_PATH)
df_clean = clean_data(df)
df_fe = feature_engineering(df_clean)

if page == "Overview":
    st.title("Lifestyle, Nutrition & Fitness Analysis — Overview")
    st.markdown("This Streamlit app demonstrates the pipeline used in the hackathon: data cleaning, EDA, clustering, and recommendation engines for meals and workouts.")
    st.metric("Rows (raw data)", len(df))
    st.metric("Rows (cleaned)", len(df_clean))
    st.metric("Meals entries", len(meals))
    st.write("Select other pages to interact with analysis and recommendations.")

if page == "Data Preview":
    st.title("Data Preview")
    st.subheader("User Data (first 50 rows)")
    st.dataframe(df_fe.head(50))
    st.subheader("Meal Metadata (first 50 rows)")
    st.dataframe(meals.head(50))

if page == "EDA":
    st.title("Exploratory Data Analysis")
    st.subheader("BMI distribution")
    if 'BMI' in df_fe.columns:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(df_fe['BMI'].dropna(), bins=25)
        ax.set_xlabel("BMI"); ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("BMI not available in dataset.")

    st.subheader("Calories burned distribution (if available)")
    cal_col = None
    for c in df_fe.columns:
        if 'cal' in c.lower():
            cal_col = c; break
    if cal_col:
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.hist(df_fe[cal_col].dropna(), bins=25)
        ax2.set_xlabel(cal_col); ax2.set_ylabel("Count")
        st.pyplot(fig2)
    else:
        st.info("Calories burned column not detected.")

    st.subheader("Macro distribution by BMI bin (if macros exist)")
    if 'pct_protein' in df_fe.columns:
        bmi_bins = pd.cut(df_fe['BMI'], bins=[0,18.5,25,30,100], labels=['Under', 'Normal', 'Over', 'Obese'])
        df_fe['BMI_bin'] = bmi_bins
        agg = df_fe.groupby('BMI_bin')[['pct_carbs','pct_protein','pct_fat']].mean().reset_index()
        st.dataframe(agg)
        fig3, ax3 = plt.subplots(figsize=(7,3))
        agg.set_index('BMI_bin')[['pct_carbs','pct_protein','pct_fat']].plot(kind='bar', stacked=True, ax=ax3)
        ax3.set_ylabel("Proportion")
        st.pyplot(fig3)
    else:
        st.info("Macro percentage columns not found.")

if page == "Clustering":
    st.title("Fitness Behavior Segmentation (KMeans)")
    # choose features for clustering
    available = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    st.write("Numeric features detected:", available)
    default_features = []
    for f in ['Workout_Frequency','Avg_BPM','Calories_Burned','Experience_Level_code','BMI']:
        if f in available:
            default_features.append(f)
    features = st.multiselect("Features to use for clustering", options=available, default=default_features)
    k = st.slider("Number of clusters (k)", min_value=2, max_value=6, value=3)
    if st.button("Run clustering"):
        if len(features) < 2:
            st.warning("Select at least 2 numeric features for clustering")
        else:
            with st.spinner("Clustering..."):
                sub, kmeans, scaler, pca = run_clustering(df_fe, features, k=k)
            st.success("Clustering completed")
            st.write("Cluster counts:")
            st.dataframe(sub['cluster'].value_counts().sort_index().rename('count').to_frame())
            figc, axc = plt.subplots(figsize=(6,4))
            for cl in sorted(sub['cluster'].unique()):
                pts = sub[sub['cluster']==cl]
                axc.scatter(pts['PC1'], pts['PC2'], s=25, label=f"Cluster {cl}")
            axc.set_xlabel("PC1"); axc.set_ylabel("PC2"); axc.legend()
            st.pyplot(figc)
            # allow download labels appended to full dataframe
            df_out = df_fe.copy()
            # align index - sub has original index preserved
            df_out.loc[sub.index, 'cluster'] = sub['cluster']
            csv = df_out.to_csv(index=False).encode('utf-8')
            st.download_button("Download full dataset with cluster labels", data=csv, file_name="Final_with_clusters.csv", mime="text/csv")

if page == "Recommendations":
    st.title("Recommendations")
    st.subheader("Meal Recommendations")
    # pick a sample user by index
    idx = st.number_input("Select user row index", min_value=0, max_value=max(0, len(df_fe)-1), value=0, step=1)
    user_row = df_fe.iloc[idx]
    # find user calorie need
    cal_need = None
    for c in df_fe.columns:
        if 'total' in c.lower() and 'cal' in c.lower():
            cal_need = user_row[c]; break
    if cal_need is None:
        for c in df_fe.columns:
            if 'expected' in c.lower() or 'burn' in c.lower():
                cal_need = user_row[c]; break
    st.write("Estimated user calorie need:", cal_need)
    st.write("Meal metadata columns:", meals.columns.tolist())
    matches = recommend_meals(meals, cal_need) if cal_need is not None else pd.DataFrame()
    if not matches.empty:
        st.dataframe(matches.head(10))
        csvm = matches.to_csv(index=False).encode('utf-8')
        st.download_button("Download meal recommendations", data=csvm, file_name="meal_recommendations.csv", mime="text/csv")
    else:
        st.info("No meals matched for this user's calorie need or meal calories missing.")

    st.subheader("Workout Recommendations")
    user_bmi = user_row.get('BMI', np.nan)
    user_exp = user_row.get('Experience_Level', user_row.get('Experience_Level_code', None))
    st.write("User BMI:", user_bmi, "Experience:", user_exp)
    recs = recommend_workout(user_bmi, user_exp)
    st.write("Recommended workout program:")
    for r in recs:
        st.write("- ", r)

if page == "Deployment":
    st.title("Deployment & Notes")
    st.markdown("""
    **How to deploy**

    1. Place `app.py`, `Final_data.csv`, and `meal_metadata.csv` in the same repository root.
    2. Create `requirements.txt` with the required libraries.
    3. Push to GitHub and connect repo to Streamlit Cloud or run locally:
       `streamlit run app.py`
    """)
    st.markdown("**Files detected in app folder**")
    files = ["Final_data.csv", "meal_metadata.csv"]
    st.write(files)
    st.markdown("**Helpful tips**: inspect console logs on Streamlit Cloud build failure. Common causes: missing data files, missing requirements, notebook files uploaded instead of app.py.")

