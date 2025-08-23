# File Name: train_model.py
# This script cleans the raw data, trains multiple machine learning models,
# evaluates them, and saves the best-performing model for the app to use.

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings

# Import all the models we want to compare
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

def train_and_save_model(file1, file2):
    """
    Main function to run the entire data cleaning and model training pipeline.
    """
    # --- Data Cleaning Sub-Function ---
    def clean_flight_data(file_path):
        """
        Reads a raw Excel file, cleans it, and engineers features.
        """
        df = pd.read_excel(
            file_path, header=None, skiprows=1,
            names=['S.No', 'Flight Number', 'Date', 'From', 'To', 'Aircraft', 'Flight time',
                   'STD', 'ATD', 'STA', 'col10', 'ATA', 'col12', 'col13']
        )
        # Forward-fill flight numbers and drop empty rows
        df['Flight Number'] = df['Flight Number'].ffill()
        df.dropna(subset=['Date'], inplace=True)
        df.drop(columns=['S.No', 'col10', 'col12', 'col13'], inplace=True)

        # Standardize the 'Actual Time of Arrival' (ATA) column
        def clean_ata(time_str):
            if isinstance(time_str, str) and 'Landed' in time_str:
                time_part = time_str.replace('Landed', '').strip()
                try: return pd.to_datetime(time_part, format='%I:%M %p').strftime('%H:%M:%S')
                except (ValueError, TypeError): return None
            elif hasattr(time_str, 'strftime'): return time_str.strftime('%H:%M:%S')
            return time_str

        df['ATA_cleaned'] = df['ATA'].apply(clean_ata)
        
        # Convert date/time columns to the correct format for calculations
        df['STA'] = df['STA'].astype(str)
        df['ATA_cleaned'] = df['ATA_cleaned'].astype(str)
        df['Date'] = pd.to_datetime(df['Date']).dt.date.astype(str)
        df['STA_datetime'] = pd.to_datetime(df['Date'] + ' ' + df['STA'], errors='coerce')
        df['ATA_datetime'] = pd.to_datetime(df['Date'] + ' ' + df['ATA_cleaned'], errors='coerce')
        
        # Feature Engineering: Calculate delay and create target variable
        df['delay_minutes'] = (df['ATA_datetime'] - df['STA_datetime']).dt.total_seconds() / 60
        df['is_delayed'] = df['delay_minutes'].apply(lambda x: 1 if x > 15 else 0)
        df['day_of_week'] = df['STA_datetime'].dt.day_name()
        df['scheduled_hour'] = df['STA_datetime'].dt.hour
        
        # --- NEW: Extract Tail Number and clean Aircraft Model ---
        df['Tail Number'] = df['Aircraft'].str.extract(r'\((.*?)\)').fillna('Unknown')
        df['Aircraft Model'] = df['Aircraft'].str.split('(').str[0].str.strip()
        
        # Clean whitespace from location columns for better matching
        df['To'] = df['To'].str.strip()
        df['From'] = df['From'].str.strip()
        
        # Select the final columns needed for the model and app
        final_df = df[['Flight Number', 'Date', 'From', 'To', 'Aircraft Model', 'Tail Number', 'STA', 'day_of_week', 'scheduled_hour', 'is_delayed', 'delay_minutes']].copy()
        
        final_df.dropna(subset=['delay_minutes', 'To', 'Aircraft Model'], inplace=True)
        return final_df

    # 1. Clean and Combine Data from both Excel sheets
    print("Starting data cleaning process...")
    df1 = clean_flight_data(file1)
    df2 = clean_flight_data(file2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.to_csv('cleaned_flight_data.csv', index=False)
    print("✅ Data cleaned and saved to 'cleaned_flight_data.csv'")

    # 2. Define Features (X) and Target (y) for the model
    features = ['From', 'To', 'Aircraft Model', 'day_of_week', 'scheduled_hour']
    target = 'is_delayed'
    X = combined_df[features]
    y = combined_df[target]

    # 3. Split data into a training set and a final test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Set up a preprocessing pipeline to handle categorical data
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', drop='first'),
                       ['From', 'To', 'Aircraft Model', 'day_of_week'])],
        remainder='passthrough'
    )

    # 5. Define the list of models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # 6. Evaluate all models using 5-fold cross-validation to find the best one
    best_model_name = ""
    best_model_score = 0.0
    print("\n--- Comparing Models using 5-Fold Cross-Validation ---")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = cv_scores.mean()
        print(f"{name} - CV Accuracy: {mean_score:.4f}")
        if mean_score > best_model_score:
            best_model_score = mean_score
            best_model_name = name

    print(f"\n🏆 Best Model: {best_model_name} with score {best_model_score:.4f}")

    # 7. Train the winning model on the full training set and save it
    best_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', models[best_model_name])])
    best_model_pipeline.fit(X_train, y_train)
    print("\n--- Final Report for Best Model on Unseen Test Data ---")
    predictions = best_model_pipeline.predict(X_test)
    print(classification_report(y_test, predictions))
    joblib.dump(best_model_pipeline, 'flight_delay_model.joblib')
    print(f"✅ Successfully saved '{best_model_name}' model to 'flight_delay_model.joblib'")


if __name__ == '__main__':
    # Ensure your Excel files are in the same folder with these exact names
    train_and_save_model(file1='6-9flight.xlsx', file2='9-12flight.xlsx')
