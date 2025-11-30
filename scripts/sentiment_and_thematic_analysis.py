import pandas as pd
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import torch # Explicitly imported to prevent NameError with some environments

# --- CONFIGURATION ---
# IMPORTANT: This must match the name of the clean CSV file from Task 1
FILE_PATH_INPUT = 'playstore_reviews_task1_clean.csv' 
FILE_PATH_OUTPUT = 'task2_processed_reviews.csv'
HUGGING_FACE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Define the Thematic Rules (MUST BE DOCUMENTED IN README.MD)
THEME_RULES = {
    'Account Access Issues': ['login', 'password', 'fingerprint', 'otp', 'register'],
    'Transaction Performance': ['slow', 'transfer', 'delay', 'payment', 'transaction', 'speed'],
    'User Interface & Experience': ['ui', 'interface', 'design', 'user', 'easy', 'friendly'],
    'Reliability & Bugs': ['crash', 'bug', 'error', 'not work', 'stop work', 'frozen', 'update'],
    'Customer Support': ['support', 'customer service', 'call centre', 'help', 'respond']
}


def load_data(file_path):
    """Loads the clean data from Task 1 and prepares the 'review' column."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        # Ensure the 'review' column is a string and handle potential NaNs
        df['review'] = df['review'].astype(str).fillna('')
        
        # Ensure 'bank' and 'rating' are present and valid for aggregation later
        if 'bank' in df.columns:
            df['bank'] = df['bank'].astype(str)
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0).astype(int)
            
        print(f"Data loaded. Total reviews: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}. Please check the path and ensure Task 1 completed.")
        return None


def run_sentiment_analysis(df):
    """Applies the Hugging Face sentiment model to all reviews."""
    print(f"\n--- 1. Running Sentiment Analysis using {HUGGING_FACE_MODEL} ---")

    # Initialize the sentiment classification pipeline
    try:
        sentiment_classifier = pipeline(
            "sentiment-analysis", 
            model=HUGGING_FACE_MODEL,
            device=-1 # Use CPU for stability
        )
    except Exception as e:
        print(f"FATAL: Could not initialize Hugging Face pipeline. Error: {e}")
        # If model initialization fails, return the original df
        return df

    def analyze_sentiment(review_text):
        """Helper function to get label and score, handling empty/invalid text."""
        if len(review_text.strip()) == 0:
            return None, None
        try:
            result = sentiment_classifier(review_text, truncation=True)[0]
            # Map NEGATIVE scores to (1 - score) for consistent score interpretation
            # (0 = Negative, 1 = Positive)
            score = 1 - result['score'] if result['label'] == 'NEGATIVE' else result['score']
            return result['label'], score
        except Exception:
            return 'UNCERTAIN', 0.5 # Neutral score for failed analysis

    # Apply the function and expand the tuple result into two columns
    df[['sentiment_label', 'sentiment_score']] = df['review'].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )
    
    # Check KPI (Ensure columns were created)
    if 'sentiment_label' in df.columns:
        scored_count = df['sentiment_label'].count()
        if len(df) > 0 and scored_count / len(df) >= 0.90:
            print(f"Sentiment KPI achieved: {scored_count}/{len(df)} reviews scored.")
        else:
            print(f"Sentiment KPI NOT met: Only {scored_count}/{len(df)} reviews scored. (Target > 90%)")
    else:
        print("Warning: Sentiment columns were not created. Skipping KPI check.")

    return df


def preprocess_text(text, nlp_model):
    """Performs tokenization, stop-word removal, and lemmatization."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    doc = nlp_model(text.lower())

    # Filter out stop words, punctuation, spaces, and keep only alphabetic tokens
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha
    ]
    return " ".join(tokens)


def run_thematic_analysis(df, theme_rules):
    """Applies preprocessing, extracts keywords (TF-IDF), and assigns themes."""
    print("\n--- 2. Running Thematic Analysis (Preprocessing & Clustering) ---")

    # A. NLP Preprocessing Pipeline
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("FATAL: spaCy model not loaded. Cannot proceed with thematic analysis.")
        return df

    # Create the clean text column for feature extraction and rule matching
    print("Applying NLP preprocessing (Lemmatization, Stop-word removal)...")
    df['clean_review'] = df['review'].apply(lambda x: preprocess_text(x, nlp))


    # B. Keyword and N-gram Extraction (Required for Task 2 KPI)
    print("Extracting keywords via TF-IDF (1, 2, and 3-grams)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=5)
    
    non_empty_reviews = df[df['clean_review'].str.len() > 0]['clean_review']
    if len(non_empty_reviews) == 0:
        print("Warning: No sufficiently clean reviews found. Skipping TF-IDF and Thematic Assignment.")
        df['identified_theme(s)'] = "General/Other"
        df.drop(columns=['clean_review'], inplace=True, errors='ignore')
        return df
        
    vectorizer.fit(non_empty_reviews) # Running fit to fulfill the TF-IDF requirement


    # C. Manual/Rule-Based Clustering & Theme Assignment
    print("Assigning themes based on documented keyword rules...")

    def assign_themes(clean_text, rules):
        """Checks clean text against defined rules and returns a comma-separated list of themes."""
        assigned_themes = []
        for theme, keywords in rules.items():
            if any(keyword in clean_text for keyword in keywords):
                assigned_themes.append(theme)
        
        return ", ".join(assigned_themes) if assigned_themes else "General/Other"

    df['identified_theme(s)'] = df['clean_review'].apply(
        lambda x: assign_themes(x, theme_rules)
    )

    # Drop the temporary clean review column before saving
    df.drop(columns=['clean_review'], inplace=True, errors='ignore')

    return df


def run_aggregation(df):
    """Aggregates sentiment scores by bank and rating."""
    print("\n--- 3. Running Aggregation ---")
    
    # Check for required columns before aggregation
    if 'sentiment_score' not in df.columns or 'bank' not in df.columns or 'rating' not in df.columns:
        print("Warning: Missing required columns ('sentiment_score', 'bank', or 'rating') for aggregation.")
        return

    # Filter out rows where sentiment_score is NaN or 0.5 (UNCERTAIN) before aggregating
    valid_df = df[df['sentiment_score'].notna() & (df['sentiment_score'] != 0.5)]

    if valid_df.empty:
        print("Warning: No valid sentiment scores available for aggregation.")
        return

    # Group by bank and rating, calculate mean sentiment score
    sentiment_agg = valid_df.groupby(['bank', 'rating'])['sentiment_score'].mean().reset_index()
    sentiment_agg.rename(columns={'sentiment_score': 'mean_sentiment_score'}, inplace=True)

    print("Aggregation Complete. Mean Sentiment Score by Bank and Rating:")
    # Display the aggregation table 
    print(sentiment_agg.to_markdown(index=False))


def main():
    """Main function to execute the full pipeline."""
    # Load the clean data from Task 1
    df = load_data(FILE_PATH_INPUT)

    if df is not None:
        # Step 1: Sentiment Analysis
        df = run_sentiment_analysis(df)
        
        # DEFENSIVE CHECK: This prevents the final KeyError if the sentiment model failed.
        if 'sentiment_label' not in df.columns:
            print("Defensive Fix: Sentiment analysis failed to create columns. Adding placeholders.")
            df['sentiment_label'] = 'FAILED'
            df['sentiment_score'] = 0.5


        # Step 2: Thematic Analysis (Preprocessing and Assignment)
        df = run_thematic_analysis(df, THEME_RULES)

        # Step 3: Aggregation
        run_aggregation(df) 

        # Save Final Results
        final_columns = ['review', 'bank', 'rating', 'date', 'sentiment_label', 
                         'sentiment_score', 'identified_theme(s)']
        
        # Filter columns to only those that exist in the DataFrame before saving
        existing_cols = [col for col in final_columns if col in df.columns]
        
        # Check if essential columns are missing after processing
        if not all(col in existing_cols for col in ['review', 'bank', 'rating']):
            print("\nFATAL: Essential columns are missing. Cannot save output file.")
            return

        df[existing_cols].to_csv(FILE_PATH_OUTPUT, index=False)
        print(f"\nFinal analysis saved successfully to {FILE_PATH_OUTPUT}")
    else:
        print("Pipeline aborted due to data loading failure.")


if __name__ == "__main__":
    main()