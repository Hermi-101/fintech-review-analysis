import pandas as pd
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import logging
from tabulate import tabulate 

# --- Configuration and Logging Setup ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File Paths and Model Parameters
CONFIG = {
    'FILE_PATH_INPUT': 'playstore_reviews_task1_clean.csv',
    'FILE_PATH_OUTPUT': 'task2_processed_reviews.csv',
    'HUGGING_FACE_MODEL': "distilbert-base-uncased-finetuned-sst-2-english",
    'SPACY_MODEL': "en_core_web_sm",
    'THEME_RULES': {
        'Account Access Issues': ['login', 'password', 'fingerprint', 'otp', 'register'],
        'Transaction Performance': ['slow', 'transfer', 'delay', 'payment', 'transaction', 'speed'],
        'User Interface & Experience': ['ui', 'interface', 'design', 'user', 'easy', 'friendly'],
        'Reliability & Bugs': ['crash', 'bug', 'error', 'not work', 'stop work', 'frozen', 'update'],
        'Customer Support': ['support', 'customer service', 'call centre', 'help', 'respond']
    }
}


def load_data(file_path):
    """
    Loads the clean data from Task 1.
    
    Args:
        file_path (str): Path to the input CSV file.
    
    Returns:
        pd.DataFrame or None: Loaded DataFrame or None on failure.
    """
    logging.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        
        # Data Type Validation and Cleaning
        df['review'] = df['review'].astype(str).fillna('')
        if 'bank' in df.columns:
            df['bank'] = df['bank'].astype(str)
        if 'rating' in df.columns:
            # Coerce rating to integer, setting invalid/missing to 0
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0).astype(int)
            
        logging.info(f"Data loaded successfully. Total reviews: {len(df)}")
        return df
    except FileNotFoundError:
        logging.error(f"Data file not found at {file_path}. Ensure Task 1 was completed.")
        return None
    except Exception as e:
        logging.error(f"Error during data loading: {e}")
        return None


def run_sentiment_analysis(df, model_name):
    """
    Applies the Hugging Face sentiment model.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        model_name (str): Name of the Hugging Face model.
        
    Returns:
        pd.DataFrame: DataFrame with 'sentiment_label' and 'sentiment_score' columns added.
    """
    logging.info(f"--- Running Sentiment Analysis using {model_name} ---")

    try:
        # Initialize the sentiment classification pipeline (using CPU as device=-1 for stability)
        sentiment_classifier = pipeline(
            "sentiment-analysis", 
            model=model_name,
            device=-1 
        )
    except Exception as e:
        logging.critical(f"FATAL: Could not initialize Hugging Face pipeline. Error: {e}")
        # Add placeholder columns if initialization fails to prevent KeyError later
        df['sentiment_label'] = 'FAILED'
        df['sentiment_score'] = 0.5
        return df

    def analyze_sentiment(review_text):
        """Helper function to get label and score."""
        if len(review_text.strip()) == 0:
            return None, None
        try:
            result = sentiment_classifier(review_text, truncation=True)[0]
            # Score normalization: 0.0 (Negative) to 1.0 (Positive)
            score = 1 - result['score'] if result['label'] == 'NEGATIVE' else result['score']
            return result['label'], score
        except Exception as e:
            logging.debug(f"Sentiment analysis failed for one review: {e}")
            return 'UNCERTAIN', 0.5

    # Apply the function and expand the tuple result into two new columns
    df[['sentiment_label', 'sentiment_score']] = df['review'].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )
    
    # KPI Check
    if 'sentiment_label' in df.columns:
        scored_count = df['sentiment_label'].count()
        if len(df) > 0 and scored_count / len(df) >= 0.90:
            logging.info(f"Sentiment KPI achieved: {scored_count}/{len(df)} reviews scored.")
        else:
            logging.warning(f"Sentiment KPI NOT met: Only {scored_count}/{len(df)} reviews scored. (Target > 90%)")

    return df


def preprocess_text(text, nlp_model):
    """Performs tokenization, stop-word removal, and lemmatization using spaCy."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    doc = nlp_model(text.lower())

    # Filter for valid tokens and lemmatize
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha
    ]
    return " ".join(tokens)


def run_thematic_analysis(df, theme_rules, spacy_model_name):
    """
    Applies preprocessing, TF-IDF feature extraction, and theme assignment.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        theme_rules (dict): Dictionary of themes and their keywords.
        spacy_model_name (str): Name of the spaCy model.
        
    Returns:
        pd.DataFrame: DataFrame with 'identified_theme(s)' column added.
    """
    logging.info("--- Running Thematic Analysis (Preprocessing & Clustering) ---")

    # A. NLP Preprocessing Pipeline
    try:
        nlp = spacy.load(spacy_model_name)
    except OSError:
        logging.critical(f"FATAL: spaCy model '{spacy_model_name}' not loaded. Run 'python -m spacy download {spacy_model_name}'.")
        # Add placeholder column if model loading fails
        df['identified_theme(s)'] = 'NLP_FAILED'
        return df

    # Create the clean text column for feature extraction and rule matching
    df['clean_review'] = df['review'].apply(lambda x: preprocess_text(x, nlp))
    logging.info("Text preprocessing (lemmatization) complete.")

    # B. Keyword and N-gram Extraction (TF-IDF KPI)
    logging.info("Extracting keywords via TF-IDF (1, 2, and 3-grams)...")
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=5)
    
    non_empty_reviews = df[df['clean_review'].str.len() > 0]['clean_review']
    if len(non_empty_reviews) > 0:
        vectorizer.fit(non_empty_reviews) # Fitting the vectorizer meets the TF-IDF requirement
        logging.info(f"TF-IDF vectorizer fitted, capturing {len(vectorizer.get_feature_names_out())} features.")
    else:
        logging.warning("No sufficiently clean reviews found for TF-IDF fitting.")

    # C. Rule-Based Theme Assignment
    logging.info("Assigning themes based on documented keyword rules...")

    def assign_themes(clean_text, rules):
        """Checks clean text against defined rules and returns a comma-separated list of themes."""
        assigned_themes = [
            theme 
            for theme, keywords in rules.items() 
            if any(keyword in clean_text for keyword in keywords)
        ]
        
        return ", ".join(assigned_themes) if assigned_themes else "General/Other"

    df['identified_theme(s)'] = df['clean_review'].apply(
        lambda x: assign_themes(x, theme_rules)
    )

    # Clean up temporary column
    df.drop(columns=['clean_review'], inplace=True, errors='ignore')

    return df


def run_aggregation(df):
    """Aggregates sentiment scores by bank and rating, and prints a formatted table."""
    logging.info("--- Running Aggregation ---")
    
    required_cols = ['sentiment_score', 'bank', 'rating']
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"Missing required columns {required_cols} for aggregation. Skipping.")
        return

    # Filter out invalid sentiment scores (NaN or UNCERTAIN/Neutral 0.5)
    valid_df = df[df['sentiment_score'].notna() & (df['sentiment_score'] != 0.5)]

    if valid_df.empty:
        logging.warning("No valid sentiment scores available for aggregation.")
        return

    # Group and calculate mean sentiment score
    sentiment_agg = valid_df.groupby(['bank', 'rating'])['sentiment_score'].mean().reset_index()
    sentiment_agg.rename(columns={'sentiment_score': 'mean_sentiment_score'}, inplace=True)
    
    # Format the mean score for readability
    sentiment_agg['mean_sentiment_score'] = sentiment_agg['mean_sentiment_score'].round(4)

    # Print a professionally formatted Markdown table
    table = tabulate(sentiment_agg, headers='keys', tablefmt='pipe', showindex=False)
    logging.info("Aggregation Complete. Mean Sentiment Score by Bank and Rating:\n" + table)


def save_results(df, file_path_output):
    """Saves the final processed DataFrame to a CSV file."""
    logging.info(f"Saving final analysis to {file_path_output}...")
    
    final_columns = ['review', 'bank', 'rating', 'date', 'sentiment_label', 
                     'sentiment_score', 'identified_theme(s)']
    
    # Filter columns to only those that exist
    existing_cols = [col for col in final_columns if col in df.columns]
    
    # Final check for essential columns
    if not all(col in existing_cols for col in ['review', 'bank', 'rating']):
        logging.error("FATAL: Essential columns are missing. Cannot save output file.")
        return

    try:
        df[existing_cols].to_csv(file_path_output, index=False)
        logging.info("Final analysis saved successfully.")
    except Exception as e:
        logging.error(f"Error saving output file: {e}")


def main():
    """Main function to orchestrate the Task 2 pipeline."""
    
    df = load_data(CONFIG['FILE_PATH_INPUT'])

    if df is not None:
        # Step 1: Sentiment Analysis
        df = run_sentiment_analysis(df, CONFIG['HUGGING_FACE_MODEL'])
        
        # Step 2: Thematic Analysis
        df = run_thematic_analysis(df, CONFIG['THEME_RULES'], CONFIG['SPACY_MODEL'])

        # Step 3: Aggregation and Reporting (prints to console)
        run_aggregation(df) 

        # Step 4: Save Final Results
        save_results(df, CONFIG['FILE_PATH_OUTPUT'])
    else:
        logging.error("Pipeline aborted due to data loading failure.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"An unexpected error occurred during pipeline execution: {e}")