import pandas as pd
import psycopg2
import logging
from datetime import datetime
import os

# --- Configuration and Logging Setup ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# You MUST fill in your actual PostgreSQL credentials here
DB_CONFIG = {
    'host': 'localhost', 
    'database': 'bank_reviews',
    'user': 'postgres', # Ensure this is your correct PostgreSQL user
    'password': 'postgres', # Ensure this is your correct password
    'port': '5432'
}

# The files are located in the 'fintech-review-analysis/' subdirectory
FILE_PATH_INPUT = 'fintech-review-analysis/task2_processed_reviews.csv'
REVIEW_COLUMNS = ['bank_name', 'rating', 'review_text', 'review_date', 'sentiment_label', 'sentiment_score', 'identified_themes']
# NOTE: The CSV should ideally contain a column that holds the bank/app name that matches the bank_map keys. 
# We will use 'Bank/App Identifier' as a placeholder and log the failure.
BANK_IDENTIFIER_COLUMN = 'app_name' 


def create_connection():
    """Establishes a connection to the PostgreSQL database."""
    logging.info("Attempting to connect to PostgreSQL...")
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        # CRITICAL FIX: Set Search Path to 'public' to find the tables
        with conn.cursor() as cursor:
            cursor.execute("SET search_path TO public;")
        logging.info("Successfully connected to PostgreSQL.")
        return conn
    except psycopg2.Error as e:
        logging.critical(f"Database connection failed: {e}")
        return None


def load_data(file_path):
    """Loads and returns the processed CSV data using pandas."""
    try:
        df = pd.read_csv(file_path)
        # --- CRITICAL DEBUGGING LINE ---
        logging.info(f"Loaded CSV. Columns detected: {list(df.columns)}")
        # ---------------------------------
        return df
    except FileNotFoundError:
        logging.critical(f"Input file not found: {file_path}. Run Task 2 first.")
        return None
    except Exception as e:
        logging.critical(f"Error loading CSV data: {e}")
        return None


def get_bank_ids(conn):
    """Fetches bank IDs and creates a mapping from bank/app name to ID."""
    logging.info("Fetching bank IDs and app names for mapping...")
    bank_map = {}
    
    # -------------------------------------------------------------
    # FIX: Define common abbreviations found in the CSV
    # These will be mapped to the full bank names found in the DB.
    ABBREVIATION_MAP = {
        'CBE': 'Commercial Bank of Ethiopia',
        'BOA': 'Bank of Abyssinia',
        'DB': 'Dashen Bank',
        'Dashen': 'Dashen Bank' # Added "Dashen" just in case
    }
    # -------------------------------------------------------------
    
    query = "SELECT bank_id, bank_name, app_name FROM banks;"
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()

            # Map from full name to ID first
            full_name_to_id = {}
            for bank_id, bank_name, app_name in results:
                full_name_to_id[bank_name.strip()] = bank_id

            # Now build the final bank_map using all identifiers
            for bank_id, bank_name, app_name in results:
                name_clean = bank_name.strip()
                
                # 1. Map the formal bank name to its ID
                bank_map[name_clean] = bank_id
                
                # 2. Map the app name (if different) to the same ID
                if app_name and app_name.strip() != name_clean:
                    bank_map[app_name.strip()] = bank_id
            
            # 3. Map common abbreviations (the CRITICAL fix)
            for abbr, full_name in ABBREVIATION_MAP.items():
                if full_name in full_name_to_id:
                    # Map the abbreviation (e.g., 'CBE') to the correct ID
                    bank_map[abbr] = full_name_to_id[full_name]


        logging.info(f"Loaded {len(results)} banks from database. Mapping has {len(bank_map)} entries including abbreviations.")
        return bank_map
    except psycopg2.Error as e:
        logging.error(f"Error fetching bank IDs: {e}")
        return None


def insert_reviews(conn, df, bank_map):
    """Inserts review data into the reviews table."""
    total_reviews = len(df)
    inserted_count = 0
    
    # Identify the correct column name dynamically
    # CRITICAL FIX: Prioritize 'bank' column name from the CSV output
    if 'bank' in df.columns:
        bank_id_col = 'bank'
    elif 'app_name' in df.columns:
        bank_id_col = 'app_name'
    elif 'bank_name' in df.columns:
        bank_id_col = 'bank_name'
    else:
        logging.error("Cannot find 'bank', 'app_name' or 'bank_name' column in the CSV for bank ID mapping. Insertion failed.")
        logging.error(f"CSV columns found were: {list(df.columns)}. Please update the logic to use one of these for the bank identifier.")
        return 0

    logging.info(f"Starting review insertion for {total_reviews} records, using column '{bank_id_col}' for mapping...")

    # SQL query template for insertion
    sql = """
    INSERT INTO reviews (bank_id, rating, review_text, review_date, sentiment_label, sentiment_score, identified_themes)
    VALUES (%s, %s, %s, %s, %s, %s, %s);
    """

    try:
        with conn.cursor() as cursor:
            for index, row in df.iterrows():
                try:
                    # 1. Get the bank identifier and strip whitespace
                    bank_identifier = str(row[bank_id_col]).strip()
                    bank_id = bank_map.get(bank_identifier)

                    if bank_id is None:
                        # Log the exact value that failed the lookup
                        if inserted_count == 0 and index < 5:
                            logging.warning(f"Row {index} skipped: Bank identifier '{bank_identifier}' not found in map. Check spelling/whitespace in CSV.")
                        continue
                    
                    # 2. Extract and format other fields
                    rating = int(row['rating'])
                    # FIX: Use 'review' column name from CSV
                    review_text = str(row['review']) 
                    
                    # Handle DATE format (assuming it's a valid date string)
                    try:
                        # FIX: Use 'date' column name from CSV
                        review_date = datetime.strptime(str(row['date']).split()[0], '%Y-%m-%d').date()
                    except:
                        # Fallback to None if date parsing fails
                        review_date = None

                    # Handle numeric and text fields
                    sentiment_label = str(row['sentiment_label'])
                    sentiment_score = float(row['sentiment_score'])
                    # FIX: Use 'identified_theme(s)' column name from CSV
                    identified_themes = str(row['identified_theme(s)']) 
                    
                    # Prepare the data tuple
                    data = (
                        bank_id,
                        rating,
                        review_text,
                        review_date,
                        sentiment_label,
                        sentiment_score,
                        identified_themes
                    )
                    
                    # Execute insertion
                    cursor.execute(sql, data)
                    inserted_count += 1

                except Exception as e:
                    logging.error(f"Failed to insert row {index} (Bank ID: {bank_id}, Identifier: {bank_identifier}): {e}")
                    # Continue to the next row if insertion fails
                    continue

            conn.commit()
            logging.info(f"Insertion complete. {inserted_count}/{total_reviews} reviews successfully inserted.")
            
    except psycopg2.Error as e:
        logging.error(f"Database error during batch insertion: {e}")
        conn.rollback()
    
    return inserted_count


def main():
    conn = create_connection()
    if conn is None:
        return

    df = load_data(FILE_PATH_INPUT)
    if df is None:
        conn.close()
        return

    bank_map = get_bank_ids(conn)
    if bank_map is None or not bank_map:
        conn.close()
        return

    insert_reviews(conn, df, bank_map)
    
    conn.close()
    logging.info("Database connection closed.")


if __name__ == "__main__":
    main()