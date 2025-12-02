import pandas as pd
import psycopg2
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- Configuration and Logging Setup ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Reuse the same DB configuration as the insertion script
DB_CONFIG = {
    'host': 'localhost', 
    'database': 'bank_reviews',
    'user': 'postgres', # Ensure this is your correct PostgreSQL user
    'password': 'postgres', # Ensure this is your correct password
    'port': '5432'
}

# --- Utility Functions ---

def create_connection():
    """Establishes a connection to the PostgreSQL database."""
    logging.info("Attempting to connect to PostgreSQL...")
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute("SET search_path TO public;")
        logging.info("Successfully connected to PostgreSQL.")
        return conn
    except psycopg2.Error as e:
        logging.critical(f"Database connection failed: {e}")
        return None

def fetch_data(conn):
    """Fetches all reviews data from the database and loads it into a Pandas DataFrame."""
    logging.info("Fetching all review data...")
    sql = """
    SELECT 
        b.bank_name,
        r.rating,
        r.review_text,
        r.sentiment_label,
        r.sentiment_score,
        r.identified_themes,
        r.review_date -- <-- ADDED: Fetch the review date
    FROM reviews r
    JOIN banks b ON r.bank_id = b.bank_id
    ORDER BY r.review_date;
    """
    try:
        df = pd.read_sql(sql, conn)
        logging.info(f"Successfully fetched {len(df)} records.")
        
        # Clean up data types and convert columns
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').astype('Int64')
        df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
        # Convert review_date to datetime objects
        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def explode_themes(df):
    """Splits comma-separated themes and explodes the DataFrame for theme-level analysis."""
    # Ensure all themes are treated as strings and handle NaN/None
    df['identified_themes'] = df['identified_themes'].astype(str).str.strip()
    df_themes = df[df['identified_themes'] != 'nan'].copy()
    
    # Split the themes string into a list and explode
    df_themes['theme'] = df_themes['identified_themes'].str.split(',').apply(lambda x: [t.strip() for t in x])
    df_exploded = df_themes.explode('theme').copy()
    
    # Classify themes by general sentiment based on high/low rating
    df_exploded['theme_type'] = df_exploded.apply(
        lambda row: 'Driver' if row['rating'] >= 4 else ('Pain Point' if row['rating'] <= 2 else 'Neutral'),
        axis=1
    )
    return df_exploded

def generate_insights(df, df_exploded):
    """Derives key insights (drivers, pain points, comparisons)."""
    insights = {}
    
    banks = df['bank_name'].unique()
    
    # 1. Drivers and Pain Points per Bank
    for bank in banks:
        bank_data = df_exploded[df_exploded['bank_name'] == bank]
        
        # Top 3 Drivers (Themes associated with high rating)
        drivers = bank_data[bank_data['theme_type'] == 'Driver']['theme'].value_counts().head(3).index.tolist()
        
        # Top 3 Pain Points (Themes associated with low rating)
        pain_points = bank_data[bank_data['theme_type'] == 'Pain Point']['theme'].value_counts().head(3).index.tolist()
        
        insights[bank] = {
            'drivers': drivers,
            'pain_points': pain_points
        }

    # 2. Comparative Analysis (e.g., average rating)
    avg_ratings = df.groupby('bank_name')['rating'].mean().sort_values(ascending=False)
    
    insights['comparison'] = {
        'avg_ratings': avg_ratings.to_dict(),
        'best_bank_rating': avg_ratings.index[0],
        'worst_bank_rating': avg_ratings.index[-1]
    }

    # 3. Overall Sentiment Distribution
    sentiment_counts = df['sentiment_label'].value_counts(normalize=True).mul(100).round(2)
    insights['overall_sentiment'] = sentiment_counts.to_dict()

    return insights

# --- Visualization Functions ---

def generate_plots(df, df_exploded):
    """Generates 3-5 plots and saves them."""
    
    # Set style for plots
    sns.set_style("whitegrid")
    
    # 1. Overall Sentiment Distribution (Pie Chart)
    plt.figure(figsize=(7, 7))
    sentiment_counts = df['sentiment_label'].value_counts()
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90,
            colors=sns.color_palette("pastel"))
    plt.title('Overall Sentiment Distribution of Fintech Reviews', fontsize=14)
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    plt.close()
    logging.info("Generated 'sentiment_distribution.png'")
    
    # 2. Rating Distribution per Bank (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='rating', hue='bank_name', palette='viridis')
    plt.title('Rating Distribution Across Banks', fontsize=14)
    plt.xlabel('Rating (1=Worst, 5=Best)')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Bank')
    plt.tight_layout()
    plt.savefig('rating_distribution.png')
    plt.close()
    logging.info("Generated 'rating_distribution.png'")

    # 3. Top 10 Pain Points vs. Drivers (Aggregated Theme Analysis)
    theme_summary = df_exploded.groupby('theme')['theme_type'].value_counts().unstack(fill_value=0)
    # Focus on themes with strong sentiment (Drivers/Pain Points)
    theme_summary['Total_Impact'] = theme_summary['Driver'] + theme_summary['Pain Point']
    top_impact_themes = theme_summary.sort_values(by='Total_Impact', ascending=False).head(10)

    top_impact_themes[['Driver', 'Pain Point']].plot(kind='bar', figsize=(12, 7), stacked=True, 
                                                     color=['#78C478', '#E67F7F'])
    plt.title('Top 10 Most Frequent Themes by Sentiment Type (Drivers vs. Pain Points)', fontsize=14)
    plt.xlabel('Theme')
    plt.ylabel('Review Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sentiment Type')
    plt.tight_layout()
    plt.savefig('top_themes_sentiment.png')
    plt.close()
    logging.info("Generated 'top_themes_sentiment.png'")
    
    # 4. Sentiment Trend Over Time (Using Rating Trend as proxy)
    # Group by month and calculate mean rating
    df_date = df.dropna(subset=['review_date']).copy()
    # Check if df_date is empty after dropping NaT dates
    if df_date.empty:
        logging.warning("No valid review dates found for trend plot. Skipping 'rating_trend.png'.")
        return
        
    df_date['month_year'] = df_date['review_date'].dt.to_period('M')
    
    # Sort for correct plotting order
    rating_trend = df_date.groupby('month_year')['rating'].mean().reset_index().sort_values('month_year')
    rating_trend['month_year'] = rating_trend['month_year'].astype(str)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=rating_trend, x='month_year', y='rating', marker='o', color='darkblue')
    plt.title('Average Rating Trend Over Time', fontsize=14)
    plt.xlabel('Month-Year')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(df['rating'].min() - 0.5, df['rating'].max() + 0.5) # Dynamic limits
    plt.tight_layout()
    plt.savefig('rating_trend.png')
    plt.close()
    logging.info("Generated 'rating_trend.png'")
    
    # 5. Word Cloud (Placeholder - Requires additional libraries like wordcloud, so we will skip actual generation 
    # and instead provide the analysis in the report)
    logging.info("Skipping Word Cloud generation due to external library dependency constraints.")


def main():
    conn = create_connection()
    if conn is None:
        return

    df = fetch_data(conn)
    conn.close()
    
    if df.empty:
        logging.error("No data available for analysis. Exiting.")
        return
    
    df_exploded = explode_themes(df)
    
    insights = generate_insights(df, df_exploded)
    
    # Print insights to console (these will be used to write the report)
    print("\n--- Bank-Specific Insights ---")
    # Determine the order of banks to ensure consistent output (BOA, CBE, Dashen)
    bank_order = sorted(insights.keys() - {'comparison', 'overall_sentiment'})

    for bank in bank_order:
        data = insights[bank]
        print(f"\n{bank}:")
        print(f"  Drivers (Positive Themes): {', '.join(data['drivers'])}")
        print(f"  Pain Points (Negative Themes): {', '.join(data['pain_points'])}")

    print("\n--- Comparative Analysis ---")
    print(f"Average Ratings: {insights['comparison']['avg_ratings']}")
    print(f"Overall Sentiment: {insights['overall_sentiment']}")

    print("\n--- Generating Visualizations ---")
    # This will generate and save the PNG files in your current working directory
    generate_plots(df, df_exploded)
    print("Visualizations saved as PNG files (e.g., 'sentiment_distribution.png').")

if __name__ == "__main__":
    main()