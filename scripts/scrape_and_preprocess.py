from google_play_scraper import Sort, reviews
import pandas as pd

APP_IDS = {
    'CBE': 'com.combanketh.mobilebanking',
    'BOA': 'com.boa.apollo', # Using the newer 'Apollo' app
    'Dashen': 'com.dashen.dashensuperapp'
}

all_reviews_data = []

for bank_name, app_id in APP_IDS.items():
    print(f"Scraping reviews for {bank_name} ({app_id})...")
    
    # Attempt to fetch over 400 reviews
    result, continuation_token = reviews(
        app_id,
        lang='en', # English language
        country='et', # Ethiopia country code
        sort=Sort.NEWEST, # Sort by newest to get fresh data
        count=500, # Request more than 400 to ensure the minimum is met
        filter_score_with=None # Get all ratings (1-5 stars)
    )
    
    # Process and tag the fetched reviews
    for r in result:
        all_reviews_data.append({
            'review': r['content'],
            'rating': r['score'],
            'date': r['at'],
            'bank': bank_name,
            'source': 'Google Play' # Required 'source' column
        })
        
print("Scraping complete.")

# Convert to a DataFrame for preprocessing
df = pd.DataFrame(all_reviews_data)
# Remove duplicates based on key columns
df.drop_duplicates(subset=['review', 'rating', 'date', 'bank'], inplace=True)
print(f"Removed duplicates. Total reviews now: {len(df)}")
# Check for missing values
missing_data_count = df.isnull().sum().sum()
total_cells = len(df) * len(df.columns)
missing_percentage = (missing_data_count / total_cells) * 100

print(f"Total missing data percentage: {missing_percentage:.2f}%")

# If the percentage is acceptable, drop rows with *critical* missing data
df.dropna(subset=['review', 'rating', 'date'], inplace=True)
# The 'at' column from the scraper is typically a datetime object, 
# but it's best practice to explicitly ensure it's in the required string format.
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
# Final check on column order and name compliance
final_df = df[['review', 'rating', 'date', 'bank', 'source']]

# Save the clean dataset to a CSV file
output_file = 'playstore_reviews_task1_clean.csv'
final_df.to_csv(output_file, index=False)

print(f"Successfully cleaned and saved data to {output_file}")