# 🏦 Fintech App Review Analysis: Week 2 Challenge

## Task 1: Data Collection and Preprocessing

The objective of Task 1 was to collect a minimum of 1,200 user reviews from the Google Play Store for three major Ethiopian banks, preprocess the data for consistency and quality, and establish a version-controlled codebase using Git.

### ⚙️ Project Setup and Version Control (Git Methodology)

The project setup followed the requirements for robust version control:

1.  **Repository & Branch:** A dedicated GitHub repository was created, and all work for this phase was conducted on the **`task-1`** branch.
2.  **Environment Files:**
    * **`requirements.txt`**: Created to list all necessary Python libraries for reproducibility, including `google-play-scraper` and `pandas`.
    * **`.gitignore`**: Configured to exclude generated files (like the large CSV dataset) and environment artifacts (`__pycache__`, `.venv`).
3.  **Committing:** Commits were made frequently, specifically after establishing the project structure, finalizing the scraping logic, and completing the data preprocessing script.

### 🕷️ Data Scraping Methodology

[cite_start]User reviews were collected from the Google Play Store using the **`google-play-scraper`** Python library[cite: 77].

#### Target Applications
The scraping targeted the primary mobile banking applications for the three specified banks:

| Bank Name | App ID (Used for Scraping) | App Name | Reviews Target |
| :--- | :--- | :--- | :--- |
| **Commercial Bank of Ethiopia (CBE)** | `com.combanketh.mobilebanking` | Commercial Bank of Ethiopia | $\geq 400$ |
| **Bank of Abyssinia (BOA)** | `com.boa.apollo` | Apollo | $\geq 400$ |
| **Dashen Bank** | `com.dashen.dashensuperapp` | Dashen Bank (Super App) | $\geq 400$ |

#### Data Points Collected
[cite_start]The scraping script was configured to retrieve the following data points for each review[cite: 77]:
* **Review Text**
* **Rating** (1-5 star score)
* **Date** (Date and time of posting)
* **Bank/App Name** (Tagged during processing)

#### Volume
[cite_start]A total of **1,200+** reviews were collected to satisfy the minimum requirement of 400 reviews per bank[cite: 78, 85].

### 🧹 Data Preprocessing Methodology

The raw data was processed using Python and the `pandas` library to ensure data quality and uniformity, as implemented in the script located at `scripts/scrape_and_preprocess.py`.

1.  [cite_start]**Tagging:** A `bank` column was added to identify the source of each review, and a `source` column was set to **"Google Play"**[cite: 81].
2.  [cite_start]**Duplicate Removal:** Duplicate entries were identified and removed based on a combination of `review` text, `rating`, `date`, and `bank` to prevent redundant analysis[cite: 79].
3.  **Missing Data Handling:** The dataset was checked for missing values (`NaN`). Rows with null values in critical columns (`review`, `rating`, `date`) were dropped to maintain integrity. [cite_start]The overall missing data rate was kept **below the KPI of 5%**[cite: 82].
4.  [cite_start]**Date Normalization:** The original review date/time objects were converted and standardized to the required **`YYYY-MM-DD`** format[cite: 80].

### ✅ Deliverables and KPIs Achieved

| KPI | Status | Note |
| :--- | :--- | :--- |
| **Total Reviews Collected** | **ACHIEVED** | [cite_start]Collected $\geq 1,200$ reviews (400+ per bank)[cite: 78, 82]. |
| **Data Quality** | **ACHIEVED** | [cite_start]Missing data was $<5\%$ after cleaning[cite: 82]. |
| **Clean Dataset** | **ACHIEVED** | Final data saved to `playstore_reviews_task1_clean.csv`. |
| **Git Organization** | **ACHIEVED** | [cite_start]Dedicated `task-1` branch used, with clear, frequent commits[cite: 75, 83]. |
| **Preprocessing Script** | **ACHIEVED** | [cite_start]Script is committed at `scripts/scrape_and_preprocess.py`[cite: 85]. |

#### Final Data Schema
[cite_start]The final output CSV contains the following columns[cite: 81]:

* `review`
* `rating`
* `date`
* `bank`
* `source` 

 Sentiment Analysis

Tool: Hugging Face transformers pipeline.

Model: distilbert-base-uncased-finetuned-sst-2-english.

Output: Adds two new columns:

sentiment_label: POSITIVE, NEGATIVE, or UNCERTAIN.

sentiment_score: A normalized score (0.0 = Very Negative, 1.0 = Very Positive).

Thematic Analysis

This step uses a hybrid approach:

Feature Extraction: TfidfVectorizer (with 1, 2, and 3-grams) is fitted to the clean text to meet the KPI requirements for keyword analysis.

Rule-Based Clustering: Themes are assigned to each review by checking for the presence of specific keywords within the cleaned text.

Aggregation and Output

Aggregation: Calculates the mean_sentiment_score grouped by bank and rating (1 to 5 stars). The aggregated results are printed to the console as a Markdown table.

Output File: The final processed DataFrame, including all generated sentiment and theme columns, is saved to task2_processed_reviews.csv.