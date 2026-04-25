import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure plots directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Theme settings
BG = '#0F1117'
PANEL = '#1A1D2E'
plt.rcParams.update({'axes.facecolor': BG, 'figure.facecolor': BG, 
                     'text.color': 'white', 'axes.labelcolor': 'white',
                     'xtick.color': 'white', 'ytick.color': 'white'})

# ==============================================================================
# 1. LOAD AND JOIN KAGGLE DATASETS
# ==============================================================================
print("Loading Kaggle CSVs...")
try:
    tracks_df = pd.read_csv('datasets/tracks.csv')
    lyrics_df = pd.read_csv('datasets/ds2.csv')
except FileNotFoundError:
    print("Error: Could not find the CSV files. Ensure they are in the 'datasets' folder.")
    exit()

print("Cleaning and joining data...")

# --- DYNAMIC COLUMN RENAMER ---
def rename_col(df, target, options):
    """Scans for possible column names and standardizes them."""
    for opt in options:
        if opt in df.columns:
            df.rename(columns={opt: target}, inplace=True)
            return
    print(f"Warning: Could not find a match for '{target}'. Available columns: {df.columns.tolist()}")

# Dynamically rename Tracks columns
rename_col(tracks_df, 'track_name', ['name', 'track_name', 'title', 'track'])
rename_col(tracks_df, 'artist_name', ['artists', 'artist_name', 'artist'])
rename_col(tracks_df, 'release_year', ['year', 'release_year'])

# Fallback: if 'year' is missing but 'release_date' exists, extract the year (YYYY)
if 'release_year' not in tracks_df.columns and 'release_date' in tracks_df.columns:
    tracks_df['release_year'] = tracks_df['release_date'].astype(str).str[:4]

# Ensure year is a clean number
if 'release_year' in tracks_df.columns:
    tracks_df['release_year'] = pd.to_numeric(tracks_df['release_year'], errors='coerce')

# Dynamically rename Lyrics columns (Kaggle lyric CSVs frequently use 'text' or 'song_name')
rename_col(lyrics_df, 'track_name', ['song', 'song_name', 'name', 'title', 'track_name'])
rename_col(lyrics_df, 'artist_name', ['artist', 'artist_name', 'artists'])
rename_col(lyrics_df, 'lyric_text', ['text', 'lyrics', 'lyric_text', 'lyric'])

# Verify the rename worked before joining
if 'track_name' not in lyrics_df.columns or 'track_name' not in tracks_df.columns:
    print("CRITICAL ERROR: Failed to map 'track_name'. Please check your CSV headers!")
    print("Tracks columns:", tracks_df.columns.tolist())
    print("Lyrics columns:", lyrics_df.columns.tolist())
    exit()

# Convert text to lowercase string to ensure a clean join
tracks_df['track_name'] = tracks_df['track_name'].astype(str).str.lower()
lyrics_df['track_name'] = lyrics_df['track_name'].astype(str).str.lower()

# Inner join: Keep ONLY songs that have both Audio Features AND Lyrics
df = pd.merge(tracks_df, lyrics_df, on=['track_name'], how='inner')

# Drop duplicates and missing values
df = df.dropna(subset=['lyric_text', 'valence', 'release_year'])
df = df.drop_duplicates(subset=['track_name'])

print(f"Successfully joined! Analyzable dataset size: {len(df)} tracks.")

# Optional: Sample the data if it's too massive (NLP takes time)
if len(df) > 10000:
    print("Dataset is massive. Sampling 10,000 random tracks for NLP speed...")
    df = df.sample(10000, random_state=42)

# ==============================================================================
# 2. ANALYSIS A: AUDIO-LYRICAL DISSONANCE
# ==============================================================================
print("Running NLP Sentiment Analysis on lyrics...")
analyzer = SentimentIntensityAnalyzer()

# Calculate Vader compound score (-1 to 1)
df['lyric_sentiment_raw'] = df['lyric_text'].astype(str).apply(
    lambda text: analyzer.polarity_scores(text)['compound']
)

# Scale lyric sentiment from [-1, 1] to [0, 1] to match Spotify's Valence scale
df['lyric_sentiment_scaled'] = (df['lyric_sentiment_raw'] + 1) / 2

# Dissonance Formula: High Audio Valence (Happy Sound) - Low Lyric Sentiment (Sad Text)
df['dissonance_score'] = df['valence'] - df['lyric_sentiment_scaled']

print("Plotting Dissonance Matrix...")
fig1, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df['lyric_sentiment_scaled'], df['valence'], 
                     c=df['dissonance_score'], cmap='coolwarm', alpha=0.6, s=15)

# Add quadrants
ax.axhline(0.5, color='gray', linestyle='--')
ax.axvline(0.5, color='gray', linestyle='--')
ax.text(0.05, 0.95, 'Sad Bangers\n(Happy Sound / Sad Lyrics)', color='white', fontsize=12, va='top')
ax.text(0.75, 0.95, 'Pure Pop\n(Happy/Happy)', color='white', fontsize=12, va='top')
ax.text(0.05, 0.05, 'Ballads\n(Sad/Sad)', color='white', fontsize=12, va='bottom')
ax.text(0.75, 0.05, 'Angry/Edgy\n(Sad Sound / Happy Lyrics)', color='white', fontsize=12, va='bottom')

ax.set_title("Audio-Lyrical Dissonance Matrix", fontsize=16)
ax.set_xlabel("Lyrical Sentiment (NLP: 0=Sad, 1=Happy)")
ax.set_ylabel("Audio Valence (Spotify: 0=Sad, 1=Happy)")
fig1.colorbar(scatter, label="Dissonance Score")

fig1.savefig('plots/dissonance_scatter.png', dpi=150, bbox_inches='tight')
print("Saved -> plots/dissonance_scatter.png")

# ==============================================================================
# 3. ANALYSIS B: CULTURAL HOMOGENIZATION OVER TIME
# ==============================================================================
print("Plotting Homogenization (Variance over time)...")

# Filter out years with too few songs to be statistically significant
year_counts = df['release_year'].value_counts()
valid_years = year_counts[year_counts > 20].index
trend_df = df[df['release_year'].isin(valid_years)]

# Group by release year and calculate the standard deviation (variance)
variance_df = trend_df.groupby('release_year')[['valence', 'energy', 'danceability', 'acousticness']].std().reset_index()
# Sort chronologically
variance_df = variance_df.sort_values('release_year')

fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(variance_df['release_year'], variance_df['valence'], label='Valence Variance', linewidth=2)
ax.plot(variance_df['release_year'], variance_df['energy'], label='Energy Variance', linewidth=2)
ax.plot(variance_df['release_year'], variance_df['danceability'], label='Danceability Variance', linewidth=2)
ax.plot(variance_df['release_year'], variance_df['acousticness'], label='Acousticness Variance', linewidth=2, linestyle=':')

ax.set_title("Cultural Homogenization: Variance of Track Audio Features Over Time", fontsize=16)
ax.set_xlabel("Release Year")
ax.set_ylabel("Standard Deviation (Lower = Tracks Sound More Alike)")
ax.legend(facecolor=PANEL, edgecolor='none')
ax.grid(alpha=0.2)

# Set x-limits to realistic decades (e.g., 1960 to 2020)
ax.set_xlim(left=1960, right=max(variance_df['release_year']))

fig2.savefig('plots/homogenization_trends.png', dpi=150, bbox_inches='tight')
print("Saved -> plots/homogenization_trends.png")
print("\nAnalysis Complete! Check the 'plots' folder.")