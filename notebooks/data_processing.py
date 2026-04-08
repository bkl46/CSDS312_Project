import pandas as pd
from langdetect import detect, DetectorFactory
import numpy as np
import spacy
import re
import argparse
import sys
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def initial_clean(df):
    #drop uneeded columns
    cols_to_drop = ['Media', 'Song URL', 'Album URL', 'Writers', 'Featured Artists']
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols)
    print(f"Dropped {existing_cols} from df")

    #convert year to int, release date to datetime
    df["Year"] = df["Year"].astype("int64")
    df["Release Date"] = pd.to_datetime(df["Release Date"], errors='coerce')
    df["Release Date"] = df["Release Date"].dt.normalize()

    #add decade column
    df['decade'] = (df['Year'] // 10) * 10
    
    return df

#drops rows with no lyrics
def drop_missing_lyrics(df):
    before = len(df)
    df = df.dropna(subset=['Lyrics'])
    after = len(df)
    print(f"Dropped {before - after} rows with missing lyrics")
    return df

#normalize a line of lyrics (lower case, no sections, normal spacing, no special characters
def clean_lyrics_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)  # remove [bracketed] content
    text = re.sub(r"(chorus|verse|bridge|intro|outro|pre-chorus|hook|feat|ft\.)\s*\d*:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters and spaces
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    text = text.strip()
    
    return text

def tokenize_and_lemmatize(text, nlp):
    """convert text to lemmatized tokens, removes stop words and single character tokens"""
    if not text or len(text) < 10:
        return []
    
    doc = nlp(text)
    tokens = [
        token.lemma_ 
        for token in doc 
        if token.is_alpha 
        and not token.is_stop 
        and len(token.lemma_) > 1
    ]
    return tokens


def add_sentiment(df):
    """Add VADER sentiment compound score to each song"""
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        if pd.isna(text) or len(text) < 20:
            return 0.0
        return analyzer.polarity_scores(text)['compound']
    
    df['sentiment'] = df['cleanedLyrics'].apply(get_sentiment)
    print(f"Sentiment added - range: [{df['sentiment'].min():.2f}, {df['sentiment'].max():.2f}]")
    return df

def add_sentiment_category(df):
    """Categorize songs as positive, neutral, or negative"""
    df['sentiment_category'] = pd.cut(
        df['sentiment'],
        bins=[-1, -0.05, 0.05, 1],
        labels=['negative', 'neutral', 'positive']
    )
    return df


def add_text_features(df):
    """add token count, unique words, lexical richness"""
    
    df['tokenCount'] = df['tokens'].apply(len)
    
    df['uniqueWordCount'] = df['tokens'].apply(lambda x: len(set(x)))
    
    df['avgWordLength'] = df['tokens'].apply(
        lambda toks: sum(len(w) for w in toks) / len(toks) if len(toks) > 0 else 0
    )
    
    df['lexicalRichness'] = df.apply(
        lambda row: row['uniqueWordCount'] / row['tokenCount'] if row['tokenCount'] > 0 else 0,
        axis=1
    )
    
    print(f"text features added - avg tokencount: {df['tokenCount'].mean():.1f}")
    return df

def remove_outliers(df):
    before = len(df)
    # remove outliers
    upper_cutoff = df['tokenCount'].quantile(0.99)
    df = df[df['tokenCount'] <= upper_cutoff]
    
    # Remove very short songs
    df = df[df['tokenCount'] >= 20]
    
    after = len(df)
    print(f"removed {before - after} outlier songs")
    return df.reset_index(drop=True)


def detect_languages(df, sample_size=5000):
    DetectorFactory.seed =0  # for reproducibility
    
    def safe_detect(text):
        try:
            if pd.isna(text) or len(text) < 50:
                return None
            return detect(text)
        except:
            return None
    
    # if small enough just make whole df the sample
    if len(df) > sample_size:
        sample = df.sample(sample_size, random_state=42)
    else:
        sample = df
    
    sample['language'] = sample['Lyrics'].apply(safe_detect)
    lang_dist = sample['language'].value_counts()
    print("Language distribution in sample:")
    print(lang_dist)
    
    # use detect to get language column
    print("Detecting language on all rows...")
    df['language'] = df['Lyrics'].apply(safe_detect)
    
    english_df = df[df['language'] == 'en'].copy()
    non_english = df[df['language'] != 'en']
    
    print(f"{len(english_df)} English songs")
    print(f"{len(non_english)} non-English songs")
    
    # save non-english, return english
    if len(non_english) > 0:
        non_english.to_csv("non_english_songs.csv", index=False)
        print("Saved non-English songs to non_english_songs.csv")
    
    return english_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Prepare lyrics dataset for analysis')
    parser.add_argument('--input', type=str, required=False, 
                        help='Path to input CSV file (raw songs data)')
    parser.add_argument('--output', type=str,
                        help='Output filename for cleaned data')
    parser.add_argument('--skip_lang_detect', action='store_true',
                        help='Skip language detection (faster, assumes English)')
    args = parser.parse_args()

    #default args
    path = args.input if args.input else "../data/raw/all_songs_data.csv"
    outpath = args.output if args.output else "../data/processed/processed.csv"
    skip = args.skip_lang_detect if args.skip_lang_detect else True
    
    df = pd.read_csv(path)
    print("raw data loaded, beginning preprocessing..")
    print("columns: ")
    print(df.columns)
    
    #initial cleaning 
    df = initial_clean(df)
    df = drop_missing_lyrics(df)


    print("cleaning lyrics text")
    df['cleanedLyrics'] = df['Lyrics'].apply(clean_lyrics_text)

    
    # text cleaning wiht spacy model
    print("loading spaCy model:")
    nlp = spacy.load("en_core_web_sm")
    print("spacy loaded")
    
    print("tokenizing and lemmatizing ")
    df['tokens'] = df['cleanedLyrics'].apply(lambda x: tokenize_and_lemmatize(x, nlp))
    
    #filter out empty songs
    before = len(df)
    df = df[df['tokens'].apply(len) > 0]
    print(f"removed {before - len(df)} songs with no tokens after cleaning")
    
    # Step 5: Add features
    df = add_text_features(df)
    df = remove_outliers(df)
    
    # sentiment analysis
    print("adding sentiment analysis")
    df = add_sentiment(df)
    df = add_sentiment_category(df)
    
    # filter languages
    if not skip:
        df = detect_languages(df)
    
    # make final dataset with only important columns
    final_columns = [
        'Rank', 'Song Title', 'Artist', 'Year', 'decade', 
        'Album', 'Release Date', 'Lyrics', 'cleanedLyrics', 
        'tokens', 'tokenCount', 'uniqueWordCount', 
        'avgWordLength', 'lexicalRichness', 
        'sentiment', 'sentiment_category'
    ]
    final_columns = [c for c in final_columns if c in df.columns]
    df_final = df[final_columns]
    
    df_final.to_csv(outpath, index=False)
    print(f"saved to {outpath}")
    print(f"number of songs:{len(df_final)}")
    print(f"date range: {df_final['Year'].min()} - {df_final['Year'].max()}")
    print(f"decades present: {sorted(df_final['decade'].unique())}")
    print(f"avg sentiment: {df_final['sentiment'].mean():.3f}")
    print(f"avg lexical richness: {df_final['lexicalRichness'].mean():.3f}")
    print(f"avg tokens per song: {df_final['tokenCount'].mean():.1f}")

if __name__ == "__main__":
    main()
