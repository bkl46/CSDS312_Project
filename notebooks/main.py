



#!/usr/bin/env python3
"""
CSDS312 - Spotify Analysis with PySpark (Corrected for Your Dataset)
Topic: Text Mining of Great Works (Hit Songs) - Lyrics Analysis
"""

import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, Tokenizer, StopWordsRemover,
    CountVectorizer, IDF, HashingTF
)
from pyspark.ml.clustering import KMeans, LDA
from pyspark.ml.classification import RandomForestClassifier
#import pandas
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator, 
    ClusteringEvaluator,
    BinaryClassificationEvaluator
)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "/mnt/vstor/courses/csds312/bkl46/Proj/data/spotify-dataset/spotify_dataset.csv"
OUTPUT_PATH = "/mnt/vstor/courses/csds312/bkl46/Proj/output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Column mappings for YOUR dataset
COL_SONG = "song"
COL_ARTIST = "Artist(s)"
COL_LYRICS = "text"  # You have lyrics!
COL_GENRE = "Genre"
COL_ALBUM = "Album"
COL_RELEASE_DATE = "Release Date"
COL_POPULARITY = "Popularity"
COL_DANCEABILITY = "Danceability"
COL_ENERGY = "Energy"
COL_VALENCE = "Positiveness"  # Your dataset calls valence "Positiveness"
COL_SPEECHINESS = "Speechiness"
COL_LIVENESS = "Liveness"
COL_ACOUSTICNESS = "Acousticness"
COL_INSTRUMENTALNESS = "Instrumentalness"
COL_TEMPO = "Tempo"
COL_KEY = "Key"
COL_LOUDNESS = "Loudness (db)"
COL_EXPLICIT = "Explicit"
COL_EMOTION = "emotion"
COL_TIME_SIG = "Time signature"

def create_spark_session():
    """Create Spark session."""
    spark = SparkSession.builder \
        .appName("CSDS312-Spotify-Lyrics-Analysis") \
        .master("local[8]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "50") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark




def load_and_clean_data(spark, data_path):
    """Load and clean YOUR dataset."""
    print("=" * 60)
    print("STEP 1: Loading and Cleaning Data")
    print("=" * 60)
    
    start = time.time()
    
    # Load data
    df = spark.read.csv(data_path, header=True, inferSchema=True)



    numeric_columns = {
        "Popularity": "double",
        "Energy": "double",
        "Danceability": "double",
        "Positiveness": "double",
        "Speechiness": "double",
        "Liveness": "double",
        "Acousticness": "double",
        "Instrumentalness": "double",
        "Tempo": "double",
        "Key": "integer",
        "Time signature": "integer",
        "Length": "double",
        "Loudness (db)": "double",
    }
    
    for col_name, col_type in numeric_columns.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(col_type))


    initial = df.count()
    print(f"Initial rows: {initial:,}")
    
    # Print schema to verify
    print("\nDataset columns:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Remove rows with null lyrics
    df = df.filter(
        F.col(COL_LYRICS).isNotNull() & 
        (F.length(F.col(COL_LYRICS)) > 0)
    )
    
    # Remove rows with null audio features
    critical_cols = [COL_POPULARITY, COL_DANCEABILITY, COL_ENERGY, 
                     COL_VALENCE, COL_TEMPO]
    for col in critical_cols:
        if col in df.columns:
            df = df.filter(F.col(col).isNotNull())
    
    # Extract year by finding 4-digit pattern (works for all date formats!)
    df = df.withColumn(
        "release_year",
        F.regexp_extract(F.col(COL_RELEASE_DATE), r"(\d{4})", 1).cast("integer")
    )
    
    # Filter out invalid years (before recordings existed or future)
    df = df.filter(
        (F.col("release_year") >= 1900) & 
        (F.col("release_year") <= 2025)
    )
    
    # Create decade
    df = df.withColumn(
        "decade",
        F.concat(
            (F.floor(F.col("release_year") / 10) * 10).cast("integer").cast("string"),
            F.lit("s")
        )
    )
    
    # Create popularity categories
    df = df.withColumn(
        "popularity_level",
        F.when(F.col(COL_POPULARITY) >= 80, "super_hit")
         .when(F.col(COL_POPULARITY) >= 60, "hit")
         .when(F.col(COL_POPULARITY) >= 40, "moderate")
         .when(F.col(COL_POPULARITY) >= 20, "low")
         .otherwise("flop")
    )
    
    # Create hit flag
    df = df.withColumn(
        "is_hit",
        F.when(F.col(COL_POPULARITY) >= 60, 1).otherwise(0)
    )
    
    # Clean lyrics text
    df = df.withColumn(
        "lyrics_clean",
        F.lower(F.col(COL_LYRICS))
    )
    
    # Remove special characters from lyrics
    df = df.withColumn(
        "lyrics_clean",
        F.regexp_replace(F.col("lyrics_clean"), r"[^a-zA-Z\s]", " ")
    )


    #regData['cleanedLyrics'] = (
        #regData['Lyrics']
        #.str.lower()    #puts them all in lowercase
        #.str.replace(r"\[.*?\]", "", regex=True)     #removes everythign inside bracketes [ ]
        #.str.replace(r"(chorus|verse|bridge|intro|outro|featuring)\s*\d*:", "", regex=True)     #remove any other 'chores', 'verse', etc. from the lyrics
        #.str.replace(r"[^a-z\s]", "", regex=True)      #strips so only letters a-z and spaces are included
        #.str.replace(r"\s+", " ", regex=True)          #remove extra spaces
        #.str.strip()                                   #remove leading/trailing spaces
    #)


    
    # Remove extra spaces
    df = df.withColumn(
        "lyrics_clean",
        F.trim(F.regexp_replace(F.col("lyrics_clean"), r"\s+", " "))
    )
    
    # Calculate lyrics length
    df = df.withColumn(
        "lyrics_word_count",
        F.size(F.split(F.col("lyrics_clean"), " "))
    )
    
    # Create sentiment column
    df = df.withColumn(
        "sentiment",
        F.when((F.col(COL_VALENCE) > 70) & (F.col(COL_ENERGY) > 70), "energetic_happy")
         .when((F.col(COL_VALENCE) > 70) & (F.col(COL_ENERGY) <= 70), "calm_happy")
         .when((F.col(COL_VALENCE) <= 40) & (F.col(COL_ENERGY) > 70), "energetic_sad")
         .when((F.col(COL_VALENCE) <= 40) & (F.col(COL_ENERGY) <= 70), "calm_sad")
         .otherwise("neutral")
    )
    
    # Convert explicit to readable format
    df = df.withColumn(
        "is_explicit",
        F.when(F.col(COL_EXPLICIT).isin("Yes", "TRUE", "1", 1), True).otherwise(False)
    )
    
    # Cache the dataframe
    df.cache()
    
    final = df.count()
    print(f"\nFinal rows: {final:,}")
    print(f"Removed: {initial - final:,}")
    print(f"Unique artists: {df.select(COL_ARTIST).distinct().count():,}")
    print(f"Time: {time.time() - start:.2f}s")
    
    # Show sample
    print("\nSample data:")
    df.select(COL_SONG, COL_ARTIST, COL_POPULARITY, COL_GENRE, 
              "release_year", "decade").show(5, truncate=False)
    
    return df





def lyrics_analysis(spark, df):
    """
    STEP 2: Analyze lyrics using TF-IDF (Parallel Text Mining)
    This is the core parallel processing demonstration for your project.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Lyrics Text Mining with TF-IDF")
    print("=" * 60)
    
    start = time.time()
    
    # Tokenize lyrics
    print("Tokenizing lyrics...")
    tokenizer = Tokenizer(inputCol="lyrics_clean", outputCol="tokens_raw")
    df = tokenizer.transform(df)
    
    # Remove stop words
    print("Removing stop words...")
    remover = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens_clean")
    df = remover.transform(df)
    
    # Calculate token statistics
    df = df.withColumn("token_count", F.size(F.col("tokens_clean")))
    df = df.withColumn("unique_tokens", F.size(F.array_distinct(F.col("tokens_clean"))))
    
    # Filter out songs with very few tokens
    df = df.filter(F.col("token_count") >= 10)
    
    # Create TF-IDF vectors
    print("Computing TF-IDF vectors...")
    cv = CountVectorizer(
        inputCol="tokens_clean",
        outputCol="raw_features",
        vocabSize=10000,
        minDF=5  # Minimum document frequency
    )
    cv_model = cv.fit(df)
    df = cv_model.transform(df)
    
    # Calculate IDF
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    idf_model = idf.fit(df)
    df = idf_model.transform(df)
    
    vocabulary = cv_model.vocabulary
    print(f"Vocabulary size: {len(vocabulary):,}")
    print(f"Average tokens per song: {df.select(F.avg('token_count')).collect()[0][0]:.1f}")
    print(f"Average unique tokens per song: {df.select(F.avg('unique_tokens')).collect()[0][0]:.1f}")
    
    # Cache TF-IDF results
    df.cache()
    
    print(f"TF-IDF computation time: {time.time() - start:.2f}s")
    
    return df, cv_model, vocabulary

def topic_modeling(spark, df, cv_model, vocabulary):
    """
    STEP 3: Topic Modeling with LDA
    Discover latent themes in song lyrics.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Topic Discovery with LDA")
    print("=" * 60)
    
    start = time.time()
    
    num_topics = 10
    print(f"Training LDA with {num_topics} topics...")
    
    lda = LDA(
        featuresCol="raw_features",
        k=num_topics,
        maxIter=50,
        seed=42,
        optimizer="online"
    )
    
    lda_model = lda.fit(df)
    
    # Display discovered topics
    print("\n🎵 DISCOVERED LYRICAL THEMES 🎵")
    print("=" * 40)
    
    topics = lda_model.describeTopics(maxTermsPerTopic=10)
    
    topic_descriptions = []
    for row in topics.collect():
        topic_id = row.topic
        term_indices = row.termIndices
        term_weights = row.termWeights
        
        terms = [vocabulary[idx] for idx in term_indices if idx < len(vocabulary)]
        
        print(f"\nTopic {topic_id + 1}:")
        for term, weight in zip(terms[:8], term_weights[:8]):
            bar = "█" * int(weight * 100)
            print(f"  {term:<15} {weight:.4f} {bar}")
        
        topic_descriptions.append({
            'topic_id': topic_id,
            'top_words': ', '.join(terms[:5])
        })
    
    # Transform to get topic distributions
    df = lda_model.transform(df)
    
    print(f"\nTopic modeling time: {time.time() - start:.2f}s")
    
    return df, lda_model, topic_descriptions

def analyze_lyrics_by_popularity(df, vocabulary):
    """
    STEP 4: Compare lyrics of hits vs non-hits
    What words make a song popular?
    """
    print("\n" + "=" * 60)
    print("STEP 4: Hit Song Lyrics Analysis")
    print("=" * 60)
    
    start = time.time()
    
    # Get average TF-IDF for hit vs non-hit songs
    print("\nWords that distinguish hit songs:")
    
    # This would normally require more complex aggregation
    # For demonstration, we'll analyze word frequency differences
    
    # Top genres for hits
    print("\nHit rate by genre:")
    df.groupBy(COL_GENRE) \
      .agg(
          F.count("*").alias("total"),
          F.round(F.sum("is_hit") / F.count("*") * 100, 2).alias("hit_rate"),
          F.round(F.avg(COL_POPULARITY), 2).alias("avg_popularity")
      ) \
      .filter(F.col("total") > 30) \
      .orderBy(F.desc("hit_rate")) \
      .show(10, truncate=False)
    
    # Lyrics length vs popularity
    print("\nLyrics length analysis:")
    df = df.withColumn(
        "lyrics_length_category",
        F.when(F.col("lyrics_word_count") < 100, "short")
         .when(F.col("lyrics_word_count") < 200, "medium")
         .when(F.col("lyrics_word_count") < 300, "long")
         .otherwise("very_long")
    )
    
    df.groupBy("lyrics_length_category") \
      .agg(
          F.count("*").alias("count"),
          F.round(F.avg(COL_POPULARITY), 2).alias("avg_popularity"),
          F.round(F.sum("is_hit") / F.count("*") * 100, 2).alias("hit_rate")
      ) \
      .orderBy("lyrics_length_category") \
      .show()
    
    print(f"Lyrics analysis time: {time.time() - start:.2f}s")
    
    return df

def audio_feature_analysis(df):
    """
    STEP 5: Analyze audio features and predict hits
    """
    print("\n" + "=" * 60)
    print("STEP 5: Audio Feature Analysis & Hit Prediction")
    print("=" * 60)
    
    start = time.time()

    #fix - cast popularity to double
    #df = df.withColumn(COL_POPULARITY, F.col(COL_POPULARITY).cast("double"))


    spark = SparkSession.builder.getOrCreate()
    spark.catalog.clearCache()
    
    # Also drop the heavy columns we no longer need
    cols_to_drop = [c for c in df.columns if c in [
        "lyrics_clean", "tokens_raw", "tokens_clean", 
        "raw_features", "tfidf_features", "topicDistribution"
    ]]
    df = df.drop(*cols_to_drop)



    
    # Correlations with popularity
    features = [COL_DANCEABILITY, COL_ENERGY, COL_VALENCE, COL_ACOUSTICNESS,
                COL_INSTRUMENTALNESS, COL_LIVENESS, COL_SPEECHINESS, COL_TEMPO]
    
    print("\nFeature correlations with popularity:")
    correlations = []
    for f in features:
        if f in df.columns:
            corr = df.stat.corr(f, COL_POPULARITY)
            correlations.append((f, corr))
            direction = "+" if corr > 0 else ""
            print(f"  {f:<25} {direction}{corr:.4f}")
    
    # Feature comparison: Hit vs Non-Hit
    print("\nAudio Features: Hit vs Non-Hit Comparison:")
    for f in features:
        if f in df.columns:
            stats = df.groupBy("is_hit").agg(F.avg(f).alias("avg")).orderBy("is_hit").collect()
            if len(stats) == 2:
                non_hit = stats[0]["avg"]
                hit = stats[1]["avg"]
                diff = hit - non_hit
                arrow = "↑" if diff > 0 else "↓"
                print(f"  {f:<25} Non-Hit: {non_hit:.2f} | Hit: {hit:.2f} ({arrow}{abs(diff):.2f})")
    
    # Train Random Forest classifier
    print("\nTraining Hit Song Predictor...")
    
    feature_cols = [c for c in features if c in df.columns]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="ml_features")
    df = assembler.transform(df)
    
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    print("data split ready")
    
    rf = RandomForestClassifier(
        featuresCol="ml_features",
        labelCol="is_hit",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    print("random forest classifier constructed")

    model = rf.fit(train_df)
    print("model fit")
    predictions = model.transform(test_df)
    
    # Evaluate
    print("running inference ...")
    evaluator = BinaryClassificationEvaluator(labelCol="is_hit")
    auc = evaluator.evaluate(predictions)
    
    correct = predictions.filter(F.col("is_hit") == F.col("prediction")).count()
    total = predictions.count()
    
    print(f"Model Performance:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy: {correct/total:.4f}")
    
    # Feature importance
    print("\nMost Important Features for Predicting Hits:")
    importances = model.featureImportances.toArray()
    feature_imp = list(zip(feature_cols, importances))
    feature_imp.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, imp) in enumerate(feature_imp, 1):
        bar = "█" * int(imp * 100)
        print(f"  {i}. {feature:<25} {imp:.4f} {bar}")
    
    print(f"Audio analysis time: {time.time() - start:.2f}s")
    
    return model, predictions





def cluster_songs(df):
    """
    STEP 6: Cluster songs to find song archetypes
    """
    print("\n" + "=" * 60)
    print("STEP 6: Song Clustering")
    print("=" * 60)
    
    start = time.time()
    
    feature_cols = [COL_DANCEABILITY, COL_ENERGY, COL_VALENCE, 
                    COL_ACOUSTICNESS, COL_SPEECHINESS]
    available_cols = [c for c in feature_cols if c in df.columns]
    
    # Assemble and scale
    assembler = VectorAssembler(inputCols=available_cols, outputCol="cluster_raw")
    scaler = StandardScaler(inputCol="cluster_raw", outputCol="cluster_features",
                            withStd=True, withMean=True)
    
    df = assembler.transform(df)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    # K-Means clustering
    k = 6
    print(f"Clustering into {k} song archetypes...")
    kmeans = KMeans(featuresCol="cluster_features", k=k, seed=42, maxIter=100)
    model = kmeans.fit(df)
    df = model.transform(df)
    
    # Analyze clusters
    print("\n📊 SONG ARCHETYPES DISCOVERED:")
    cluster_analysis = df.groupBy("prediction") \
        .agg(
            F.count("*").alias("size"),
            F.round(F.avg(COL_POPULARITY), 1).alias("avg_popularity"),
            F.round(F.avg(COL_DANCEABILITY), 1).alias("danceability"),
            F.round(F.avg(COL_ENERGY), 1).alias("energy"),
            F.round(F.avg(COL_VALENCE), 1).alias("positiveness"),
            F.round(F.avg(COL_TEMPO), 0).alias("tempo"),
            F.round(F.sum("is_hit") / F.count("*") * 100, 1).alias("hit_rate")
        ) \
        .orderBy("prediction")
    
    cluster_analysis.show(truncate=False)
    
    # Show songs from best cluster
    best_cluster = cluster_analysis.orderBy(F.desc("avg_popularity")).first()["prediction"]
    
    print(f"\nTop songs from Best Archetype (Cluster {best_cluster}):")
    df.filter(F.col("prediction") == best_cluster) \
      .select(COL_SONG, COL_ARTIST, COL_POPULARITY, COL_GENRE) \
      .orderBy(F.desc(COL_POPULARITY)) \
      .show(10, truncate=False)
    
    print(f"Clustering time: {time.time() - start:.2f}s")
    
    return df, model

#def save_results(df, output_path, topic_descriptions):
    #"""
    #STEP 7: Save all analysis results
    #"""
    #print("\n" + "=" * 60)
    #print("STEP 7: Saving Results")
    #print("=" * 60)
    #
    ## Summary statistics
    #print("\nGenerating summary statistics...")
    #
    ## 1. Decade summary
    #decade_summary = df.groupBy("decade") \
        #.agg(
            #F.count("*").alias("track_count"),
            #F.round(F.avg(COL_POPULARITY), 1).alias("avg_popularity"),
            #F.round(F.avg(COL_DANCEABILITY), 1).alias("avg_danceability"),
            #F.round(F.avg(COL_ENERGY), 1).alias("avg_energy"),
            #F.round(F.avg(COL_VALENCE), 1).alias("avg_valence"),
            #F.round(F.sum("is_hit") / F.count("*") * 100, 1).alias("hit_rate_pct")
        #).orderBy("decade")
    #
    #decade_summary.toPandas().to_csv(f"{output_path}/decade_trends.csv", index=False)
    #print(f"✓ Decade trends saved")
    #
    ## 2. Genre summary
    #genre_summary = df.groupBy(COL_GENRE) \
        #.agg(
            #F.count("*").alias("tracks"),
            #F.round(F.avg(COL_POPULARITY), 1).alias("avg_popularity"),
            #F.round(F.sum("is_hit") / F.count("*") * 100, 1).alias("hit_rate")
        #).orderBy(F.desc("hit_rate"))
    #
    #genre_summary.toPandas().to_csv(f"{output_path}/genre_analysis.csv", index=False)
    #print(f"✓ Genre analysis saved")
    #
    ## 3. Topics discovered
    #import json
    #with open(f"{output_path}/lyrical_themes.json", "w") as f:
        #json.dump(topic_descriptions, f, indent=2)
    #print(f"✓ Lyrical themes saved")
    #
    ## 4. Feature comparison
    #features = [COL_DANCEABILITY, COL_ENERGY, COL_VALENCE, COL_ACOUSTICNESS,
                #COL_INSTRUMENTALNESS, COL_LIVENESS, COL_SPEECHINESS, COL_TEMPO]
    #
    #feature_comparison = []
    #for f in features:
        #if f in df.columns:
            #stats = df.groupBy("is_hit").agg(F.avg(f).alias("avg")).orderBy("is_hit").collect()
            #if len(stats) == 2:
                #feature_comparison.append({
                    #'feature': f,
                    #'non_hit_avg': stats[0]["avg"],
                    #'hit_avg': stats[1]["avg"],
                    #'difference': stats[1]["avg"] - stats[0]["avg"]
                #})
    #
    #import pandas as pd
    #pd.DataFrame(feature_comparison).to_csv(
        #f"{output_path}/hit_feature_comparison.csv", index=False
    #)
    #print(f"✓ Feature comparison saved")
    #
    #print(f"\nAll results saved to: {output_path}")
#

def save_results(df, output_path, topic_descriptions):
    """
    STEP 7: Save all analysis results (no pandas)
    """
    print("\n" + "=" * 60)
    print("STEP 7: Saving Results")
    print("=" * 60)
    
    os.makedirs(output_path, exist_ok=True)
    
    # 1. Decade summary
    decade_summary = df.groupBy("decade") \
        .agg(
            F.count("*").alias("track_count"),
            F.round(F.avg("Popularity"), 1).alias("avg_popularity"),
            F.round(F.avg("Danceability"), 1).alias("avg_danceability"),
            F.round(F.avg("Energy"), 1).alias("avg_energy"),
            F.round(F.avg("Positiveness"), 1).alias("avg_valence"),
            F.round(F.sum("is_hit") / F.count("*") * 100, 1).alias("hit_rate_pct")
        ).orderBy("decade")
    
    decade_summary.coalesce(1).write.mode("overwrite") \
        .option("header", "true").csv(f"{output_path}/decade_trends")
    print("  Decade trends saved")
    
    # 2. Genre summary
    genre_summary = df.groupBy("Genre") \
        .agg(
            F.count("*").alias("tracks"),
            F.round(F.avg("Popularity"), 1).alias("avg_popularity"),
            F.round(F.sum("is_hit") / F.count("*") * 100, 1).alias("hit_rate")
        ).orderBy(F.desc("hit_rate"))
    
    genre_summary.coalesce(1).write.mode("overwrite") \
        .option("header", "true").csv(f"{output_path}/genre_analysis")
    print("  Genre analysis saved")
    
    # 3. Topics as text file
    with open(f"{output_path}/lyrical_themes.txt", "w") as f:
        for t in topic_descriptions:
            f.write(f"Topic {t['topic_id']}: {t['top_words']}\n")
    print("  Lyrical themes saved")
    
    # 4. Feature comparison
    features = ["Danceability", "Energy", "Positiveness", "Acousticness",
                "Instrumentalness", "Liveness", "Speechiness", "Tempo"]
    
    with open(f"{output_path}/hit_feature_comparison.txt", "w") as f:
        f.write("Feature,Non-Hit Avg,Hit Avg,Difference\n")
        for feat in features:
            if feat in df.columns:
                try:
                    hit_avg = df.filter("is_hit == 1").agg(F.avg(feat)).collect()[0][0]
                    non_hit_avg = df.filter("is_hit == 0").agg(F.avg(feat)).collect()[0][0]
                    diff = hit_avg - non_hit_avg
                    f.write(f"{feat},{non_hit_avg:.2f},{hit_avg:.2f},{diff:+.2f}\n")
                except:
                    pass
    print("  Feature comparison saved")
    
    print(f"\n  All results saved to: {output_path}")







def print_project_summary(df, start_time):
    """Print a nice summary for the project."""
    print("\n" + "=" * 60)
    print("🎵 PROJECT SUMMARY: WHAT MAKES A HIT SONG? 🎵")
    print("=" * 60)
    
    total = df.count()
    hits = df.filter(F.col("is_hit") == 1).count()
    
    print(f"\nDataset: {total:,} songs analyzed in parallel using Spark")
    print(f"Hit songs: {hits:,} ({hits/total*100:.1f}%)")
    
    # Top findings
    print(f"\n📊 KEY FINDINGS:")
    
    # Best genre
    best_genre = df.groupBy(COL_GENRE) \
        .agg(F.round(F.sum("is_hit") / F.count("*") * 100, 1).alias("hit_rate")) \
        .orderBy(F.desc("hit_rate")).first()
    print(f"  Best genre: {best_genre[COL_GENRE]} ({best_genre['hit_rate']}% hit rate)")
    
    # Optimal features
    features = [COL_DANCEABILITY, COL_ENERGY, COL_VALENCE]
    for f in features:
        hit_avg = df.filter(F.col("is_hit") == 1).agg(F.avg(f)).collect()[0][0]
        non_hit_avg = df.filter(F.col("is_hit") == 0).agg(F.avg(f)).collect()[0][0]
        direction = "higher" if hit_avg > non_hit_avg else "lower"
        print(f"  Hit songs have {direction} {f} ({hit_avg:.1f} vs {non_hit_avg:.1f})")
    
    elapsed = time.time() - start_time
    print(f"\n⏱ Total execution time: {elapsed/60:.1f} minutes")
    print(f"⚡ Processing speed: {total/elapsed:.0f} songs/second")

def main():
    """Main execution pipeline."""
    total_start = time.time()
    
    print("\n" + "🎵" * 30)
    print("CSDS312 - PARALLEL SONG ANALYSIS WITH PYSPARK")
    print("Using: TF-IDF, LDA Topic Modeling, K-Means Clustering, Random Forest")
    print("🎵" * 30 + "\n")
    
    spark = create_spark_session()
    
    try:
        # Step 1: Load and clean
        df = load_and_clean_data(spark, DATA_PATH)
        
        # Step 2: TF-IDF Lyrics Analysis
        df, cv_model, vocabulary = lyrics_analysis(spark, df)
        
        # Step 3: LDA Topic Modeling
        df, lda_model, topic_descriptions = topic_modeling(spark, df, cv_model, vocabulary)
        
        # Step 4: Lyrics popularity analysis
        df = analyze_lyrics_by_popularity(df, vocabulary)
        
        # Step 5: Audio feature analysis & ML
        model, predictions = audio_feature_analysis(df)
        
        # Step 6: Clustering
        df, cluster_model = cluster_songs(df)
        
        # Step 7: Save results
        save_results(df, OUTPUT_PATH, topic_descriptions)
        
        # Print summary
        print_project_summary(df, total_start)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        spark.stop()
        print("\n✓ Spark session stopped")

if __name__ == "__main__":
    main()
