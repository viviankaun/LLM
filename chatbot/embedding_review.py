
from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np
import pandas as pd
import streamlit as st

# Load and preprocess the data
df = pd.read_csv('product_reviews.csv')
#df = df.head(20)  # Load only the first 20 rows for this example

# Specify a writable cache directory
cache_dir = "."

# Initialize the SentenceTransformer model and specify the cache directory
# model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)

# Set up the database connection
conn = psycopg2.connect(
    dbname="llm16",
    user="postgres",
    password="P0stGr3sP@ss",
    host="postgres",  
    port="5432"
)
cur = conn.cursor()

# Drop the 'reviews' table if it already exists
cur.execute('DROP TABLE IF EXISTS reviews')
# Create the 'review' table if it doesn't already exist
cur.execute('''
    CREATE TABLE IF NOT EXISTS reviews (
        product_id INT,
        product_review TEXT
    )
''')

 
# Generate embeddings and store them in PostgreSQL
 
for product_review, product_id in zip(df['Review'], df['product_id']):
  
    
    # Insert the text content and the embedding vector into the items table
    cur.execute(
        'INSERT INTO reviews (product_id, product_review) VALUES (%s, %s)',
        (product_id, product_review)
    )
    
    # Write the content and embedding to the Streamlit app
    st.write(f"product_id: {product_id}")
    st.write(f"product_review: {product_review}")

# Commit the transaction and close the connection
conn.commit()
cur.close()
conn.close()

st.write("reivews  successfully inserted into the database!")