import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the dataset
file_path = 'British_Airway_Review.csv'
data = pd.read_csv(file_path)

# Extract the 'reviews' column (which contains the text data)
reviews = data['reviews'].dropna()  # Dropping any missing values

# Get the list of stopwords from NLTK
stop_words = set(stopwords.words('english'))

# Function to tokenize, clean the text, and remove stopwords
def tokenize(text):
    # Convert to lowercase, remove non-alphabetical characters, and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    # Remove stopwords using NLTK
    words = [word for word in words if word not in stop_words]
    return words

# Tokenize the entire dataset
tokenized_reviews = reviews.apply(tokenize)

# Build the vocabulary (unique words without stopwords)
vocabulary = sorted(set(word for review in tokenized_reviews for word in review))

# Create an empty term-document matrix
term_document_matrix = []

# Populate the term-document matrix (full matrix)
for review in tokenized_reviews:
    word_count = Counter(review)
    row = [word_count.get(word, 0) for word in vocabulary]  # Create a row for each document
    term_document_matrix.append(row)

# Convert the term-document matrix to a DataFrame
tdm_df = pd.DataFrame(term_document_matrix, columns=vocabulary)

# Save the full term-document matrix to a CSV file
tdm_df.to_csv('term_document_matrix.csv', index=False)
print("Full term-document matrix saved to 'term_document_matrix.csv'.")

# Display a sample of the full term-document matrix
print("Term-Document Matrix (Sample):")
print(tdm_df.head())

# Count the frequency of each word in the entire dataset
total_word_count = Counter(word for review in tokenized_reviews for word in review)

# Get the top 100 terms by frequency
top_100_terms = [word for word, count in total_word_count.most_common(100)]

# Create an empty term-document matrix for the top 100 terms
term_document_matrix_top_100 = []

# Populate the term-document matrix for the top 100 terms
for review in tokenized_reviews:
    word_count = Counter(review)
    row = [word_count.get(word, 0) for word in top_100_terms]  # Create a row for each document
    term_document_matrix_top_100.append(row)

# Convert the partial term-document matrix to a DataFrame
tdm_top_100_df = pd.DataFrame(term_document_matrix_top_100, columns=top_100_terms)

# Save the partial term-document matrix to a CSV file
tdm_top_100_df.to_csv('term_document_matrix_top_100.csv', index=False)
print("Partial term-document matrix with top 100 terms saved to 'term_document_matrix_top_100.csv'.")

# Display a sample of the partial term-document matrix
print("Partial Term-Document Matrix for Top 100 Terms (Sample):")
print(tdm_top_100_df.head())

# Display the top 100 terms
print("Top 100 Terms:")
print(top_100_terms)
