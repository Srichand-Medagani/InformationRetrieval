import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pandas as pd
from collections import defaultdict
from gensim.utils import simple_preprocess
import math

start= datetime.datetime.now()

def stopwords():
    with open(r"C:\Users\srich\OneDrive\Desktop\stopwords.txt",'r') as s:
        stop_words = s.read().split("\n")
    return stop_words

def generate_tokens(directory):
    tokens = []
    stop_words = stopwords()

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r',encoding='ISO-8859-1') as file:
            text = file.read()
            text = re.sub(r'<[^>]*>', '', text)
            tokenized_text = simple_preprocess(text)
            filtered_tokens = [token.lower() for token in tokenized_text if token.isalpha() and token not in stop_words and len(token) > 1]
            tokens.append(filtered_tokens)

    return tokens

def calculate_similarity_matrix(tokens):
    num_documents = len(tokens)
    similarity_matrix = np.zeros((num_documents, num_documents))

    for i in range(num_documents):
        for j in range(i + 1, num_documents):
            doc1 = tokens[i]
            doc2 = tokens[j]
            similarity = calculate_cosine_similarity(doc1, doc2)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    return similarity_matrix

def calculate_cosine_similarity(doc1, doc2):
    doc1_vector = defaultdict(int)
    doc2_vector = defaultdict(int)

    for token in doc1:
        doc1_vector[token] += 1

    for token in doc2:
        doc2_vector[token] += 1

    common_terms = set(doc1_vector.keys()) & set(doc2_vector.keys())
    dot_product = sum(doc1_vector[term] * doc2_vector[term] for term in common_terms)

    magnitude_doc1 = math.sqrt(sum(doc1_vector[term] ** 2 for term in doc1_vector.keys()))
    magnitude_doc2 = math.sqrt(sum(doc2_vector[term] ** 2 for term in doc2_vector.keys()))

    cosine_similarity = dot_product / (magnitude_doc1 * magnitude_doc2)
    return 1 - cosine_similarity

def perform_clustering(similarity_matrix, threshold):
    num_documents = similarity_matrix.shape[0]
    clusters = [[i] for i in range(num_documents)]
    merging_process = []

    while True:
        max_similarity = np.max(similarity_matrix)
        if max_similarity <= threshold:
            break

        i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        cluster_i = clusters[i]
        cluster_j = clusters[j]
        merged_cluster = cluster_i + cluster_j

        merging_process.append((cluster_i, cluster_j))

        similarity_matrix[i, :] = np.maximum(similarity_matrix[i, :], similarity_matrix[j, :])
        similarity_matrix[:, i] = np.maximum(similarity_matrix[:, i], similarity_matrix[:, j])
        similarity_matrix = np.delete(similarity_matrix, j, axis=0)
        similarity_matrix = np.delete(similarity_matrix, j, axis=1)

        clusters[i] = merged_cluster
        clusters.pop(j)

    return merging_process

# Input directories
input_directory = r"C:\Users\srich\OneDrive\Desktop\input_files"

# Generate tokens from input documents
tokens = generate_tokens(input_directory)

# Calculate similarity matrix
similarity_matrix = calculate_similarity_matrix(tokens)

# Set the similarity threshold for clustering
threshold = 0.4

# Perform clustering
merging_process = perform_clustering(similarity_matrix, threshold)

# Print the first 100 lines of merging process
for i in range(len(merging_process)):
    cluster_i, cluster_j = merging_process[i]
    print(f"Merging clusters {cluster_i} and {cluster_j}")


most_similar_pair = merging_process[0]

print("Most Similar Pair:",most_similar_pair)

most_dissimilar_pair = merging_process[-1]

print("Most Dissimilar Pair:",most_dissimilar_pair)

centroid_similarity = similarity_matrix.mean(axis=0)  # Calculate centroid similarity
closest_document_index = np.argmax(centroid_similarity)  # Find index of the closest document
closest_document = closest_document_index + 1  # Add 1 to match document numbering

print("Closet document to corpus centroid:",closest_document)

# Calculate and print the processing time
end = datetime.datetime.now()
processing_time = end - start
print(f"Processing time: {processing_time}")