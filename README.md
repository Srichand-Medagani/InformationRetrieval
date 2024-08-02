# InformationRetrieval

I performed document clustering on a corpus of HTML 
documents using the agglomerative clustering algorithm with the complete link 
method. The goal was to identify similarities and dissimilarities between 
documents and determine the closest document to the corpus centroid.
Implementation
Step-1: Tokenization and Preprocessing:
The input HTML documents were tokenized using the simple_preprocess function 
from the gensim library.
I removed HTML tags using regular expressions and applied additional 
preprocessing steps such as converting tokens to lowercase and removing 
stopwords.
The resulting tokens were used as the input for clustering.
Step-2: Similarity Matrix Calculation:
I calculated the similarity between documents using the cosine similarity metric.
A similarity matrix was constructed to store pairwise similarities between 
documents.
The similarity matrix was implemented as a 2D numpy array, where each entry 
represented the similarity between two documents.
Step-3: Agglomerative Clustering:
The agglomerative clustering algorithm with the complete link method was 
employed.
Initially, each document was treated as a singleton cluster.
At each step, the two clusters/documents with the highest similarity were merged.
The merging process was recorded to track the clusters being merged.
The clustering process continued until no two clusters/documents had a similarity 
greater than the specified threshold.
threshold = 0.4
Step-4: Cluster Naming Convention:
To assign names to the clusters, the merging process is tracked. Each cluster is 
represented by the indices of the documents it contains. When two clusters are 
merged, the new cluster retains the indices of both clusters. This allows us to keep 
track of the document indices associated with each cluster throughout the 
clustering process.
First 100 lines of Output:

Method for Giving Names to Clusters: The names for the clusters are assigned 
based on the index of the representative document within each cluster. The 
representative document is assumed to be the first document in the cluster. The 
cluster names are derived by adding 1 to the index since indexing starts from 0. 
Therefore, the cluster names correspond to the document numbers in the corpus.
Implementation of the Similarity Matrix and Major Data Structures:
Similarity Matrix: The similarity matrix is implemented as a two-dimensional 
NumPy array named similarity_matrix. It is initialized with zeros and has 
dimensions (num_documents, num_documents) to store the pairwise similarity 
values between documents. The similarity matrix is calculated in the 
calculate_similarity_matrix function by iterating over the tokens of each document 
pair and calling the calculate_cosine_similarity function.
Data Structures:
tokens: It is a list of lists where each inner list represents the tokenized text of a 
document. It is generated by the generate_tokens function by reading the HTML 
files from the input directory, performing tokenization, and applying preprocessing 
steps.
clusters: It is a list of lists where each inner list represents a cluster. Initially, each 
document is considered as a separate cluster. The perform_clustering function 
merges clusters iteratively until the similarity falls below the threshold. The clusters
list is updated during the merging process.
merging_process: It is a list that keeps track of the merging steps during clustering. 
Each entry in the list represents the merging of two clusters/documents. The 
merging process is recorded by appending tuples of cluster indices to the 
merging_process list.
doc1_vector and doc2_vector: These are defaultdicts from the collections module 
used to store the term frequencies of tokens in two documents. They are created 
and populated in the calculate_cosine_similarity function.
Which pair of HTML documents is the most similar?
The pair of HTML documents that are the most similar can be determined by 
examining the merging process during clustering. The first pair of documents that 
are merged into a cluster indicate the highest similarity. By printing the merging 
process, we can identify the first merged clusters and determine the pair of 
documents that are most similar.
In this case the most similar pair of HTML documents are ([0],[12])
Which pair of documents is the most dissimilar?
The pair of documents that is the most dissimilar can be identified by examining 
the last entry in the merging process list. The last entry represents the final merging
step where all clusters/documents have been merged. Therefore, the last entry in 
the merging process list represents the pair of documents that is the most 
dissimilar.
In this case the most dissimilar pair of documents are ([285, 296], [285, 296])
Which document is the closest to the corpus centroid?
To determine the document closest to the corpus centroid, we need to compute 
the similarity between each document and the centroid. The centroid can be 
calculated as the average of the similarity matrix's rows or columns. We can find 
the document closest to the centroid by finding the row or column in the similarity 
matrix with the highest similarity value to the centroid.
In this case the document closest to the corpus centroid is 1.
Output:
Result:
In this analysis, I have performed document clustering on a corpus of HTML 
documents. The code successfully generates a similarity matrix, performs 
hierarchical agglomerative clustering using the complete link method, and provides 
information about the most similar, most dissimilar, and closest document to the 
corpus centroid. The implementation relies on tokenization, cosine similarity
calculation, and tracking the merging process to assign names to clusters
