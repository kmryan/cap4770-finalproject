import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#dataset = pd.read_json('practice_dataset.json', lines = True)
#practice_set = dataset["short_description"][0:200]
#print(practice_set.index)

dataset = pd.read_pickle('n100.pkl')
practice_data = dataset['body_text'][0:100]


#print(practice_data[0])
sentences = practice_data[0].split('.')
practice_set = np.array(sentences)


model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
sentence_embeddings_bert = model.encode(practice_set)

g = input("Enter short description of article you are looking for: ")
query_embedded = model.encode(g)

def find_similar(vector_query, vector_total_set, k=1):
    similarity_matrix = cosine_similarity(vector_query, vector_total_set)
    np.fill_diagonal(similarity_matrix, 0)
    similarities = similarity_matrix[0]
    if k == 1:
        return [np.argmax(similarities)]
    elif k is not None:
        return np.flip(similarities.argsort()[-k:][::1])

similar_indexes = find_similar(query_embedded,sentence_embeddings_bert , 5)
print()
print("5 most similar descriptions using Sentence-Bert")
for num, index in enumerate(similar_indexes,1):
    print(str(num) + ". " + practice_set[index])
