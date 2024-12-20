from collections import deque
from itertools import islice
import torch
import pandas as pd
import random
import ast
import spacy 
from sentence_transformers import SentenceTransformer
from utils import get_embeddings, extract_sentences, compute_weights, compute_rmse


sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
# Load SpaCy model for sentence tokenization
nlp = spacy.load('en_core_web_md')

# Dictionary for caching embeddings of unique documents
embedding_cache = {}

def get_document_embedding(doc_id, doc_text):
    """Retrieve or compute the embedding of a document using sbert_model."""
    if doc_id not in embedding_cache:
        # Compute the embedding if it's not already in the cache
        embedding_cache[doc_id] = get_embeddings(doc_text, model= sbert_model)
    return embedding_cache[doc_id]

def perturbate_trajectory(row, news_df, summ_df, purturbed_summ_df, curr_summ_id, previous_doc_steps=10, decay_rate=0.3, perturbation_probability=0.8):
    
    print("-----------")
    # Before processing each row
    print(f"Row {row.name}: Docs column contains {type(row['Docs'])}")

    doc_list = ast.literal_eval(row['Docs']) 
    action_list = ast.literal_eval(row['Action'])
    assert len(doc_list) == len(action_list)

    history_embeddings_deque = deque()

    for doc_idx, doc in enumerate(doc_list):
        if action_list[doc_idx] == 'gen_summ':

            print('Doc ID', doc_list[doc_idx])

            # Query document and extract sentences
            query_document = news_df.loc[doc]['News body']
            sentences = extract_sentences(query_document)
            sentences_scores = [0] * len(sentences)

            # Generate sentence embeddings
            for sentence_idx, sentence in enumerate(sentences):
                sentence_embedding = get_embeddings(sentence , model = sbert_model)


                # Iterate over History Docs if any
                if len(history_embeddings_deque) > 0:
                    for recentness, doc_embedding in enumerate(islice(history_embeddings_deque, previous_doc_steps)):
                        # Replace cosine similarity with RMSE
                        sentences_scores[sentence_idx] += compute_rmse(sentence_embedding, doc_embedding) * compute_weights(recentness, decay_rate)

            tensor_array = torch.stack([torch.tensor(score, dtype=torch.float32).clone().detach() for score in sentences_scores])

            # Find the sentence with the highest score
            argmin_index = torch.argmin(tensor_array)

            # Choose whether to perturb or use existing summary
            perturbation_choice = random.choices([0, 1], weights=[1 - perturbation_probability, perturbation_probability], k=1)

            if perturbation_choice[0] == 1:
                # Add perturbed summary
                new_summary = pd.DataFrame([[f"S-{curr_summ_id}", doc_list[doc_idx], row['UserID'], sentences[argmin_index]]], 
                                        columns=['SummID', 'NewsID', 'UserID', 'Summary'])
                purturbed_summ_df = pd.concat([purturbed_summ_df, new_summary], ignore_index=True)
                doc_list[doc_idx + 1] = f"S-{curr_summ_id}"
                print(f"Added perturbed summary: {new_summary}")
            else:
                # Use existing summary
                existing_summary = summ_df.loc[doc_list[doc_idx + 1]]['Summary']
                new_summary = pd.DataFrame([[f'S-{curr_summ_id}', doc_list[doc_idx], row['UserID'], existing_summary]], 
                                        columns=['SummID', 'NewsID', 'UserID', 'Summary'])
                purturbed_summ_df = pd.concat([purturbed_summ_df, new_summary], ignore_index=True)
                doc_list[doc_idx + 1] = f'S{curr_summ_id}'
                print(f"Added existing summary: {existing_summary}")
        


            curr_summ_id += 1

        if action_list[doc_idx] == 'click':
            # Retrieve or compute document embedding (from cache if available)
            doc_embedding = get_document_embedding(doc, news_df.loc[doc]['News body'])
            history_embeddings_deque.appendleft(doc_embedding)

    return purturbed_summ_df, curr_summ_id, doc_list

# -----------------------------------------------------------------------------------------



