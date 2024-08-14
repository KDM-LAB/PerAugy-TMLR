import ast
import spacy
from collections import deque
from itertools import islice
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5Model
from utils import get_embeddings
from utils import extract_sentences
from utils import compute_weights

# Initialize model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5Model.from_pretrained(model_name)

# Load SpaCy model
nlp = spacy.load('en_core_web_md')

def perturbate_trajectory(row,news_df, summ_df ,previous_doc_steps=30, decay_rate=0.3):
    doc_list = ast.literal_eval(row['Docs'])
    action_list = ast.literal_eval(row['Action'])
    assert(len(doc_list) == len(action_list))

    history_embeddings_deque = deque()
    
    for doc_idx, doc in enumerate(doc_list):
        if action_list[doc_idx] == 'gen_summ':
            print('Doc ID', doc_list[doc_idx])

            # Query Doc
            query_document = news_df.loc[doc]['News body']
            sentences = extract_sentences(query_document, nlp=nlp)
            sentences_scores = [0] * len(sentences)
            
            # Iterate over Sentences
            for sentence_idx, sentence in enumerate(sentences):
                sentence_embedding = get_embeddings(sentence, model=model, tokenizer=tokenizer)
                
                # Iterate over History Docs
                for recentness, doc_embedding in enumerate(islice(history_embeddings_deque, previous_doc_steps)):
                    sentences_scores[sentence_idx] += F.cosine_similarity(sentence_embedding, doc_embedding, dim=1) * compute_weights(recentness, decay_rate) 

            tensor_array = torch.stack(sentences_scores)

            # Use torch.argmax to find the index of the maximum value
            argmax_index = torch.argmax(tensor_array)

            print(sentences[argmax_index])
            summ_df.at[doc_list[doc_idx + 1], 'PertubedSummary'] = sentences[argmax_index]
            
        if action_list[doc_idx] == 'click':
            doc_embedding = get_embeddings(news_df.loc[doc]['News body'], model=model, tokenizer=tokenizer)
            history_embeddings_deque.appendleft(doc_embedding)