
import torch
import spacy
from transformers import T5Tokenizer

def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    decoder_input_ids = tokenizer("summarize:", return_tensors="pt").input_ids
    
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
        hidden_states = outputs.encoder_hidden_states
        last_layer_hidden_state = hidden_states[-1]
        embeddings = last_layer_hidden_state[:, 0, :]  # First token embedding
    return embeddings

def extract_sentences(document, nlp = spacy.load('en_core_web_md'), min_words = 4):
    """
    Extracts sentences from a document, filtering out sentences with 4 or fewer words.

    Args:
        document (str): The text from which to extract sentences.
        min_words (int): The minimum number of words a sentence must have to be included.

    Returns:
        list: A list of sentences with more than `min_words` words.
    """
    # Process the document
    doc = nlp(document)

    # Extract sentences and filter out those with 4 or fewer words
    sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > min_words]

    return sentences

def compute_weights(recentness, decay_rate):
    """
    Compute weight as e^(lambda * t) for a sequence based on recentness using PyTorch.

    Args:
        recentness (int or float): The position in the sequence.
        decay_rate (float): The rate of decay (lambda). Higher values increase weights faster.

    Returns:
        torch.Tensor: The weight for the given recentness.
    """
    # Convert recentness and decay_rate to tensors if they are not already
    recentness = torch.tensor(recentness, dtype=torch.float32)
    decay_rate = torch.tensor(decay_rate, dtype=torch.float32)
    
    weight = torch.exp(- decay_rate * recentness)
    return weight
