import torch
from sentence_transformers import SentenceTransformer
import spacy
# Load the model 
sbert_model_name = 'all-MiniLM-L6-v2'
sbert_model = SentenceTransformer(sbert_model_name)

# Define the functions

def get_embeddings(text, model= sbert_model):
    embedding = model.encode(text, convert_to_tensor = True)
    return embedding

    """
    # Generate sentence embeddings using sbert_model and tokenizer.

    # Args:
    #     text (str): The input text for which to compute embeddings.
    #     model (Engine): The DeepSparse model engine (default is the loaded engine).

    # Returns:
    #     torch.Tensor: Sentence embedding vector.
    # """
    # # Tokenize the input text with max_length=128 to match model's expected input size
    # tokens = tokenizer(text, return_tensors='np', padding='max_length', truncation=True, max_length=128)
    
    # # Convert input_ids and attention_mask to numpy arrays with int64 dtype
    # input_ids = np.array(tokens['input_ids'], dtype=np.int64)
    # attention_mask = np.array(tokens['attention_mask'], dtype=np.int64)

    # # Pass both input_ids and attention_mask to the model
    # embeddings = model([input_ids, attention_mask])
    
    # # Convert the embedding result to a torch tensor (first element of the first batch)
    # embedding = torch.tensor(embeddings[0][0])

    # return embedding



nlp = spacy.load('en_core_web_md')

def extract_sentences(document, min_words=4):
    """
    Extracts sentences from a document, filtering out sentences with min_words or fewer.

    Args:
        document (str): The text from which to extract sentences.
        nlp: The SpaCy NLP model to use (default: en_core_web_md).
        min_words (int): The minimum number of words a sentence must have to be included.

    Returns:
        list: A list of sentences with more than `min_words` words.
    """
    # if nlp is None:
    #     nlp = spacy.load('en_core_web_md')
        
    # Process the document
    doc = nlp(document)

    # Extract sentences and filter out those with min_words or fewer
    sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > min_words]

    return sentences

def compute_rmse(embedding1, embedding2):
    """
    Compute the RMSE between two embeddings.

    Args:
        embedding1 (torch.Tensor): The first embedding tensor.
        embedding2 (torch.Tensor): The second embedding tensor.

    Returns:
        torch.Tensor: RMSE score between the embeddings.
    """
    # Compute RMSE (Root Mean Square Error)
    rmse = torch.sqrt(torch.mean((embedding1 - embedding2) ** 2))
    return rmse

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
    
    weight = torch.exp(-decay_rate * recentness)
    return weight
