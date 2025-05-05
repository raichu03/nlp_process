import numpy as np
import os
import logging

import gensim.downloader as api

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='./logs/embedding.log'
)

class GloveEmbeddingGenerator:
    """
    A class to load a GloVe model and generte word embeddings.
    """
    
    def __init__(self, model_name = 'glove-wiki-gigaword-100'):
        """
        Initializes the speified model.

        Args:
            model_name: The name of the GloVe model to load Defaults to 'glove-wiki-gigaword-100.
        """
        
        self.model_name = model_name
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """
        Loads the pre-trained GloVe model using gensim.
        """
        try:
            if self.model_name not in api.info()['models']:
                logging.error(f"Error: Model '{self.model_name}' not found in gensim's available models.")
                self._model = None
                return
            
            self._model = api.load(self.model_name)
        except Exception as e:
            logging.error(f"An error occured while loading the Glove model: {e}")
            self._model = None
           
    def get_word_embedding(self, word: str):
        """
        Gets the vector embedding for a given word from the loaded GloVe model.
        
        Args:
            word: The word for which to get the embedding.
            
        Returns:
            A numpy array representing the word embedding, or None if the word is 
            not in the vocabulary or the model is not loaded.
        """
        
        if self._model is None:
            logging.error("Error: Glove model is not loaded.")
            return None
        
        try:
            embedding = self._model[word]
            return embedding
        
        except KeyError:
            logging.warning(f"Warning: Word '{word}' not found in the vocabulary.")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occured while getting embedding for '{word}': {e}")
            return None
        
    def get_word_embeddings(self, words: list[str]):
        """
        Gets vector embeddings for a list of words.
        
        Args:
            wrods: A list of words for which to get embeddings.

        Returns:
            A dictionary where keys are words and values are their numpy array embeddings.
            Words not found in the vocabulary will not be included in the dictionary.
        """
        
        if self._model is None:
            logging.error("GloVe model is not loaded.")
            return {}
        
        embeddings = {}
        for word in words:
            embedding = self.get_word_embedding(word)
            if embedding is not None:
                embeddings[word] = embedding
        
        return embeddings