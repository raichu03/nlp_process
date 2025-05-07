import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='./logs/steamming.log'
)

class Stemmer:
    """
    A stemmer using NLTK with a specified stemming algorithm.
    """
    
    def __init__(self, stemmer_type='porter'):
        """
        Initializes the Stemmer with a specified stemming algorithm.
        
        Args:
            stemmer_type (str, optional): The type of stemmer to use.
            Defaults to "porter". Options are "porter" or "snowball".
        
        Raises:
            ValueError: If an invalid stemmer type is provided.
        """
        
        if stemmer_type.lower() not in ["porter", "snowball"]:
            logging.error(f"Invalid stemmer type: {stemmer_type}. Must be 'porter' or 'snowball'.")
            raise ValueError(f"Invalid stemmer type: {stemmer_type}. Must be 'porter' or 'snowball'.")
        self.stemmer_type = stemmer_type.lower()
        
        if self.stemmer_type == 'porter':
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = SnowballStemmer("english")
    
    def stem(self, text: str) -> list[str]:
        """
        Stems the input string using the selected NLTK stemmer.
        
        Args:
            text (str): The input string too be stemmed.
            
        Returns:
            list[str]: A list of stemmed words. Returns an empty list if the input is
                        None or an empty string.
        
        Raises:
            TypeError: If the input is not a string.
        """
        
        if text is None:
            return []
        
        if not isinstance(text, str):
            logging.error(f"Input must be a string, but received: {type(text)}")
            raise TypeError(f"Input must be a string, but received: {type(text)}")
        
        if not text.strip():
            return []
        
        try:
            tokens = word_tokenize(text)
            stemmed_words = [self.stemmer.stem(word) for word in tokens]
            return stemmed_words
        except Exception as e:
            logging.info(f"An error occured during stemming: {e}")
            return []