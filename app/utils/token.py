import spacy
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='../logs/token.log'
)

class Tokenizer:
    """
    A tokenizer using spaCy for robust text processing.
    """
    
    def __init__(self, model_name="en_core_web_sm"):
        """
        Initializes the Tokenizer with a specified spaCy language model.
        
        Args:
            model_name(str, optional): The name of the spaCy language model to load.
                                        Defaults to "en_core_web_sm" (English, small).
        
        Raises:
            OSError: If the specified spaCy model cannot be loaded.
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError as e:
            
            logging.error(
                f"Error loading spaCy model'{model_name}."
                f"Please ensure it is installed. You can install it by running: "
                f"`python -m spacy download {model_name}`"
            )
            
            raise OSError(
                f"Error loading spaCy model'{model_name}."
                f"Please ensure it is installed. You can install it by running: "
                f"`python -m spacy download {model_name}`"
            ) from e
        
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes the input string using the loaded spaCy model.
        
        Args:
            text (str): The input string to be tokenized.
        
        Returns:
            list[str]: A list of string tokens. Returns an empty list if the input
                        is None or an empty string.

        Raises:
            TypeError: If the input is not a string.
        """
        
        if text is None:
            return []
        if not isinstance(text,str):
            logging.error(f"Input must be a string, but received: {type(text)}")
            raise TypeError(f"Input must be a string, but received: {type(text)}")
        if not text.strip():
            return []
        
        try:
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
            return tokens
        except Exception as e:
            logging.info(f"An error occured during tokenization: {e}")
            return []