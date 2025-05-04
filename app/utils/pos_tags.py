import spacy

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='./logs/pos_tag.log'
)

class PosTagger:
    """
    A pos (Part-of-Speech) tagger using spaCy.
    """
    
    def __init__(self, model_name="en_core_web_sm"):
        """
        Initializes the PosTagger with a specified spaCy language model.
        
        Args:
            model_name (str, optional): The name of the language model to load.
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
    
    def get_pos_tags(self, text: str) -> list[tuple[str, str]]:
        """
        Performs POS tagging on the input string using the loaded spaCy model.
        
        Args:
            text (str): The input string to be tagged.
            
        Returns:
            list[tupel[str, str]]: A lsit of (tokens, POS tag) tupels. Returns an empty list
                                    if the input in None or an empty string.
            
        Raises:
            TypeError: If the input is not a string.
        """
        if text is None:
            return []
        
        if not isinstance(text, str):
            logging.error(f"Input must be a stirng, but received: {type(text)}")
            raise TypeError(f"Input must be a stirng, but received: {type(text)}")
        
        if not text.strip():
            return []
        
        try:
            doc = self.nlp(text)
            pos_tags = [(token.text, token.tag_) for token in doc]
            return pos_tags
        except Exception as e:
            logging.info(f"An erro occured during POS taggin: {e}")
            return []