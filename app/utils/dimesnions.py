import numpy as np
import logging
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='./logs/dimesnion.log'
)

class DimensionReduction:
    """
    A code to handle word embeddings, perform PCA dimensionality redutction,
    and export the results.
    """
    
    def __init__(self, embeddings_dict=None):
        """
        Initializes the DimensionReduction with a dictioniary of words and embeddings.
        
        Args:
            embeddings_dict (dict, optional): A dictionary where keys are words and values
                                                and values are lists or numpy arrays 
                                                representing their embeddings. Defaults to None.
        """
        self.embeddings_dict = embeddings_dict if embeddings_dict is not None else {}
        self.words = []
        self.embeddings = None
        self.reduced_embeddings = None
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Extracts words and converts embeddings to a NumPy array.
        Handles cases where embeddings might not be uniform in size.
        """
        if not self.embeddings_dict:
            logging.warning("No embeddings provided.")
            self.words = []
            self.embeddings = None
            return
        
        embedding_dims = [len(emb) for emb in self.embeddings_dict.values()]
        if not all(dim == embedding_dims[0] for dim in embedding_dims):
            logging.error("Embeddings have inconsistent dimesnions. Cannot proceed with PCA.")
            self.words = []
            self.embeddings = None
            return
        
        self.words = list(self.embeddings_dict.keys())
        ### Convert list of embeddings to a NumPy array. ### 
        try:
            self.embeddings = np.array(list(self.embeddings_dict.values()))
        except Exception as e:
            logging.error(f"Error converting embeddings to numpy array: {e}")
            self.words = []
            self.embeddings_dict = None

    def perform_pca(self, n_components: int = 2):
        """
        Performs Principal Component Analysis (PCA) on the embeddings.
        
        Args:
            n_components (int): The number of components to keep after PCA.
                                Defaluts to 2 (for 2D visualization).

        Returns:
            numpy.ndarray or None: The reduced embeddings if successful, None otherwise.
        """
        
        if self.embeddings is None or len(self.embeddings) == 0:
            logging.error("No valid embeddings data available for PCA.")
            return None
        
        if n_components > self.embeddings.shape[1]:
            logging(
                f"Error: n_componenets ({n_components}) cannot be greater than "
                f"the original dimension ({self.embeddings.shape[1]})."
            )
            return None
        
        try:
            pca = PCA(n_components=n_components)
            self.reduced_embeddings = pca.fit_transform(self.embeddings)
            return self.reduced_embeddings
        except Exception as e:
            logging.error(f"Error during PCA execution: {e}")
            return None
    
    def get_reduced_data(self):
        """
        Reduces the words and their corresponding embeddings.
        
        Retruns:
            list of dict or None: A list of dictionaries, where each dictionary
                                    contains 'word' and 'embeddings' (reduced) keys,
                                    or None if PCA has not been performed or failed.
        """
        if self.words is None or self.reduced_embeddings is None:
            logging.error("PCA has not been successfully performed.")
            return None
        
        if len(self.words) != len(self.reduced_embeddings):
            logging.error("Mismatch between number of wrods and reduced embeddings.")
            return None
        
        reduced_data = []
        for i in range(len(self.words)):
            reduced_data.append({
                'word': self.words[i],
                'embedding': self.reduced_embeddings[i].tolist()
            })
        return reduced_data