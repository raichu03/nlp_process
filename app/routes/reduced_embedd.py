from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
import logging
import asyncio
from typing import Dict

from utils.embedding import GloveEmbeddingGenerator
from utils.dimesnions import DimensionReduction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='./logs/embedding_router.log'
)

try:
    GLOBAL_EMBEDDING_INSTANCE: GloveEmbeddingGenerator = GloveEmbeddingGenerator()
    logging.info("Embedding initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize Embedding: {e}")
    GLOBAL_EMBEDDING_INSTANCE = None

def get_embedding() -> GloveEmbeddingGenerator:
    """
    Provides the initialized Embedder instance.
    """
    if GLOBAL_EMBEDDING_INSTANCE is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedder service is not available."
        )
    return GLOBAL_EMBEDDING_INSTANCE


class TextInput(BaseModel):
    """
    Input model for the stemming request.
    """
    text: list[str]

class WordEmbeddings(BaseModel):
    """
    Defines the format for the output
    """
    word: str
    embedding: list
    
class ReducedResponse(BaseModel):
    """
    Output model for the stemmer reaponse.
    """
    reduced: list[WordEmbeddings]

## --- Router --- ##
router = APIRouter()

@router.post(
    "/embed",
    response_model=ReducedResponse,
    status_code=status.HTTP_200_OK,
    summary="Embedds the input text",
    description="Embedds the given word and reduces the dimension of the list of words"
)
async def stemm_data(data: TextInput, embedd: GloveEmbeddingGenerator = Depends(get_embedding)):
    """
    Receives the list of words and generates the embeddings
    and reduces the dimesnions of the embeddings
    """
    
    logging.info(f"Received reduction request for text: '{data.text}...'")
    
    try:
        embed = await asyncio.to_thread(embedd.get_word_embeddings, words = data.text)
        logging.info(f"Successfullly embedd text. Generated {len(embed)} embeddings.")
        
        
        reducer = DimensionReduction(embeddings_dict=embed)
        print("yes")
        embed_pca = reducer.perform_pca(n_components=2)
        response = reducer.get_reduced_data()
        print(response)
        
        if response is not None:
            return ReducedResponse(reduced=response)
        else:
            return ReducedResponse(reduced=None)
        
    except Exception as e:
        logging.error(f"Error during embedding for text: '{data.text}...' - {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occured during embedding: {e}"
        )