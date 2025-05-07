from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
import logging
import asyncio

from utils.stem import Stemmer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='./logs/stemm_router.log'
)

try:
    GLOBAL_STEMMER_INSTANCE: Stemmer = Stemmer()
    logging.info("Stemmer initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize Stemmer: {e}")
    GLOBAL_STEMMER_INSTANCE = None

def get_stemmer() -> Stemmer:
    """
    Provides the initialized Stemmer instance.
    """
    if GLOBAL_STEMMER_INSTANCE is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stemmer service is not available."
        )
    return GLOBAL_STEMMER_INSTANCE


class TextInput(BaseModel):
    """
    Input model for the stemming request.
    """
    text: str

class StemmingResponse(BaseModel):
    """
    Output model for the stemmer reaponse.
    """
    stems: list[str]

## --- Router --- ##
router = APIRouter()

@router.post(
    "/stemmize",
    response_model=StemmingResponse,
    status_code=status.HTTP_200_OK,
    summary="Stemmizes the input text",
    description="Splits the provided text into the tokes using the configured tokenizer."
)
async def stemm_data(data: TextInput, stemmer: Stemmer = Depends(get_stemmer)):
    """
    Receives text input and returns a list of tokens.
    Handels potenttial errors during stemming.
    """
    
    logging.info(f"Received stemming request for text: '{data.text[:50]}...'")
    
    try:
        stem = await asyncio.to_thread(stemmer.stem, text = data.text)
        logging.info(f"Successfullly stemmed text. Generated {len(stem)} steams.")
        
        return StemmingResponse(stems=stem)
    except Exception as e:
        logging.error(f"Error during stemming for text: '{data.text[:50]}...' - {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occured during stemming: {e}"
        )