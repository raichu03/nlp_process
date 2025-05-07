from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
import logging
import asyncio

from utils.token import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='./logs/token_router.log'
)

try:
    GLOBAL_TOKENIZER_INSTANCE: Tokenizer = Tokenizer()
    logging.info("Tokenizer initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize Tokenizer: {e}")
    GLOBAL_TOKENIZER_INSTANCE = None

def get_tokenizer() -> Tokenizer:
    """
    Provides the initialized Tokenizer instance.
    """
    if GLOBAL_TOKENIZER_INSTANCE is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tokenizer service is not available."
        )
    return GLOBAL_TOKENIZER_INSTANCE


class TextInput(BaseModel):
    """
    Input model for the tokenization request.
    """
    text: str

class TokenozeResponse(BaseModel):
    """
    Output model for the tokenization reaponse.
    """
    tokens: list[str]

## --- Router --- ##
router = APIRouter()

@router.post(
    "/tokenize",
    response_model=TokenozeResponse,
    status_code=status.HTTP_200_OK,
    summary="Tokenizes the input text",
    description="Splits the provided text into the tokes using the configured tokenizer."
)
async def tokenize_data(data: TextInput, tokenizer: Tokenizer = Depends(get_tokenizer)):
    """
    Receives text input and returns a list of tokens.
    Handels potenttial errors during tokenization.
    """
    
    logging.info(f"Received tokenization request for text: '{data.text[:50]}...'")
    
    try:
        tokens = await asyncio.to_thread(tokenizer.tokenize, text = data.text)
        logging.info(f"Successfullly tokenize text. Generated {len(tokens)} tokens.")
        
        return TokenozeResponse(tokens=tokens)
    except Exception as e:
        logging.error(f"Error during tokenization for text: '{data.text[:50]}...' - {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occured during tokenization: {e}"
        )