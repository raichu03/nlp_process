from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
import logging
import asyncio

from utils.ner import NerRecognizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='./logs/ner_router.log'
)

try:
    GLOBAL_NER_INSTANCE: NerRecognizer = NerRecognizer()
    logging.info("Ner initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize Ner: {e}")
    GLOBAL_NER_INSTANCE = None

def get_ner() -> NerRecognizer:
    """
    Provides the initialized NER instance.
    """
    if GLOBAL_NER_INSTANCE is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="NER service is not available."
        )
    return GLOBAL_NER_INSTANCE


class TextInput(BaseModel):
    """
    Input model for the NER request.
    """
    text: str

class NerResponse(BaseModel):
    """
    Output model for the NER reaponse.
    """
    ner: list[tuple[str, str, tuple[int, int]]]

## --- Router --- ##
router = APIRouter()

@router.post(
    "/ner",
    response_model= NerResponse,
    status_code=status.HTTP_200_OK,
    summary="recognizes the input text",
    description="Recognizes the different named entity within the given string text."
)
async def recognize_data(data: TextInput, ner: NerRecognizer = Depends(get_ner)):
    """
    Receives text input and returns a list of named entity.
    Handels potential errors during ner.
    """
    
    logging.info(f"Received ner request for text: '{data.text[:50]}...'")
    
    try:
        ner = await asyncio.to_thread(ner.recognize_entities, text = data.text)
        logging.info(f"Successfullly tokenize text. Generated {len(ner)} tokens.")
        
        return NerResponse(ner=ner)
    except Exception as e:
        logging.error(f"Error during entity recognition for text: '{data.text[:50]}...' - {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occured during recognition: {e}"
        )