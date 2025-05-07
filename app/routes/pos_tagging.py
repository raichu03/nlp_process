from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
import logging
import asyncio

from utils.pos_tags import PosTagger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='logs/pos_router.log'
)

try:
    GLOBAL_POS_INSTANCE: PosTagger = PosTagger()
    logging.info("Pos tagger initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize Pos tagger: {e}")
    GLOBAL_POS_INSTANCE = None

def get_pos() -> PosTagger:
    """
    Provides the initialized Pos tagger instance.
    """
    if GLOBAL_POS_INSTANCE is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pos tagger service is not available."
        )
    return GLOBAL_POS_INSTANCE


class TextInput(BaseModel):
    """
    Input model for the Pos tagger request.
    """
    text: str

class PosResponse(BaseModel):
    """
    Output model for the Pos tagger reaponse.
    """
    pos: list[tuple[str, str]]

## --- Router --- ##
router = APIRouter()

@router.post(
    "/pos",
    response_model= PosResponse,
    status_code=status.HTTP_200_OK,
    summary="recognizes the input text",
    description="Identifies the POS the different tokens within the given string text."
)
async def recognize_data(data: TextInput, pos_tag: PosTagger = Depends(get_pos)):
    """
    Receives text input and returns a list of tokens and POS tag.
    Handels potential errors during pos tagging.
    """
    
    logging.info(f"Received pos tagging request for text: '{data.text[:50]}...'")
    
    try:
        pos = await asyncio.to_thread(pos_tag.get_pos_tags, text = data.text)
        logging.info(f"Successfullly generated pos tags. Generated {len(pos)} tokens.")
        
        return PosResponse(pos=pos)
    except Exception as e:
        logging.error(f"Error during pos tagging for text: '{data.text[:50]}...' - {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occured during pos tagging: {e}"
        )