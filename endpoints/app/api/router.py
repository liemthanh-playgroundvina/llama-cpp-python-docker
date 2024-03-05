from fastapi import APIRouter
router = APIRouter()

from app.api import (
    chatbot,

)
router.include_router(chatbot.router, tags=[
                    "chatbot"], prefix="/chatbot")
