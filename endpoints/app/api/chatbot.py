import io
import logging
import os

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Request, Body

from app.helpers.exception_handler import CustomException
from app.schemas.base import DataResponse
from app.schemas.chatbot import LLMChatRequest, LLMChatVisionRequest, LLMEmbedDocRequest

from app.services.upload_s3 import load_file_from_s3
from app.services.chatbot import LLMChatService, LLMChatDocService

logger = logging.getLogger()
router = APIRouter()


@router.post(
    "/chat",
    # response_model=DataResponse[]
)
def chatbot(request: LLMChatRequest) -> Any:
    """
    API LLM Chatbot version 2: Chatbot & Chat Documentation

    Params:

        collection_name (str): Using for Chat Document, and "" or null will use chatbot

        messages (list):

        chat_model (dict):
            - platform (str): ["Google", "OpenAI", "Local"]
            - model_name (str): {
                "Google": ["gemini-pro"],
                "OpenAI": ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"],
                "Local": ["gemma-2b"]
            }
            - temperature (float): [0 -> 1.0]
            - max_tokens (int): {
                # Google
                "gemini-pro": 8192,
                # OpenAI
                "gpt-3.5-turbo-1106": 4096,
                "gpt-4-1106-preview": 4096,
                # Local
                "gemma-2b": 2048,
            }

    Returns:

        response: [DATA_STREAMING] <string_data> [DONE] [METADATA] <json_metadata>

        in <string_data>: '\\n' is replaced to '<!<newline>!>'

    Note:

        If have not "system" role in messages, using "friendify" system prompt.
        If collection_name is null or "" will use Chatbot, and use Chat Document when else
    """
    return LLMChatService().chat(request)


@router.post(
    "/chat-vision",
    # response_model=DataResponse[]
)
def chat_vision(request: LLMChatVisionRequest) -> Any:
    """
    API LLM Chat version 2: Chat Vision

    Params:

        system_prompt (str): system prompt for chat vision model.

        question (str):

        image_url (str): in types ["image/jpeg", "image/png", "image/heif", "image/heic"]

        language (str): ISO 639-1 /static/data/translation_longtext.json

        chat_model (dict):
            - platform (str): ["Google", "OpenAI"]
            - model_name (str): {
                "Google": ["gemini-pro-vision"],
                "OpenAI": ["gpt-4-vision-preview"],
            }
            - temperature (float): [0 -> 1.0]
            - max_tokens (int): {
                # Google
                "gemini-pro-vision": 2048,
                # OpenAI
                "gpt-4-vision-preview": 4096,
            }

    Returns:

        response: [DATA_STREAMING] <string_data> [DONE] [METADATA] <json_metadata>

        in <string_data>: '\\n' is replaced to '<!<newline>!>'

    Note:

    """
    try:
        file_path, headers = load_file_from_s3(request['image_url'])
        with open(file_path, 'rb') as f:
            file_contents = f.read()
        file = UploadFile(filename=os.path.basename(file_path), file=io.BytesIO(file_contents), headers=headers)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        raise CustomException(http_code=400, code='400', message=str(e))

    if file is not None:
        image_type = ["image/jpeg", "image/png", "image/heif", "image/heic"]
        supported_type = image_type

        if file.content_type not in supported_type:
            message = f"Invalid file format. Only {supported_type} type files are supported (current format is '{file.content_type}')"
            raise CustomException(http_code=400, code='400', message=str(message))

    return LLMChatService().chat_vision(request, file)


@router.post(
    "/chat-doc/embed",
    # response_model=DataResponse[]
)
def embed_doc(request: LLMEmbedDocRequest) -> Any:
    """
    API Embed Document for Chat Document
    Vector Store: Chromadb
    Embedding Model: BAAI/bge-large-en-v1.5

    Params:

        type_data (str): [file, web_url]
        url (str): ["s3_file", "https://www.playgroundx.site/"]

    Returns:

        data_id (str): The collection name to chat

    Note:

        Language in document must is "English" language
    """
    request = LLMEmbedDocRequest(**request)
    try:
        if request.type_data == "file":
            file_path, headers = load_file_from_s3(request.url)
            with open(file_path, 'rb') as f:
                file_contents = f.read()
            file = UploadFile(filename=os.path.basename(file_path), file=io.BytesIO(file_contents), headers=headers)
            if os.path.exists(file_path):
                os.remove(file_path)
        else:
            file = None
    except Exception as e:
        raise CustomException(http_code=400, code='400', message=str(e))

    if file is not None:
        text_type = ['text/plain', 'text/markdown']
        doc_type = ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
        pdf_type = ['application/pdf']
        supported_type = text_type + doc_type + pdf_type

        if file.content_type not in supported_type:
            message = f"Invalid file format. Only {supported_type} type files are supported (current format is '{file.content_type}')"
            raise CustomException(http_code=400, code='400', message=str(message))

    return LLMChatDocService().embed_doc(request, file)
