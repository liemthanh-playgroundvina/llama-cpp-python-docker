import json
import logging
import os
import time
from datetime import datetime
from typing import Optional, Tuple, Text, Dict, List

import requests
from fastapi import UploadFile, File, Body
from typing_extensions import Annotated, Literal

from app.helpers.exception_handler import CustomException
from app.core.config import settings
from app.schemas.base import DataResponse

from app.services.common import CommonService
from sse_starlette import EventSourceResponse

# Load OpenAI API
from openai import OpenAI

client = OpenAI(api_key=settings.OPENAI_KEY)

# # Load Google Gemini API
import google.generativeai as genai

genai.configure(api_key=settings.GOOGLE_GEMINI_KEY)


class LLMChatService(object):
    __instance = None

    @staticmethod
    def chat(request: dict):
        """
        "example": {
                "collection_name": "",
                "messages": [
                    {"role": "user", "content": "Xin chào"},
                    {"role": "assistant", "content": "Chào bạn. Tôi có thể giúp gì cho bạn?"},
                    {"role": "user", "content": "Bạn tên gì?"},
                ],
                "chat_model": {
                    "platform": "OpenAI",
                    "model_name": "gpt-4-1106-preview",
                    "temperature": 0.7,
                    "max_tokens": 4096,

                },
            }
        """
        try:
            # Ask bot
            if request['chat_model']["platform"] == "OpenAI":
                return EventSourceResponse(chat_openai(request))
            elif request['chat_model']["platform"] == "Google":
                return EventSourceResponse(chat_google(request))
            elif request['chat_model']["platform"] == "Local":
                return EventSourceResponse(chat_local(request))

        except ValueError as e:
            raise CustomException(http_code=400, code='400', message=str(e))

        except Exception as e:
            logging.getLogger('app').debug(Exception(e), exc_info=True)
            raise CustomException(http_code=500, code='500', message="Internal Server Error")

    @staticmethod
    def chat_vision(request: dict, file: Optional[UploadFile]):
        """
        "example": {
                "system_prompt": "You are a helpful assistant.",
                "image_url": "https://friendify-bucket.s3.ap-southeast-1.amazonaws.com/avatar/pet/6538d6a6064edefbe5f3a265/cho-bulldog.png",
                "question": "Mô tả ảnh",
                "language": "vi",
                "chat_model": {
                    "platform": "OpenAI",
                    "model_name": "gpt-4-vision-preview",
                    "temperature": 0.7,
                    "max_tokens": 4096,

                },
            }
        """
        try:
            # Ask bot
            if request['chat_model']["platform"] == "OpenAI":
                # Convert to .png and upload s3
                image_url = upload_image_s3(file)
                return EventSourceResponse(chat_vision_openai(request, image_url))
            elif request['chat_model']["platform"] == "Google":
                # Convert to .png
                file_name = CommonService.save_upload_file(file, 'chatbot')
                if file_name in ["image/heif", "image/heic"]:
                    file_name = CommonService.heif_2_png(file_name)
                return EventSourceResponse(chat_vision_google(request, file_name))

        except ValueError as e:
            raise CustomException(http_code=400, code='400', message=str(e))

        except Exception as e:
            logging.getLogger('app').debug(Exception(e), exc_info=True)
            raise CustomException(http_code=500, code='500', message="Internal Server Error")


def chat_openai(request: dict):
    logging.getLogger('app').info(
        f"*** AI CHATBOT VERSION 2 for {request['client_id']}: {request['chat_model']['model_name']} ***")

    message_id = f"message_id_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[DATA_STREAMING]",
    }

    # OpenAI Stream data
    messages, context = get_messages_openai(request)

    # Model
    openai_stream = client.chat.completions.create(
        model=request['chat_model']['model_name'],
        temperature=request['chat_model']['temperature'],
        messages=messages,
        max_tokens=request['chat_model']['max_tokens'],
        stream=True,
    )
    answer = ""
    for line in openai_stream:
        if line.choices[0].delta.content:
            current_response = line.choices[0].delta.content
            answer += current_response
            yield {
                "event": "new_message",
                "id": message_id,
                "retry": settings.RETRY_TIMEOUT,
                "data": current_response.replace("\n", "<!<newline>!>"),
            }
    # End stream
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[DONE]",
    }
    # Metadata
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[METADATA]",
    }
    input_str = ""
    for mess in messages:
        input_str += mess['content']

    input_tokens = num_tokens_from_string_openai(input_str)
    output_tokens = num_tokens_from_string_openai(answer)

    metadata = {
        "platform": request['chat_model']['platform'],
        "model": request['chat_model']['model_name'],
        "temperature": request['chat_model']['temperature'],
        "max_tokens": request['chat_model']['max_tokens'],
        "usage": {
            "input_tokens": input_tokens + 24,
            "output_tokens": output_tokens,
        },
        "context": context,
    }
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": metadata,
    }


def get_messages_openai(request: dict) -> Tuple[List, Text]:
    from app.helpers.prompts.chatbot import openai_chat_system_prompt, openai_chat_user_prompt
    from app.helpers.prompts.chatbot import openai_chat_document_user_prompt
    language = CommonService().iso693_1_to_name(request['language'])
    language = language["language_name"]

    # Initialize
    context = ""

    # Custom system prompt
    if request['messages'][0]['role'] == "system":
        # system_prompt
        messages = request['messages'][:-1]
        # user_prompt following mode
        original_question = request['messages'][-1]['content']
        if request['collection_name']:
            query = google_translate(original_question, 'en')
            context = get_context(request['collection_name'], query)
            user_pmt = openai_chat_document_user_prompt(
                original_question,
                context,
                language
            )
            messages.append({"role": "user", "content": user_pmt})
        else:
            messages.append(request['messages'][-1])

    # Friendify system prompt
    else:
        # system_prompt
        system_prompt = openai_chat_system_prompt(language)

        # user_prompt following mode
        original_question = request['messages'][-1]['content']
        if request['collection_name']:
            query = google_translate(original_question, 'en')
            context = get_context(request['collection_name'], query)
            user_pmt = openai_chat_document_user_prompt(
                original_question,
                context,
                language
            )
        else:
            user_pmt = openai_chat_user_prompt(original_question, language)

        # Conversation
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        for data in request['messages'][:-1]:
            messages.append(data)

        messages.append({"role": "user", "content": user_pmt})

    # Log
    logging.getLogger('app').info("-- PROMPT CHATBOT VERSION 2:")
    if request['collection_name']:
        logging.getLogger('app').info("-- Mode: Chat Document")
    else:
        logging.getLogger('app').info("-- Mode: Chat Bot")
    mess_str = ""
    for mess in messages:
        mess_str += "\n" + json.dumps(mess, ensure_ascii=False)
    logging.getLogger('app').info(mess_str)

    return messages, context


def chat_google(request: dict):
    logging.getLogger('app').info(
        f"*** AI CHATBOT VERSION 2 for {request['client_id']}: {request['chat_model']['model_name']} ***")

    message_id = f"message_id_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[DATA_STREAMING]",
    }

    # Google stream data
    messages, context = get_messages_google(request)
    # Model
    model = genai.GenerativeModel(request['chat_model']['model_name'],
                                  generation_config=genai.types.GenerationConfig(
                                      candidate_count=1,
                                      max_output_tokens=request['chat_model']['max_tokens'],
                                      temperature=request['chat_model']['temperature'],
                                  ))
    chat = model.start_chat()
    response = chat.send_message(messages, stream=True)

    answer = ""
    for chunk in response:
        answer += chunk.text
        list_text = chunk.text.split(' ')
        for text in list_text:
            text = text + " "
            yield {
                "event": "new_message",
                "id": message_id,
                "retry": settings.RETRY_TIMEOUT,
                "data": text.replace("\n", "<!<newline>!>"),
            }
            time.sleep(0.01)
    # End stream
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[DONE]",
    }
    # Metadata
    input_tokens, output_tokens = num_tokens_from_string_google(messages), num_tokens_from_string_google(answer)
    metadata = {
        "platform": request['chat_model']['platform'],
        "model": request['chat_model']['model_name'],
        "temperature": request['chat_model']['temperature'],
        "max_tokens": request['chat_model']['max_tokens'],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "context": context,
    }
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[METADATA]",
    }
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": metadata,
    }


def get_messages_google(request: dict) -> Tuple[Text, Text]:
    from app.helpers.prompts.chatbot import openai_chat_system_prompt, openai_chat_user_prompt
    from app.helpers.prompts.chatbot import openai_chat_document_user_prompt
    language = CommonService().iso693_1_to_name(request['language'])
    language = language["language_name"]

    # Init
    context = ""

    # Custom system_prompt
    if request['messages'][0]['role'] == "system":
        # History
        messages = ""
        for mess in request['messages'][-1]:
            messages += f"{mess['role']}: {mess['content']}\n"
        # Question
        original_question = request['messages'][-1]['content']
        if request['collection_name']:
            query = google_translate(original_question, 'en')
            context = get_context(request['collection_name'], query)
            user_pmt = openai_chat_document_user_prompt(
                original_question,
                context,
                language
            )
            messages += f"user: {user_pmt.strip()}\n"
        else:
            messages += f"user: {original_question}\n"

        messages += """assistant: \n"""

    # Friendify system_prompt
    else:
        # system_prompt
        system_prompt = openai_chat_system_prompt(language)

        # user_prompt following mode
        original_question = request['messages'][-1]['content']
        if request['collection_name']:
            query = google_translate(original_question, 'en')
            context = get_context(request['collection_name'], query)
            user_pmt = openai_chat_document_user_prompt(
                original_question,
                context,
                language
            )
        else:
            user_pmt = openai_chat_user_prompt(original_question, language)

        # Conversation
        messages = system_prompt
        for mess in request['messages'][:-1]:
            if mess['role'] == "assistant":
                mess['role'] = "**Firendify**"
            messages += f"{mess['role']}: {mess['content']}\n"
        messages += f"user: {user_pmt.strip()}"

    # Log
    logging.getLogger('app').info("-- PROMPT CHATBOT VERSION 2:")
    if request['collection_name']:
        logging.getLogger('app').info("-- Mode: Chat Document")
    else:
        logging.getLogger('app').info("-- Mode: Chat Bot")
    logging.getLogger('app').info(messages)

    return messages, context


def chat_local(request: dict):
    logging.getLogger('app').info(
        f"*** AI CHATBOT: {request['chat_model']['model_name']} ***")

    message_id = f"message_id_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[DATA_STREAMING]",
    }

    # Local Stream data
    messages, context = get_messages_local(request)

    # Model
    headers = {
        "accept": "application/json",
        'Content-Type': 'application/json',
    }
    url = settings.URL_OPEN_LLM
    body = {
        "model": request['chat_model']['model_name'],
        "messages": messages,
        "stream": True,
        "max_tokens": request['chat_model']['max_tokens'],
        "temperature": request['chat_model']['temperature'],
    }
    session = requests.Session()
    response = session.post(url=url, headers=headers, json=body, stream=True)
    if response.status_code != 200:
        raise Exception(response.content)

    answer = ""
    for chunk in response.iter_content(decode_unicode=True, chunk_size=1024):
        json_str = chunk.split('data: ')[1]
        try:
            data_dict = json.loads(json_str)
            if 'content' in data_dict['choices'][0]['delta']:
                content = data_dict['choices'][0]['delta']['content']
                answer += content
                yield {
                    "event": "new_message",
                    "id": message_id,
                    "retry": settings.RETRY_TIMEOUT,
                    "data": content.replace("\n", "<!<newline>!>"),
                }
            else:
                yield {
                    "event": "new_message",
                    "id": message_id,
                    "retry": settings.RETRY_TIMEOUT,
                    "data": "",
                }
        except:
            yield {
                "event": "new_message",
                "id": message_id,
                "retry": settings.RETRY_TIMEOUT,
                "data": "",
            }

    # End stream
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[DONE]",
    }

    # Metadata
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[METADATA]",
    }
    input_str = ""
    for mess in messages:
        input_str += mess['content']

    input_tokens = num_tokens_from_string_openai(input_str)
    output_tokens = num_tokens_from_string_openai(answer)

    metadata = {
        "platform": request['chat_model']['platform'],
        "model": request['chat_model']['model_name'],
        "temperature": request['chat_model']['temperature'],
        "max_tokens": request['chat_model']['max_tokens'],
        "usage": {
            "input_tokens": input_tokens + 24,
            "output_tokens": output_tokens,
        },
        "context": context,
    }
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": metadata,
    }


def get_messages_local(request: dict) -> Tuple[List, Text]:
    from app.helpers.prompts.chatbot import openai_chat_document_user_prompt
    # Init
    context = ""

    messages = request['messages'][:-1]

    # Query context
    if request['collection_name']:
        query = request['messages'][-1]['content']
        context = get_context(request['collection_name'], query)
        user_pmt = openai_chat_document_user_prompt(
            query,
            context,
        )
        messages.append({"role": "user", "content": user_pmt})
    else:
        messages.append(request['messages'][-1])

    # Log
    if request['collection_name']:
        logging.getLogger('app').info("-- Mode: Chat Document")
    else:
        logging.getLogger('app').info("-- Mode: Chat Bot")
    logging.getLogger('app').info("-- PROMPT CHATBOT:")
    mess_str = ""
    for mess in messages:
        mess_str += "\n" + json.dumps(mess, ensure_ascii=False)
    logging.getLogger('app').info(mess_str)

    return messages, context


# Chat Vision

def chat_vision_openai(request: dict, image_url: str):
    logging.getLogger('app').info(
        f"*** AI CHAT VISION VERSION 2: {request['chat_model']['model_name']} ***")

    message_id = f"message_id_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[DATA_STREAMING]",
    }
    # Message
    message = get_messages_vision_openai(request, image_url)

    # OpenAI stream data
    # Model
    openai_stream = client.chat.completions.create(
        model=request['chat_model']['model_name'],
        temperature=request['chat_model']['temperature'],
        messages=message,
        max_tokens=request['chat_model']['max_tokens'],
        stream=True,
    )
    answer = ""
    for line in openai_stream:
        if line.choices[0].delta.content:
            current_response = line.choices[0].delta.content
            answer += current_response
            yield {
                "event": "new_message",
                "id": message_id,
                "retry": settings.RETRY_TIMEOUT,
                "data": current_response.replace("\n", "<!<newline>!>"),
            }
    # End stream
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[DONE]",
    }
    # Metadata
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[METADATA]",
    }
    input_str = ""
    for mess in message:
        if isinstance(mess['content'], str):
            input_str += mess['content']
        elif isinstance(mess['content'], list):
            input_str += mess['content'][0]["text"]

    input_tokens = num_tokens_from_string_openai(input_str)
    input_tokens += 85  # 1 images = 85 tokens
    output_tokens = num_tokens_from_string_openai(answer)

    metadata = {
        "platform": request['chat_model']['platform'],
        "model": request['chat_model']['model_name'],
        "temperature": request['chat_model']['temperature'],
        "max_tokens": request['chat_model']['max_tokens'],
        "usage": {
            "input_tokens": input_tokens + 24,
            "output_tokens": output_tokens,
        },
    }
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": metadata,
    }


def get_messages_vision_openai(request: dict, image_url: str) -> List:
    from app.helpers.preprompts.chat_vision import chat_vision_system_prompt, chat_vision_user_prompt

    # Check custom system prompt
    if request['system_prompt'] or len(request['system_prompt'].strip()) > 0:
        content = [
            {"type": "text", "text": request['question']},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}}
        ]
        message = [
            {"role": "system", "content": request['system_prompt']},
            {"role": "user", "content": content},
        ]
    else:
        language = CommonService().iso693_1_to_name(request['language'])
        language = language["language_name"]

        # Conversation
        content = [
            {"type": "text", "text": chat_vision_user_prompt(request['question'], language)},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}}
        ]
        message = [
            {"role": "system", "content": chat_vision_system_prompt()},
            {"role": "user", "content": content},
        ]
    # Log
    logging.getLogger('app').info(f"-- PROMPT CHATBOT VERSION 2:")
    mess_str = ""
    for mess in message:
        mess_str += "\n" + json.dumps(mess, ensure_ascii=False)
    logging.getLogger('app').info(mess_str)

    return message


def chat_vision_google(request: dict, file_name: str):
    from PIL import Image
    logging.getLogger('app').info(
        f"*** AI CHAT VISION VERSION 2: {request['chat_model']['model_name']} ***")

    message_id = f"message_id_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[DATA_STREAMING]",
    }
    # Messages
    message = get_messages_vision_google(request)

    # Google stream data
    # Model
    model = genai.GenerativeModel(request['chat_model']['model_name'],
                                  generation_config=genai.types.GenerationConfig(
                                      candidate_count=1,
                                      max_output_tokens=request['chat_model']['max_tokens'],
                                      temperature=request['chat_model']['temperature'],
                                  ))

    response = model.generate_content(
        [message] + [Image.open(file_name).convert('RGB')],
        stream=True,
    )
    answer = ""
    for chunk in response:
        answer += chunk.text
        list_text = chunk.text.split(' ')
        for text in list_text:
            text = text + " "
            yield {
                "event": "new_message",
                "id": message_id,
                "retry": settings.RETRY_TIMEOUT,
                "data": text.replace("\n", "<!<newline>!>"),
            }
            time.sleep(0.01)

    # End stream
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[DONE]",
    }

    # Metadata
    input_tokens = num_tokens_from_string_google(request['question'], request['chat_model']['model_name'])
    input_tokens += 10000  # 1 image = 10000 characters
    output_tokens = num_tokens_from_string_google(answer, request['chat_model']['model_name'])
    metadata = {
        "platform": request['chat_model']['platform'],
        "model": request['chat_model']['model_name'],
        "temperature": request['chat_model']['temperature'],
        "max_tokens": request['chat_model']['max_tokens'],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": "[METADATA]",
    }
    yield {
        "event": "new_message",
        "id": message_id,
        "retry": settings.RETRY_TIMEOUT,
        "data": metadata,
    }
    try:
        os.remove(file_name)
    except:
        pass


def get_messages_vision_google(request: dict) -> Text:
    from app.helpers.preprompts.chat_vision import chat_vision_system_prompt, chat_vision_user_prompt

    # Check custom system prompt
    if request['system_prompt'] or len(request['system_prompt'].strip()) > 0:
        message = f"{request['system_prompt']}\nUser: {request['question']}\nAssistant: "
    else:
        language = CommonService().iso693_1_to_name(request['language'])
        language = language["language_name"]

        # Conversation
        message = chat_vision_system_prompt() + chat_vision_user_prompt(request['question'], language)

    # Log
    logging.getLogger('app').info(f"-- PROMPT CHATBOT VERSION 2:")
    logging.getLogger('app').info(message)

    return message


########################################################################################################################
# Common
def google_translate(query: str, target: str, source: str = "auto") -> Text:
    from deep_translator import GoogleTranslator

    def split_string_max_length(input_str: str, max_length: int, split_character: str) -> List:
        parts = input_str.split(split_character)

        result = []
        current_part = ""

        for part in parts:
            if len(current_part) + len(part) + 1 <= max_length:
                if current_part:
                    current_part += split_character
                current_part += part
            else:
                result.append(current_part)
                current_part = part

        if current_part:
            result.append(current_part)

        return result

    max_length_character = 4500
    split_char = ". "
    queries = split_string_max_length(query, max_length_character, split_char)
    queries = list(filter(lambda x: x is not None and x.strip() != "", queries))
    queries = [q.strip()[:max_length_character] for q in queries]

    my_translator = GoogleTranslator(source=source, target=target)
    translated = my_translator.translate_batch(batch=queries)
    translated = list(filter(lambda x: x is not None and x.strip() != "", translated))
    text_translated = split_char.join(translated)

    return text_translated


def upload_image_s3(file: Optional[UploadFile] = File(None)) -> Text:
    content_type = file.content_type
    # Save
    file_name = CommonService.save_upload_file(file, 'chatbot')

    # Convert to .png
    if file_name in ["image/heif", "image/heic"]:
        file_name = CommonService.heif_2_png(file_name)

    # Upload to s3
    url = CommonService.upload_s3_file(file_name, content_type, "chatbot/predict")

    try:
        os.remove(file_name)
    except:
        pass

    return url['url']


def num_tokens_from_string_openai(string: str, encoding_name: str = "cl100k_base") -> int:
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_string_google(text: str, model_name: Text = "gemini-pro") -> int:
    """
    Calculates tokens google gemini
    """

    # Json
    body = {
        "contents": [{
            "parts": [{
                "text": text}]
        }]
    }

    # Request api count tokens
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model_name}:countTokens?key=' + settings.GOOGLE_GEMINI_KEY

    # input
    response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(body))
    if response.status_code == 200:
        token = json.loads(response.content.decode('utf-8'))
        token = token['totalTokens']
        # print(total_input_tokens)
    else:
        raise ValueError(str(response.content))

    return token


########################################################################################################################
# Chat documentation
from app.schemas.chatbot import LLMEmbedDocRequest
from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceHubEmbeddings
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingServer
import chromadb
from chromadb.config import Settings

# Model: "BAAI/bge-large-en-v1.5" (/home/ai/ai-demo-tool/docker-compose.yml line 45)
embeddings_model = HuggingFaceHubEmbeddings(model=settings.URL_EMBED_MODEL)
embeddings_collection = HuggingFaceEmbeddingServer(url=settings.URL_EMBED_MODEL)
domain_db = settings.URL_VECTOR_STORE.split("//")[-1].replace("/", "").split(":")
chromadb_client = chromadb.HttpClient(host=domain_db[0],
                                      port=domain_db[1],
                                      settings=Settings(allow_reset=True))


class LLMChatDocService(object):
    __instance = None

    @staticmethod
    def embed_doc(request: LLMEmbedDocRequest, file: Annotated[UploadFile, File(...)] = None):
        try:
            # Load docs
            docs = load_docs(request, file)
            # Add docs into VectorStore
            if request.type_data == 'file':
                collection_name = add_collection(docs, file.filename)
            else:
                collection_name = add_collection(docs, request.url)
            return DataResponse().success_response(data={"collection_name": collection_name})

        except ValueError as e:
            raise CustomException(http_code=400, code='400', message=str(e))

        except Exception as e:
            logging.getLogger('app').debug(Exception(e), exc_info=True)
            raise CustomException(http_code=500, code='500', message="Internal Server Error")


def load_docs(request: LLMEmbedDocRequest, file: Annotated[UploadFile, File(...)] = None) -> list[Document]:
    from langchain_community.document_loaders import UnstructuredFileLoader
    from langchain_community.document_loaders import UnstructuredURLLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    if request.type_data == 'file' and file is not None:
        file_path = CommonService().save_upload_file(file)
        loader = UnstructuredFileLoader(file_path)
        raw_documents = loader.load()
    elif request.type_data == 'web_url':
        loader = UnstructuredURLLoader(urls=[request.url])
        raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(raw_documents)
    logging.getLogger('app').info(f"Documents: {documents}")

    return documents


def add_collection(documents: list[Document], filename: str) -> str:
    import uuid
    import re
    filename = re.sub(r"[^a-zA-Z0-9]", "", filename)[-30:]
    collection_name = f"db_{os.path.basename(filename)}_{str(time.time())}"
    collection = chromadb_client.create_collection(
        collection_name,
        embedding_function=embeddings_collection
    )
    for doc in documents:
        collection.add(
            ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
        )
    return collection_name


def get_context(collection_name: str, question: str, number_context: int = 2) -> str:
    """Get Context in Vector Store following collection name"""
    from langchain_community.vectorstores import Chroma

    db_vector_store = Chroma(
        client=chromadb_client,
        collection_name=collection_name,
        # embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    embedding_vector = embeddings_model.embed_query(question)
    docs_query = db_vector_store.similarity_search_by_vector_with_relevance_scores(embedding_vector, k=number_context)
    contexts = ""
    # logging.getLogger('app').info(f"Question: {question}")
    for index, document in enumerate(docs_query):
        cosine_score = document[1]
        logging.getLogger('app').info(f"Cosine score of : {cosine_score}")
        logging.getLogger('app').info(f"# Context {index + 1}:\n{document[0].page_content}\n\n")
        ##  "BAAI/bge-large-en-v1.5"
        # THRESHOLD_bge = 0.85
        # if cosine_score <= THRESHOLD_bge:
        # BAAI/bge-m3
        contexts += f"# Context {index + 1}:\n{document[0].page_content}\n\n"

    return contexts.strip()
