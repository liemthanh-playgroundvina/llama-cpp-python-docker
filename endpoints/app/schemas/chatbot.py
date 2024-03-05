import json
import os
from typing import Optional

from pydantic import BaseModel, root_validator

from app.core.config import settings
from app.helpers.exception_handler import CustomException


class LLMChatRequest(BaseModel):
    collection_name: Optional[str] = ""
    messages: Optional[list]
    chat_model: Optional[dict]

    class Config:
        schema_extra = {
            "example": {
                "collection_name": "",
                "messages": [
                    {"role": "user", "content": "Xin chào"},
                    {"role": "assistant", "content": "Chào bạn. Tôi có thể giúp gì cho bạn?"},
                    {"role": "user", "content": "Bạn tên gì?"},
                ],
                "chat_model": {
                    "platform": "Local",
                    "model_name": "gemma-2b",
                    "temperature": 0.7,
                    "max_tokens": 2048,

                },
            }
        }

    @root_validator()
    def validate(cls, values):
        messages = values['messages']
        chat_model = values['chat_model']
        try:
            collection_name = values['collection_name']
        except:
            values['collection_name'] = ""

    # Handler
        # role
        list_role = ["user", "assistant", "system"]
        for mess in messages:
            if mess['role'] not in list_role:
                message = f'[role] in messages must in {list_role}'
                raise CustomException(http_code=400, code='400', message=message)

        # chat_model
        if chat_model:
            # Fields required
            required_fields = ["platform", "model_name", "temperature", "max_tokens"]
            missing_fields = [field for field in required_fields if field not in chat_model]
            if missing_fields:
                message = f"Missing fields in [chat_model]: {', '.join(missing_fields)}"
                raise CustomException(http_code=400, code='400', message=message)
            platform = chat_model['platform']
            model_name = chat_model['model_name']
            temperature = chat_model['temperature']
            max_tokens = chat_model['max_tokens']

            # platform & model & temperature
            platforms = {
                "Google": ["gemini-pro"],
                "OpenAI": ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"],
                "Local": ["gemma-2b"],
            }
            list_platform = list(platforms.keys())
            if platform not in list_platform:
                message = f"Don't support '{platform}'.\n{list_platform} is supported."
                raise CustomException(http_code=400, code='400', message=message)
            if model_name not in platforms[platform]:
                message = f"Don't support '{model_name}'.\n{platforms[platform]} is supported."
                raise CustomException(http_code=400, code='400', message=message)

            # temperature
            if not (0 <= temperature <= 1.0):
                message = f"[temperature] must in [0.0, 1.0]. Your current temperature is {temperature}."
                raise CustomException(http_code=400, code='400', message=message)
            # max_tokens
            tokens = {
                # Google
                "gemini-pro": 8192,
                # OpenAI
                "gpt-3.5-turbo-1106": 4096,
                "gpt-4-1106-preview": 4096,
                # Local
                "gemma-2b": 2048,
            }

            if not (256 <= max_tokens <= tokens[model_name]):
                message = f"[max_tokens] of '{model_name}' must be between [256, {tokens[model_name]}]"
                raise CustomException(http_code=400, code='400', message=message)

        return values


class LLMChatVisionRequest(BaseModel):
    system_prompt: Optional[str] = ""
    image_url: Optional[str]
    question: Optional[str]
    language: Optional[str]
    chat_model: Optional[dict]

    class Config:
        schema_extra = {
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
        }

    @root_validator()
    def validate(cls, values):
        system_prompt = values['system_prompt']
        image_url = values['image_url']
        question = values['question']
        language = values['language']
        chat_model = values['chat_model']

        # Read languages file
        try:
            LANGUAGES_FILE = os.path.join(settings.STATIC_URL, "public", "data", "translation_longtext.json")
            with open(LANGUAGES_FILE, 'r') as f:
                data = json.load(f)
        except Exception:
            message = "Cant read " + LANGUAGES_FILE
            raise CustomException(http_code=400, code='400', message=message)

    # Handler
        # language
        list_language = list(data.keys())
        if language not in list_language:
            message = f"Don't support language {language}"
            raise CustomException(http_code=400, code='400', message=message)

        # question
        if len(question.strip()) == 0:
            message = f"[question] is not empty"
            raise CustomException(http_code=400, code='400', message=message)

        # chat_model
        if chat_model:
            # Fields required
            required_fields = ["platform", "model_name", "temperature", "max_tokens"]
            missing_fields = [field for field in required_fields if field not in chat_model]
            if missing_fields:
                message = f"Missing fields in [chat_model]: {', '.join(missing_fields)}"
                raise CustomException(http_code=400, code='400', message=message)
            platform = chat_model['platform']
            model_name = chat_model['model_name']
            temperature = chat_model['temperature']
            max_tokens = chat_model['max_tokens']

            # platform & model & temperature
            platforms = {
                "Google": ["gemini-pro-vision"],
                "OpenAI": ["gpt-4-vision-preview"],
            }
            list_platform = list(platforms.keys())
            if platform not in list_platform:
                message = f"Don't support '{platform}'.\n{list_platform} is supported."
                raise CustomException(http_code=400, code='400', message=message)
            if model_name not in platforms[platform]:
                message = f"Don't support '{model_name}'.\n{platforms[platform]} is supported."
                raise CustomException(http_code=400, code='400', message=message)

            # temperature
            if not (0 <= temperature <= 1.0):
                message = f"[temperature] must in [0.0, 1.0]. Your current temperature is {temperature}."
                raise CustomException(http_code=400, code='400', message=message)
            # max_tokens
            tokens = {
                # Google
                "gemini-pro-vision": 2048,
                # OpenAI
                "gpt-4-vision-preview": 4096,
            }

            if not (256 <= max_tokens <= tokens[model_name]):
                message = f"[max_tokens] of '{model_name}' must be between [256, {tokens[model_name]}]"
                raise CustomException(http_code=400, code='400', message=message)

        return values


# # Chat doc
class LLMEmbedDocRequest(BaseModel):
    type_data: str
    url: str

    class Config:
        schema_extra = {
            "example": {
                "type_data": "file",
                "url": "https://aiservices-bucket.s3.amazonaws.com/translate_document/2024_02_05_03_31_19_849_2024_02_05_03_31_19_801_[VIE]%20GPT4,%203.5%20Prompt%20Guide_translated_en.docx",

            }
        }

    @root_validator()
    def validate(cls, values):
        type_data = values['type_data']
        url = values['url']
        return values
