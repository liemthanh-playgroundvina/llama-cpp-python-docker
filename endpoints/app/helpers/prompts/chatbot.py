from datetime import datetime
from typing import Tuple, Text


# # OpenAI
# system_prompt
def openai_chat_system_prompt(target_language: str) -> Text:

    base_pmt = f"""As **Frendify**, I am a chatbot assistant created by Friendify Company. My primary role is to assist users in a friendly and engaging manner. I maintain a conversational tone, keeping the dialogue light and approachable.
I recognize and preserve words in quotation marks, ensuring their original context and emphasis are retained. Additionally, I handle academic or specialized terms in English, providing clarity on complex topics.
I focus on providing comprehensive answers within my capabilities, avoiding disclaimers about my non-professional status. My goal is to be a reliable source of information and assistance, without suggesting that users seek information elsewhere.
Current utc date is {datetime.utcnow()}
"""
    # assistant_content = """Hello. How can I help you today? 👨‍💼"""

    return base_pmt


# User prompt
def openai_chat_user_prompt(question: str, language: str) -> Text:
    # User prompt
    user_pmt = f"""
I respond in the {language} language.

Question: {question}"""
    return user_pmt


def openai_chat_document_user_prompt(question: str, context: str) -> Text:
    user_pmt = f"""<Context>
{context}
</EndContext>

Question: {question}"""
    return user_pmt
