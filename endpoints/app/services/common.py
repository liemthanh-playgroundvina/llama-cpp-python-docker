import os
import json

from typing import Text, List, Union, Dict, Tuple, Any
from app.core.config import settings


class CommonService(object):
    __instance = None

    @staticmethod
    def iso693_1_to_name(code_name) -> Dict[Text, Text]:
        """Convert ISO 693-1 code to language name {name, ISO693-2 code}."""
        try:
            lang_file = os.path.join(
                settings.STATIC_URL, "public", "data", "iso_language.json"
            )
            with open(lang_file, "r") as f:
                data = json.load(f)
        except Exception:
            message = "Cant read " + lang_file
            raise Exception(message)

        if code_name not in data:
            message = f"Language {code_name} is not supported"
            raise ValueError(message)

        iso693_2_code = data.get(code_name).get("code_3c")
        language_name = data.get(code_name).get("language")

        return {"language_name": language_name, "iso693_2": iso693_2_code}

