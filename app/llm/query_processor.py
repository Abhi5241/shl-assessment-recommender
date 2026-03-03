import json
import re
from google import genai

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()


class QueryProcessor:

    def __init__(self):
        logger.info("Initializing Gemini client")
        self.client = genai.Client()

    def _build_prompt(self, query: str) -> str:
        return f"""
You are an HR assessment recommendation AI.

Extract structured hiring intent from the user query.

Return ONLY valid JSON (no explanation).

JSON format:
{{
  "technical_skills": [],
  "soft_skills": [],
  "role_type": "",
  "seniority_level": ""
}}

User Query:
"{query}"
"""

    def extract_intent(self, query: str) -> dict:
        prompt = self._build_prompt(query)

        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
            )

            text = response.text.strip()

            json_match = re.search(r"\{.*\}", text, re.DOTALL)

            if json_match:
                parsed_json = json.loads(json_match.group())
                logger.info("Intent extracted using Gemini")
                return parsed_json

            logger.warning("Could not parse JSON, returning raw output")
            return {"raw_output": text}

        except Exception as e:
            logger.error(f"Gemini intent extraction failed: {e}")
            return {"error": str(e)}


# ---------------------------------------------------------
# Local Testing
# ---------------------------------------------------------
if __name__ == "__main__":
    qp = QueryProcessor()

    result = qp.extract_intent(
        "Looking for Python developers with collaboration and teamwork skills"
    )

    from pprint import pprint
    pprint(result)