from app.llm.query_processor import QueryProcessor
from app.vectorstore.faiss_store import FAISSVectorStore
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger()


class RecommendationService:

    def __init__(self):
        logger.info("Initializing Recommendation Service")

        self.query_processor = QueryProcessor()
        self.vector_store = FAISSVectorStore()
        self.vector_store.load_index()

    # --------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------
    def recommend(
        self,
        user_query: str,
        use_llm: bool = True,
        generate_explanation: bool = False
    ):
        """
        use_llm = False → disables Gemini (for evaluation runs)
        generate_explanation = False → skip explanation generation
        """

        logger.info(f"Processing query: {user_query}")

        # ---------------------------
        # 1️⃣ Intent Extraction (optional)
        # ---------------------------
        intent = {}

        if use_llm:
            try:
                intent = self.query_processor.extract_intent(user_query)
            except Exception as e:
                logger.warning(f"Intent extraction failed: {e}")

        # ---------------------------
        # 2️⃣ Query Enhancement
        # ---------------------------
        enhanced_query = user_query

        if isinstance(intent, dict):
            skills = intent.get("technical_skills", [])
            soft = intent.get("soft_skills", [])

            enhanced_query += " " + " ".join(skills + soft)

        logger.info(f"Enhanced query: {enhanced_query}")

        # ---------------------------
        # 3️⃣ FAISS Search
        # ---------------------------
        results = self.vector_store.search(
            enhanced_query,
            top_k=settings.TOP_K_RESULTS
        )

        response = {
            "query": user_query,
            "intent": intent,
            "recommendations": results
        }

        # ---------------------------
        # 4️⃣ Explanation (optional)
        # ---------------------------
        if generate_explanation and use_llm:
            try:
                response["explanation"] = self.generate_explanation(
                    user_query,
                    results
                )
            except Exception as e:
                logger.error(f"Explanation generation failed: {e}")

        return response

    # --------------------------------------------------
    # LLM EXPLANATION
    # --------------------------------------------------
    def generate_explanation(self, query: str, recommendations: list):

        if not recommendations:
            return "No recommendations available."

        context = "\n".join(
            f"{r['name']}: {r['description']}"
            for r in recommendations[:3]
        )

        prompt = f"""
You are an HR assessment expert.

Explain briefly WHY these assessments are recommended.

Hiring Requirement:
{query}

Assessments:
{context}

Return 3-5 bullet points only.
"""

        response = self.query_processor.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt
        )

        return response.text.strip()


# quick test
if __name__ == "__main__":
    service = RecommendationService()

    result = service.recommend(
        "Looking for Python developers with teamwork skills",
        use_llm=False
    )

    from pprint import pprint
    pprint(result)