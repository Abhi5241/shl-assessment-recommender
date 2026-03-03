import pandas as pd
from app.services.recommendation_service import RecommendationService
from app.core.logging import get_logger

logger = get_logger()

# -----------------------------
# FILE PATHS
# -----------------------------
INPUT_FILE = "data/source/test.csv"
OUTPUT_FILE = "data/predictions.csv"


def generate_predictions():

    logger.info("Loading test dataset...")

    # SHL test file contains ONLY Query column
    df = pd.read_csv(INPUT_FILE, encoding="latin1")

    if "Query" not in df.columns:
        raise ValueError("Input CSV must contain 'Query' column")

    service = RecommendationService()

    output_rows = []

    # -----------------------------------
    # Generate predictions
    # -----------------------------------
    for query in df["Query"]:

        logger.info(f"Processing: {query}")

        response = service.recommend(
            query,
            use_llm=False,              # ✅ NO GEMINI (avoid quota)
            generate_explanation=False  # ✅ NOT REQUIRED
        )

        recommendations = response["recommendations"]

        # take TOP result only (SHL format)
        if recommendations:
            best_url = recommendations[0]["url"]
        else:
            best_url = ""

        output_rows.append({
            "Query": query,
            "Assessment_url": best_url
        })

    # -----------------------------------
    # Save CSV
    # -----------------------------------
    output_df = pd.DataFrame(output_rows)

    output_df.to_csv(OUTPUT_FILE, index=False)

    logger.success(f"Predictions saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_predictions()