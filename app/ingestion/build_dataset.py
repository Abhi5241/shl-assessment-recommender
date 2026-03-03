import pandas as pd
import json
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()


class DatasetBuilder:

    def __init__(self):
        self.input_file = "data/source/shl_catalog.csv"
        self.output_file = Path(settings.RAW_DATA_PATH) / "shl_catalog_raw.json"

        Path(settings.RAW_DATA_PATH).mkdir(parents=True, exist_ok=True)

    def build(self):
        logger.info("Loading SHL catalog dataset...")

        df = pd.read_csv(self.input_file, encoding="latin1")

        logger.info(f"Rows found: {len(df)}")
        logger.info(f"Columns: {list(df.columns)}")

        dataset = []

        for _, row in df.iterrows():
            dataset.append({
                "name": row["test_name"],
                "url": row["url"],
                "description": row["description"],
                "test_type": row["test_type"],
                "duration": row["duration"],
                "adaptive_support": row["adaptive_support"],
                "remote_support": row["remote_support"],
            })

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        logger.success(
            f"Saved {len(dataset)} assessments → {self.output_file}"
        )


if __name__ == "__main__":
    DatasetBuilder().build()