"""Small local configuration; this module never reads .env files."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ApiSettings:
    workbook_path: Path = Path("data/sample_cars_dataset.xlsx")
    cache_path: Path = Path(".cache/inventory_embeddings.json")
    semantic_enabled: bool = True
    chat_model: str = "gemini-3.8-flash"
    state_db_path: Path = Path("data/dubizzle_assistant.db")
    leads_csv_path: Path = Path("data/leads.csv")

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        enabled = os.environ.get("DUBIZZLE_SEMANTIC_ENABLED", "true").lower().strip()
        if enabled not in {"true", "false"}:
            raise ValueError("DUBIZZLE_SEMANTIC_ENABLED must be true or false")
        return cls(
            workbook_path=Path(
                os.environ.get("DUBIZZLE_WORKBOOK_PATH", "data/sample_cars_dataset.xlsx")
            ),
            cache_path=Path(
                os.environ.get("DUBIZZLE_EMBEDDING_CACHE_PATH", ".cache/inventory_embeddings.json")
            ),
            semantic_enabled=enabled == "true",
            chat_model=os.environ.get("DUBIZZLE_CHAT_MODEL", "gemini-3.8-flash"),
            state_db_path=Path(
                os.environ.get("DUBIZZLE_STATE_DB_PATH", "data/dubizzle_assistant.db")
            ),
            leads_csv_path=Path(os.environ.get("DUBIZZLE_LEADS_CSV_PATH", "data/leads.csv")),
        )
