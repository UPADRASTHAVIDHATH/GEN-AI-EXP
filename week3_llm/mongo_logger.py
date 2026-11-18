"""Week 3: MongoDB logger for persisting LLM prompts, responses, and costs."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from dotenv import load_dotenv
from pymongo import MongoClient


class MongoLogger:
    """Persists LLM interaction telemetry records into a MongoDB collection."""

    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
        collection_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        load_dotenv()
        self.uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = db_name or os.getenv("MONGO_DB", "genai_exp")
        self.collection_name = collection_name or os.getenv(
            "MONGO_COLLECTION", "llm_logs"
        )
        self._client = client if client is not None else MongoClient(self.uri, serverSelectionTimeoutMS=3000)
        self._collection = self._client[self.db_name][self.collection_name]

    def log_interaction(
        self,
        prompt: str,
        response_text: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        total_cost_usd: float,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Insert a structured interaction record and return its stringified ID."""
        document: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "prompt": prompt,
            "response": response_text,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": total_tokens,
            },
            "cost_usd": total_cost_usd,
            "latency_ms": round(latency_ms, 2),
            "metadata": metadata or {},
        }
        result = self._collection.insert_one(document)
        return str(result.inserted_id)

    def fetch_recent(self, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve the most recent logged interactions."""
        cursor = self._collection.find().sort("timestamp_utc", -1).limit(limit)
        return list(cursor)

    def close(self) -> None:
        """Close the underlying MongoDB client connection."""
        self._client.close()
