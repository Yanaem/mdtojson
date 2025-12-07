#!/usr/bin/env python3
import os
import json
import logging
import datetime as dt

from google.cloud import storage

from ocr_analysis import OCRAnalyzer, DEFAULT_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_markdown_content(analyzer: OCRAnalyzer) -> str:
    """Récupère le markdown à analyser (MARKDOWN_PATH ou MARKDOWN)."""
    markdown_path = os.getenv("MARKDOWN_PATH")
    if markdown_path:
        logger.info("Markdown via Supabase Storage: %s", markdown_path)
        return analyzer.download_markdown(markdown_path)

    markdown_env = os.getenv("MARKDOWN")
    if markdown_env:
        logger.info("Markdown via variable d'environnement MARKDOWN")
        return markdown_env

    raise SystemExit(
        "Aucun markdown fourni. "
        "Définis MARKDOWN_PATH (chemin dans 'markdown-files') "
        "ou MARKDOWN (contenu brut)."
    )


def compute_gcs_object_name() -> str:
    """Nom du fichier JSON dans le bucket GCS."""
    explicit = os.getenv("GCS_OBJECT")
    if explicit:
        return explicit

    markdown_path = os.getenv("MARKDOWN_PATH")
    if markdown_path:
        base = os.path.basename(markdown_path)
        if base.lower().endswith(".md"):
            base = base[:-3]
        return f"{base}.json"

    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"result-{ts}.json"


def upload_json_to_gcs(json_data: dict) -> str:
    """Upload du JSON dans le bucket GCS. Retourne gs://bucket/objet."""
    bucket_name = os.getenv("GCS_BUCKET", "mdtojson")
    object_name = compute_gcs_object_name()

    logger.info("Upload du JSON vers GCS: bucket=%s, object=%s", bucket_name, object_name)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    payload = json.dumps(json_data, ensure_ascii=False, indent=2)
    blob.upload_from_string(payload, content_type="application/json")

    full_path = f"gs://{bucket_name}/{object_name}"
    logger.info("JSON écrit dans %s", full_path)
    return full_path


def main() -> int:
    dry_run_env = os.getenv("DRY_RUN", "false").lower()
    dry_run = dry_run_env in ("1", "true", "yes", "y")
    model = os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)

    try:
        logger.info("Init OCRAnalyzer (mode markdown direct)")
        analyzer = OCRAnalyzer(dry_run=dry_run, model=model)

        markdown_content = get_markdown_content(analyzer)
        logger.info("Markdown chargé (%d caractères)", len(markdown_content))

        # On ne fait plus de substitution ici : le prompt est déjà préparé côté edge
        prompt = analyzer.get_default_prompt()  # lit PROMPT_GCS_URI si défini

        # Appel Claude (avec thinking, puisque call_claude a été modifié dans ocr_analysis.py)
        text_response, usage = analyzer.call_claude(prompt, markdown_content)

        json_data = analyzer.parse_json_response(text_response)

        gcs_path = upload_json_to_gcs(json_data)

        print(json.dumps(json_data, ensure_ascii=False, indent=2))

        logger.info(
            "mdtojson OK (input_tokens=%s, output_tokens=%s, gcs_path=%s)",
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            gcs_path,
        )
        return 0

    except Exception as exc:  # noqa: BLE001
        logger.exception("Erreur dans le job mdtojson: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
