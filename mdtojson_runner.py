#!/usr/bin/env python3
import os
import sys
import json
import logging

from ocr_analysis import OCRAnalyzer, DEFAULT_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_markdown_content(analyzer: OCRAnalyzer) -> str:
    """
    Récupère le markdown à analyser.

    Priorité :
      1. MARKDOWN_PATH  -> chemin dans le bucket Supabase 'markdown-files'
      2. MARKDOWN       -> markdown brut passé en variable d'environnement

    Si rien n'est fourni -> exit(1).
    """
    markdown_path = os.getenv("MARKDOWN_PATH")
    if markdown_path:
        logger.info("Utilisation du markdown depuis Supabase Storage: %s", markdown_path)
        return analyzer.download_markdown(markdown_path)

    markdown_env = os.getenv("MARKDOWN")
    if markdown_env:
        logger.info("Utilisation du markdown depuis la variable d'environnement MARKDOWN")
        return markdown_env

    raise SystemExit(
        "Aucun markdown fourni. "
        "Définis soit MARKDOWN_PATH (chemin dans 'markdown-files'), "
        "soit MARKDOWN (contenu markdown brut)."
    )


def main() -> int:
    # DRY_RUN garde la même logique (utile plus tard si on veut réécrire en base)
    dry_run_env = os.getenv("DRY_RUN", "false").lower()
    dry_run = dry_run_env in ("1", "true", "yes", "y")

    # Permet d'override le modèle via variable d'env
    model = os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)

    try:
        logger.info("Initialisation de OCRAnalyzer (mode markdown direct)")
        analyzer = OCRAnalyzer(dry_run=dry_run, model=model)

        # 1. Récupérer le markdown
        markdown_content = get_markdown_content(analyzer)
        logger.info("Markdown chargé (%d caractères)", len(markdown_content))

        # 2. Récupérer éventuellement l'entreprise pour injecter l'activité dans le prompt
        company = None
        company_id = os.getenv("COMPANY_ID")
        if company_id:
            logger.info("Récupération de l'entreprise %s pour injection dans le prompt", company_id)
            company = analyzer.get_company(company_id)

        # 3. Récupérer le prompt principal dans la table claude_prompts
        prompt = analyzer.get_default_prompt()

        # 4. Injection des variables (activité, nom de la société, etc.)
        prompt = analyzer.inject_variables(prompt, company)

        # 5. Appel Claude
        text_response, usage = analyzer.call_claude(prompt, markdown_content)

        # 6. Parsing du JSON renvoyé par Claude
        json_data = analyzer.parse_json_response(text_response)

        # ⚠️ On n'écrit rien en base ici : on renvoie juste le JSON
        # On affiche UNIQUEMENT le JSON sur stdout
        print(json.dumps(json_data, ensure_ascii=False, indent=2))

        logger.info(
            "Job mdtojson terminé avec succès (input_tokens=%s, output_tokens=%s)",
            usage.get("input_tokens"),
            usage.get("output_tokens"),
        )
        return 0

    except Exception as exc:  # noqa: BLE001
        logger.exception("Erreur dans le job mdtojson: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
