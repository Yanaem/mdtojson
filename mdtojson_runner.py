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


def get_invoice_id() -> str:
    """
    Récupère l'ID de facture à traiter.

    Priorité :
      1. premier argument de la ligne de commande
      2. variable d'env INVOICE_ID

    On plante explicitement si rien n'est fourni.
    """
    # 1) argument CLI (Cloud Run Jobs permet de passer des args au run)
    if len(sys.argv) > 1:
        return sys.argv[1]

    # 2) variable d'environnement
    env_val = os.getenv("INVOICE_ID")
    if env_val:
        return env_val

    raise SystemExit(
        "Aucun ID de facture fourni. "
        "Passe-le en premier argument du conteneur ou via la variable INVOICE_ID."
    )


def main() -> int:
    invoice_id = get_invoice_id()
    logger.info("Lancement de mdtojson pour invoice_id=%s", invoice_id)

    # DRY_RUN=true/1/yes/y => pas d'écriture en base
    dry_run_env = os.getenv("DRY_RUN", "false").lower()
    dry_run = dry_run_env in ("1", "true", "yes", "y")

    # Permet éventuellement d'override le modèle depuis l'env
    model = os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)

    try:
        analyzer = OCRAnalyzer(dry_run=dry_run, model=model)
        result = analyzer.analyze_invoice(invoice_id)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        logger.info("Job terminé avec succès")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erreur dans le job mdtojson: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
