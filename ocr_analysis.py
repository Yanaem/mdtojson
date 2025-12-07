#!/usr/bin/env python3
"""
OCR Analysis Script - Équivalent Python de l'edge function extract-invoice-data
Ce script reproduit exactement le flux "Analyse OCR" de l'application Lovable.

Usage:
    python ocr_analysis.py --invoice-id <UUID>
    python ocr_analysis.py --invoice-id <UUID> --dry-run
    python ocr_analysis.py --invoice-id <UUID> --model claude-sonnet-4-5

Prérequis:
    pip install supabase anthropic python-dotenv

Variables d'environnement requises:
    SUPABASE_URL=https://rqlohkiciaonesrvdijn.supabase.co
    SUPABASE_SERVICE_ROLE_KEY=<votre_service_role_key>
    ANTHROPIC_API_KEY=<votre_anthropic_api_key>
"""

import os
import sys
import json
import re
import time
import argparse
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

try:
    from supabase import create_client, Client
    from anthropic import Anthropic
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Installez les dépendances avec: pip install supabase anthropic python-dotenv")
    sys.exit(1)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://rqlohkiciaonesrvdijn.supabase.co')
SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Modèle Claude par défaut (même que l'edge function)
DEFAULT_MODEL = "claude-sonnet-4-5"

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2  # secondes


class OCRAnalyzer:
    """Classe principale pour l'analyse OCR des factures."""
    
    def __init__(self, dry_run: bool = False, model: str = DEFAULT_MODEL):
        self.dry_run = dry_run
        self.model = model
        
        if not SUPABASE_SERVICE_ROLE_KEY:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY non défini")
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY non défini")
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        self.anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
        
        logger.info(f"OCRAnalyzer initialisé (dry_run={dry_run}, model={model})")
    
    def get_invoice(self, invoice_id: str) -> Dict[str, Any]:
        """Récupère une facture depuis Supabase."""
        logger.info(f"Récupération de la facture {invoice_id}")
        
        response = self.supabase.table('invoices').select('*').eq('id', invoice_id).single().execute()
        
        if not response.data:
            raise ValueError(f"Facture {invoice_id} non trouvée")
        
        return response.data
    
    def get_company(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations d'une entreprise."""
        if not company_id:
            return None
        
        logger.info(f"Récupération de l'entreprise {company_id}")
        
        response = self.supabase.table('companies').select('*').eq('id', company_id).single().execute()
        return response.data
    
    def get_ocr_job(self, company_id: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Trouve le job OCR correspondant à la facture."""
        logger.info(f"Recherche du job OCR pour {file_path}")
        
        # Extraire le nom du fichier depuis file_path
        file_name = file_path.split('/')[-1] if '/' in file_path else file_path
        
        # Chercher le job OCR avec markdown généré
        response = self.supabase.table('ocr_jobs') \
            .select('*') \
            .eq('company_id', company_id) \
            .eq('status', 'markdown_generated') \
            .order('created_at', desc=True) \
            .execute()
        
        if not response.data:
            logger.warning("Aucun job OCR trouvé avec status 'markdown_generated'")
            return None
        
        # Chercher le job qui correspond au fichier
        for job in response.data:
            job_file = job.get('input_pdf_path', '').split('/')[-1]
            if job_file == file_name:
                logger.info(f"Job OCR trouvé: {job['id']}")
                return job
        
        # Si pas de correspondance exacte, prendre le plus récent
        logger.info(f"Pas de correspondance exacte, utilisation du job le plus récent: {response.data[0]['id']}")
        return response.data[0]
    
    def download_markdown(self, markdown_path: str) -> str:
        """Télécharge le contenu markdown depuis Supabase Storage."""
        logger.info(f"Téléchargement du markdown: {markdown_path}")
        
        # Nettoyer le chemin
        clean_path = markdown_path
        if clean_path.startswith('markdown-files/'):
            clean_path = clean_path[len('markdown-files/'):]
        
        response = self.supabase.storage.from_('markdown-files').download(clean_path)
        
        if not response:
            raise ValueError(f"Impossible de télécharger le markdown: {markdown_path}")
        
        content = response.decode('utf-8')
        logger.info(f"Markdown téléchargé: {len(content)} caractères")
        return content
    
    def get_supplier_template(self, siren: Optional[str]) -> Optional[Dict[str, Any]]:
        """Cherche un template de prompt spécifique au fournisseur."""
        if not siren:
            return None
        
        logger.info(f"Recherche de template pour SIREN: {siren}")
        
        response = self.supabase.table('supplier_prompt_templates') \
            .select('*') \
            .eq('siren', siren) \
            .eq('is_active', True) \
            .single() \
            .execute()
        
        if response.data:
            logger.info(f"Template trouvé: {response.data.get('supplier_name')}")
        
        return response.data
    
    def get_default_prompt(self) -> str:
        """Récupère le prompt par défaut 'Full Analysis Prompt Supplier'."""
        logger.info("Récupération du prompt par défaut")
        
        response = self.supabase.table('claude_prompts') \
            .select('*') \
            .eq('prompt_name', 'Full Analysis Prompt Supplier') \
            .eq('is_active', True) \
            .single() \
            .execute()
        
        if not response.data:
            raise ValueError("Prompt 'Full Analysis Prompt Supplier' non trouvé")
        
        return response.data['prompt_content']
    
    def inject_variables(self, prompt: str, company: Optional[Dict[str, Any]]) -> str:
        """Injecte les variables dynamiques dans le prompt."""
        if not company:
            return prompt
        
        activity = company.get('activity', '')
        company_name = company.get('company_name', '')
        
        # Remplacer les variables
        prompt = prompt.replace('{activité}', activity or '')
        prompt = prompt.replace('{activity}', activity or '')
        prompt = prompt.replace('{company_name}', company_name or '')
        prompt = prompt.replace('{nom_société}', company_name or '')
        
        logger.info(f"Variables injectées: activité='{activity}', company_name='{company_name}'")
        return prompt
    
    def call_claude(self, prompt: str, markdown_content: str) -> Tuple[str, Dict[str, Any]]:
        """Appelle l'API Claude avec retry et backoff exponentiel."""
        
        full_prompt = f"{prompt}\n\n---\n\nContenu du document (Markdown):\n\n{markdown_content}"
        
        logger.info(f"Appel Claude ({self.model}) - Prompt: {len(prompt)} chars, Markdown: {len(markdown_content)} chars")
        
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                
                response = self.anthropic.messages.create(
                    model=self.model,
                    max_tokens=8000,
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ]
                )
                
                duration = time.time() - start_time
                
                # Extraire le texte de la réponse
                text_content = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        text_content += block.text
                
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "duration_seconds": round(duration, 2)
                }
                
                logger.info(f"Réponse Claude reçue en {duration:.1f}s - {usage['input_tokens']} input, {usage['output_tokens']} output tokens")
                
                return text_content, usage
                
            except Exception as e:
                error_str = str(e)
                
                # Vérifier si c'est une erreur retryable
                is_retryable = any(code in error_str for code in ['429', '500', '502', '503', '529', 'overloaded'])
                
                if is_retryable and attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Erreur retryable (tentative {attempt + 1}/{MAX_RETRIES}): {e}")
                    logger.info(f"Nouvelle tentative dans {delay}s...")
                    time.sleep(delay)
                else:
                    raise
        
        raise Exception("Nombre maximum de tentatives atteint")
    
    def parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse le JSON depuis la réponse Claude (même s'il est entouré de texte)."""
        logger.info("Parsing de la réponse JSON")
        
        # Essayer de parser directement
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Chercher un bloc JSON dans le texte
        # Pattern 1: ```json ... ```
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Pattern 2: { ... } (objet JSON)
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Pattern 3: [ ... ] (array JSON)
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.warning("Impossible de parser le JSON, retour du texte brut")
        return {"raw_response": text}
    
    def update_invoice(self, invoice_id: str, extraction_data: Dict[str, Any], 
                       existing_data: Optional[Dict[str, Any]] = None) -> None:
        """Met à jour la facture avec les données extraites."""
        
        # Préserver les champs critiques
        if existing_data:
            if 'pages' in existing_data:
                extraction_data['pages'] = existing_data['pages']
            if 'invoice_id' in existing_data:
                extraction_data['invoice_id'] = existing_data['invoice_id']
        
        update_data = {
            'extraction_data': extraction_data,
            'validation_status': 'extracted',
            'updated_at': datetime.utcnow().isoformat()
        }
        
        # Extraire les champs principaux si présents
        if 'invoice_number' in extraction_data:
            update_data['invoice_number'] = extraction_data['invoice_number']
        if 'invoice_date' in extraction_data:
            update_data['invoice_date'] = extraction_data['invoice_date']
        if 'amount_ttc' in extraction_data:
            try:
                update_data['amount_ttc'] = float(extraction_data['amount_ttc'])
            except (ValueError, TypeError):
                pass
        if 'supplier_name' in extraction_data:
            update_data['client_name'] = extraction_data['supplier_name']
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Mise à jour de la facture {invoice_id}:")
            logger.info(json.dumps(update_data, indent=2, default=str, ensure_ascii=False))
        else:
            logger.info(f"Mise à jour de la facture {invoice_id}")
            self.supabase.table('invoices').update(update_data).eq('id', invoice_id).execute()
            logger.info("Facture mise à jour avec succès")
    
    def analyze_invoice(self, invoice_id: str) -> Dict[str, Any]:
        """Analyse complète d'une facture - fonction principale."""
        
        logger.info("=" * 60)
        logger.info(f"Démarrage de l'analyse OCR pour la facture: {invoice_id}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # 1. Récupérer la facture
        invoice = self.get_invoice(invoice_id)
        company_id = invoice.get('company_id')
        file_path = invoice.get('file_path')
        existing_extraction = invoice.get('extraction_data')
        
        logger.info(f"Facture trouvée: {file_path}")
        
        # 2. Récupérer l'entreprise
        company = self.get_company(company_id) if company_id else None
        
        # 3. Trouver le job OCR et récupérer le markdown
        ocr_job = self.get_ocr_job(company_id, file_path)
        
        if not ocr_job:
            raise ValueError("Aucun job OCR trouvé pour cette facture")
        
        markdown_path = ocr_job.get('markdown_path')
        if not markdown_path:
            raise ValueError("Le job OCR n'a pas de fichier markdown")
        
        # 4. Télécharger le markdown
        markdown_content = self.download_markdown(markdown_path)
        
        # 5. Chercher un template spécifique au fournisseur
        supplier_siren = None
        if existing_extraction and isinstance(existing_extraction, dict):
            supplier_siren = existing_extraction.get('supplier_siren')
        
        template = self.get_supplier_template(supplier_siren)
        
        # 6. Récupérer le prompt
        if template:
            prompt = template['prompt_template']
            logger.info(f"Utilisation du template fournisseur: {template['supplier_name']}")
        else:
            prompt = self.get_default_prompt()
            logger.info("Utilisation du prompt par défaut")
        
        # 7. Injecter les variables dynamiques
        prompt = self.inject_variables(prompt, company)
        
        # 8. Appeler Claude
        response_text, usage = self.call_claude(prompt, markdown_content)
        
        # 9. Parser la réponse JSON
        extraction_data = self.parse_json_response(response_text)
        
        # 10. Mettre à jour la facture
        self.update_invoice(invoice_id, extraction_data, existing_extraction)
        
        total_duration = time.time() - start_time
        
        result = {
            "success": True,
            "invoice_id": invoice_id,
            "duration_seconds": round(total_duration, 2),
            "usage": usage,
            "extraction_data": extraction_data,
            "used_template": template['supplier_name'] if template else None
        }
        
        logger.info("=" * 60)
        logger.info(f"Analyse terminée en {total_duration:.1f}s")
        logger.info(f"Tokens utilisés: {usage['input_tokens']} input, {usage['output_tokens']} output")
        logger.info("=" * 60)
        
        return result


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Analyse OCR d'une facture avec Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python ocr_analysis.py --invoice-id 123e4567-e89b-12d3-a456-426614174000
    python ocr_analysis.py --invoice-id 123e4567-e89b-12d3-a456-426614174000 --dry-run
    python ocr_analysis.py --invoice-id 123e4567-e89b-12d3-a456-426614174000 --model claude-sonnet-4-5

Variables d'environnement:
    SUPABASE_URL              URL du projet Supabase
    SUPABASE_SERVICE_ROLE_KEY Clé service role Supabase
    ANTHROPIC_API_KEY         Clé API Anthropic
        """
    )
    
    parser.add_argument(
        '--invoice-id',
        required=True,
        help='UUID de la facture à analyser'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Mode simulation (ne modifie pas la base de données)'
    )
    
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help=f'Modèle Claude à utiliser (défaut: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mode verbose (affiche plus de détails)'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        analyzer = OCRAnalyzer(dry_run=args.dry_run, model=args.model)
        result = analyzer.analyze_invoice(args.invoice_id)
        
        print("\n" + "=" * 60)
        print("RÉSULTAT DE L'ANALYSE")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        
        return 0
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
