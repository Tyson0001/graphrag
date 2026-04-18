#!/usr/bin/env python3
"""
Migration script for GraphRAG v2.0 -> v3.0 (text-embedding-3-large).
Wipes the database and re-ingests documents to ensure embedding consistency.
"""
import argparse
import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from core.graph_db import graph_db
from ingestion.document_processor import document_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("migration")

def clear_database():
    """Clear all data from the database."""
    try:
        logger.warning("⚠️  CLEARING ALL DATABASE DATA for migration...")
        with graph_db._get_driver().session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("✅ Database cleared!")
        return True
    except Exception as e:
        logger.error(f"❌ Database clear failed: {e}")
        return False

def setup_indexes():
    """Re-setup indexes (CRITICAL for vector search)."""
    try:
        logger.info("Re-initializing indexes...")
        graph_db.setup_indexes()
        logger.info("✅ Indexes initialized!")
        return True
    except Exception as e:
        logger.error(f"❌ Index setup failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Migrate GraphRAG to text-embedding-3-large")
    parser.add_argument("--input-dir", "-d", type=Path, required=True, help="Directory containing documents to re-ingest")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process directories recursively")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        logger.error(f"Directory not found: {args.input_dir}")
        sys.exit(1)

    print(f"\n🚀 MIGRATION STARTING")
    print(f"====================")
    print(f"Target Model: {settings.embedding_model}")
    print(f"Chunk Size:   {settings.chunk_size}")
    print(f"Input Dir:    {args.input_dir}")
    print(f"====================\n")
    print("THIS WILL DELETE ALL EXISTING DATA IN NEO4J.")
    
    if not args.yes:
        confirm = input("Are you sure you want to proceed? (yes/no): ")
        if confirm.lower() != "yes":
            print("Migration cancelled.")
            sys.exit(0)

    # 1. Clear DB
    if not clear_database():
        sys.exit(1)
        
    # 2. Setup Indexes
    if not setup_indexes():
        sys.exit(1)
        
    # 3. Re-ingest
    logger.info(f"Starting re-ingestion from {args.input_dir}...")
    results = document_processor.process_directory(args.input_dir, args.recursive)
    
    successful = sum(1 for result in results if result.get("status") == "success")
    total = len(results)
    
    print(f"\n✅ MIGRATION COMPLETE")
    print(f"--------------------")
    print(f"Files processed: {successful}/{total}")
    print(f"New Model:       {settings.embedding_model}")
    print(f"--------------------\n")
    
    if successful < total:
        print("Note: Some files failed to process. Check logs for details.")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
