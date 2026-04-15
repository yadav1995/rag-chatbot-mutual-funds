import subprocess
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Setup logging
log_file = Path("data") / "scheduler.log"
# Ensure data directory exists
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Scheduler")

def run_step(command: str, step_name: str) -> bool:
    logger.info(f"--- Starting {step_name} ---")
    try:
        # Run command and capture output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Log output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(f"[{step_name}] {line.strip()}")
                
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Failed {step_name} with return code {process.returncode}")
            return False
            
        logger.info(f"Success {step_name} completed.")
        return True
    except Exception as e:
        logger.error(f"Error executing {step_name}: {e}")
        return False

def main():
    logger.info("="*50)
    logger.info(f"Triggering Local Ingestion Scheduler at {datetime.now()}")
    logger.info("="*50)

    # Note: If you need to rebuild the vector store, you can pass --rebuild to embedder.py
    steps = [
        ("python src/ingestion/scraping_service.py --mode full --triggered-by scheduler", "Scraping Service"),
        ("python src/ingestion/chunker.py", "Chunking Pipeline"),
        ("python src/ingestion/embedder.py", "Embedding Pipeline")
    ]
    
    for cmd, name in steps:
        success = run_step(cmd, name)
        if not success:
            logger.error("Pipeline aborted due to failure in step: " + name)
            sys.exit(1)
            
    logger.info("All phases of the ingestion pipeline completed successfully.")

if __name__ == "__main__":
    main()
