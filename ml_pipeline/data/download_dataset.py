"""
Download Netflix dataset from Kaggle using kagglehub
"""
import kagglehub
from pathlib import Path
import shutil
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | <level>{message}</level>")

def download_dataset():
    """Download Netflix titles dataset from Kaggle"""
    try:
        logger.info("📥 Downloading Netflix dataset from Kaggle...")
        
        # Download latest version
        path = kagglehub.dataset_download("venkateshsuvarna27/netflix-title")
        
        logger.success(f"✅ Dataset downloaded to: {path}")
        
        # Create data directory structure
        data_dir = Path("./data")
        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"
        
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy dataset to our data directory
        source_path = Path(path)
        for file in source_path.glob("*.csv"):
            dest = raw_dir / file.name
            shutil.copy(file, dest)
            logger.info(f"📄 Copied {file.name} to {dest}")
        
        logger.success("✅ Dataset setup complete!")
        return str(raw_dir)
        
    except Exception as e:
        logger.error(f"❌ Error downloading dataset: {e}")
        logger.info("💡 Make sure you have Kaggle credentials configured")
        logger.info("   Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        logger.info("   Or place kaggle.json in ~/.kaggle/")
        raise

if __name__ == "__main__":
    download_dataset()
