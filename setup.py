# Setup Script for Physician Notetaker
# This script helps with initial setup and dependency installation

import subprocess
import sys
import os

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def run_command(command, description):
    print(f"📦 {description}...")
    try:
        subprocess.check_call(command, shell=True)
        print(f"✅ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}\n")
        return False

def main():
    print_header("🩺 Physician Notetaker - Setup Script")
    
    # Check Python version
    print("🐍 Checking Python version...")
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {python_version.major}.{python_version.minor}")
        sys.exit(1)
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected\n")
    
    # Upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing dependencies")
    else:
        print("⚠️  requirements.txt not found!")
    
    # Download SpaCy model
    print_header("📥 Downloading Medical NLP Models")
    
    spacy_model_url = "https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz"
    run_command(f"{sys.executable} -m pip install {spacy_model_url}", "Installing SciSpacy medical model")
    
    # Verify installation
    print_header("🔍 Verifying Installation")
    
    verification_script = """
import spacy
import transformers
import torch
print("✅ SpaCy imported successfully")
print("✅ Transformers imported successfully")
print("✅ PyTorch imported successfully")

try:
    nlp = spacy.load("en_core_sci_md")
    print("✅ Medical NLP model loaded successfully")
except:
    print("⚠️  Medical NLP model not found - please install manually")
"""
    
    try:
        exec(verification_script)
    except Exception as e:
        print(f"⚠️  Verification failed: {e}")
    
    # Setup complete
    print_header("✅ Setup Complete!")
    print("You can now run the application:")
    print("  python main.py --demo")
    print("\nOr open the Jupyter notebook:")
    print("  jupyter notebook demo.ipynb")
    print()

if __name__ == "__main__":
    main()
