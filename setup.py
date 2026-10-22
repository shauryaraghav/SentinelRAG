#!/usr/bin/env python3
"""
setup.py – One-shot setup script for Privacy-Preserving RAG System
===================================================================
Run: python setup.py
This installs all dependencies and downloads the spaCy model.
"""

import subprocess
import sys
import os


def run(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  ⚠️  Command exited with code {result.returncode} — continuing...")
    else:
        print(f"  ✅  Done.")


def main():
    print("\n🛡️  Privacy-Preserving RAG System – Setup\n")

    # 1. Install requirements
    run(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies…"
    )

    # 2. Download spaCy model
    run(
        f"{sys.executable} -m spacy download en_core_web_lg",
        "Downloading spaCy English model (en_core_web_lg)…"
    )

    # 3. Fallback to smaller model
    run(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading spaCy small model as fallback (en_core_web_sm)…"
    )

    # 4. Create .env if it doesn't exist
    if not os.path.exists(".env"):
        import shutil
        shutil.copy(".env.example", ".env")
        print("\n✅  Created .env from .env.example — add your API keys!")
    else:
        print("\n✅  .env already exists.")

    print("\n" + "="*60)
    print("  Setup complete! 🎉")
    print("="*60)
    print("\nNext steps:")
    print("  1. Edit .env and add your GEMINI_API_KEY or OPENAI_API_KEY")
    print("  2. Run the server:")
    print("     python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print("  3. Open http://localhost:8000 in your browser")
    print("  4. API docs: http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    main()
