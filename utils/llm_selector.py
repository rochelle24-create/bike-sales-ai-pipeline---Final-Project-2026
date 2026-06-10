# utils/llm_selector.py
# AI Backend Selection - Ollama or Anthropic
# Author: Rachel Barazani - AI Developer
# Course: AI Developer Program - Hebrew University 2026

import os
import requests
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()


def get_llm() -> LLM:
    """
    Prompt user to select AI backend at startup.
    Returns configured LLM instance passed to all agents.
    """
    print("\n" + "+" + "=" * 58 + "+")
    print("|" + "BIKE SALES AI PIPELINE".center(58) + "|")
    print("|" + "HEBREW UNIVERSITY 2026 FINAL PROJECT".center(58) + "|")
    print("|" + "Rachel Barazani - AI Developer".center(58) + "|")
    print("+" + "=" * 58 + "+")
    print("|  Select AI backend:                                      |")
    print("|                                                          |")
    print("|  [1] Local  - Ollama        (free, runs offline)         |")
    print("|  [2] Cloud  - Anthropic Haiku   (~$0.05/run)             |")
    print("|  [3] Cloud  - Anthropic Sonnet  (~$0.20/run)             |")
    print("|                                                          |")
    print("|  Your API key is never stored in GitHub.                 |")
    print("|  It is read from your local .env file only.              |")
    print("+" + "=" * 58 + "+")

    choice = os.getenv("LLM_CHOICE") or input("\nEnter 1, 2 or 3: ").strip()

    if choice == "1":
        return _setup_ollama()
    elif choice == "2":
        return _setup_anthropic("claude-haiku-4-5-20251001", "Haiku")
    elif choice == "3":
        return _setup_anthropic("claude-sonnet-4-20250514", "Sonnet")
    else:
        print("  Invalid choice - defaulting to Ollama.")
        return _setup_ollama()


def _setup_ollama() -> LLM:
    """Configure and validate Ollama local LLM"""

    # Check Ollama is running
    try:
        requests.get("http://localhost:11434", timeout=3)
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "\n[X] Ollama is not running.\n"
            "   Start it with: ollama serve\n"
            "   Then pull a model: ollama pull llama3\n"
        )

    print("\n  Available models: llama3, mistral, codellama, gemma")
    model = os.getenv("OLLAMA_MODEL") or input("  Model name (press Enter for llama3): ").strip()
    model = model if model else "llama3"

    # Verify model is available
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"].split(":")[0]
                     for m in response.json().get("models", [])]
        if model not in available:
            raise ValueError(
                f"\n[X] Model '{model}' not found in Ollama.\n"
                f"   Pull it with: ollama pull {model}\n"
                f"   Available models: {', '.join(available)}\n"
            )
    except requests.exceptions.RequestException:
        pass  # Skip check if tags endpoint unavailable

    llm = LLM(
        model=f"ollama/{model}",
        base_url="http://localhost:11434"
    )

    print(f"\n  [OK] Using Ollama - {model}")
    return llm


def _setup_anthropic(model: str, model_name: str) -> LLM:
    """Configure and validate Anthropic API LLM"""

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("\n  No API key found in .env file.")
        api_key = input("  Enter your Anthropic API key: ").strip()

    # Validate key format
    if not api_key.startswith("sk-ant-"):
        raise ValueError(
            "\n[X] Invalid Anthropic API key format.\n"
            "   Key must start with: sk-ant-\n"
            "   Get your key at: https://console.anthropic.com\n"
        )

    # Save to .env for future runs
    if not os.getenv("ANTHROPIC_API_KEY"):
        with open(".env", "a") as f:
            f.write(f"\nANTHROPIC_API_KEY={api_key}")
        print("  [OK] API key saved to .env for future runs")

    llm = LLM(
        model=model,
        api_key=api_key
    )

    print(f"\n  [OK] Using Anthropic {model_name} - {model}")
    return llm


def get_backend_name() -> str:
    """
    Returns a readable backend name for logging and display.
    Called after get_llm() to label artifacts and reports.
    """
    return os.getenv("SELECTED_BACKEND", "Unknown")