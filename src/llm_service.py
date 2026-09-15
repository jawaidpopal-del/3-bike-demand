import os
import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def ask_model(prompt: str) -> str:
    """Send a prompt to the local Ollama LLM."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"LLM request failed: {response.text}")

    return response.json()["response"]


def explain_prediction(predicted_demand: float, hour: int, temperature: float) -> str:
    """Ask the local LLM to explain the bike-demand prediction."""
    prompt = f"""
You are helping a user understand a Seoul bike rental demand prediction.

The machine learning model predicted approximately {predicted_demand:.0f} rented bikes.

The input includes:
- Hour: {hour}
- Temperature: {temperature:.1f} Celsius

Explain what this prediction means in simple language and give one practical recommendation for someone planning to use the bike-sharing system.

Do not calculate or change the prediction. Use the prediction provided by the machine learning model.
"""

    return ask_model(prompt)