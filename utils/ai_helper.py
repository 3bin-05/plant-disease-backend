import requests
from config import Config
import json

cache = {}

def get_disease_info(disease_name):

    if disease_name in cache:
        return cache[disease_name]

    prompt = f"""
You are an agricultural expert.

The detected plant disease is: {disease_name}

IMPORTANT:
- Only explain THIS disease
- Do NOT explain anything unrelated
- If the name is not a valid plant disease, say "Unknown disease"

Return ONLY valid JSON:
{{
  "description": "...",
  "causes": ["...", "..."],
  "remedies": ["...", "..."]
}}
"""

    # =========================
    # TRY OPENAI FIRST
    # =========================
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}]
            }
        )

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)

        cache[disease_name] = data
        return data

    except Exception as e:
        print("OpenAI failed:", e)

    # =========================
    # FALLBACK → OPENROUTER
    # =========================
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}]
            }
        )

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)

        cache[disease_name] = data
        return data

    except Exception as e:
        print("Fallback failed:", e)

    # FINAL FAILSAFE
    return {
        "description": "Could not fetch details.",
        "causes": [],
        "remedies": []
    }