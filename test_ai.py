import os
import sys
from dotenv import load_dotenv

sys.path.append(r"e:\plant diseases detection\plant-disease-backend")
load_dotenv(r"e:\plant diseases detection\plant-disease-backend\.env")

from utils.ai_helper import get_disease_info
import requests
import json
from config import Config

def test_openrouter():
    prompt = """
You are an agricultural expert.

The detected plant disease is: Apple Scab

IMPORTANT:
- Only explain THIS disease
- Do NOT explain anything unrelated
- If the name is not a valid plant disease, say "Unknown disease"

Return ONLY valid JSON:
{
  "description": "...",
  "causes": ["...", "..."],
  "remedies": ["...", "..."]
}
"""
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
        print("Status:", response.status_code)
        
        try:
            result = response.json()
            print("parse success")
        except Exception as e:
            print("parse error:", e)
            print("raw text:", repr(response.text))
            return
            
        try:
            content = result["choices"][0]["message"]["content"]
            print("content:", repr(content))
            
            content = content.replace("```json", "").replace("```", "").strip()
            # find first '{' and last '}' to handle any extra text
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != -1:
                content = content[start:end]
            
            data = json.loads(content)
            print("Data loaded successfully:", data)
        except Exception as e:
            print("Error parsing content:", e)
            print("Content was:", repr(result.get("choices", [])))
            
    except Exception as e:
        print("Fallback failed:", e)

test_openrouter()
