import os
import json
import time
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from dotenv import load_dotenv
from models.FundSummary import FundSummary

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def load_fund_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def summarise_fund(fund_text: str) -> FundSummary:
    from prompts import SYSTEM_PROMPT

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                ),
                contents=f"Summarise this fund:\n\n{fund_text}"
            )
            break
        except ServerError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))

    raw = response.text
    data = json.loads(raw)
    return FundSummary(**data)

if __name__ == "__main__":
    fund_text = load_fund_text("data/funds.txt")
    result = summarise_fund(fund_text)

    print("\n=== Fund Summary ===")
    print(f"Name:        {result.name}")
    print(f"Strategy:    {result.strategy}")
    print(f"Risk level:  {result.risk_level}")
    print(f"Min invest:  ${result.minimum_investment_usd:,}")
    print(f"Summary:     {result.summary}")
    print("\nRaw JSON:")
    print(result.model_dump_json())