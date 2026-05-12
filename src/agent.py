import os
import json
from urllib import response
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tools import search_funds, get_fund_details, calculate_return, list_all_funds
from google.genai.errors import ClientError

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- Tool definitions (tell Gemini what tools exist) ---

TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_funds",
                description="Search for fund information using a natural language query",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="Natural language search query about funds",
                        )
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_fund_details",
                description="Get detailed information about a specific fund by name",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "fund_name": types.Schema(
                            type=types.Type.STRING,
                            description="The exact or partial name of the fund",
                        )
                    },
                    required=["fund_name"],
                ),
            ),
            types.FunctionDeclaration(
                name="list_all_funds",
                description="List all funds available in the database",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={},
                ),
            ),
            types.FunctionDeclaration(
                name="calculate_return",
                description="Calculate compound investment return given principal, rate, and years",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "principal_usd": types.Schema(
                            type=types.Type.NUMBER,
                            description="Initial investment amount in USD",
                        ),
                        "annual_rate_percent": types.Schema(
                            type=types.Type.NUMBER,
                            description="Annual return rate as a percentage e.g. 12.5",
                        ),
                        "years": types.Schema(
                            type=types.Type.INTEGER,
                            description="Number of years to hold the investment",
                        ),
                    },
                    required=["principal_usd", "annual_rate_percent", "years"],
                ),
            ),
        ]
    )
]

# --- Tool dispatcher ---

TOOL_MAP = {
    "search_funds": search_funds,
    "get_fund_details": get_fund_details,
    "calculate_return": calculate_return,
    "list_all_funds": list_all_funds,
}


def run_tool(name: str, args: dict) -> str:
    """Execute whichever tool Gemini chose."""
    print(f"\n  [Tool called] {name}({args})")
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    return fn(**args)


# --- Agent loop ---


def ask(question: str) -> str:
    """
    Single-turn agent:
    1. Send question + tool definitions to Gemini
    2. Gemini decides which tool to call
    3. We execute the tool
    4. Send result back to Gemini
    5. Gemini returns final answer
    """
    print(f"\nQ: {question}")

    # Turn 1 — model decides which tool to call
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(tools=TOOLS),
        contents=question,
    )
    part = response.candidates[0].content.parts[0]

    # Did the model want to call a tool?
    if part.function_call:
        tool_name = part.function_call.name
        tool_args = dict(part.function_call.args)
        tool_result = run_tool(tool_name, tool_args)
        print(f"  [Tool Result] {tool_result}")
        # Turn 2 — send tool result back, get final answer
        final_response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(tools=TOOLS),
            contents=[
                question,
                response.candidates[0].content,
                types.Content(
                    role="tool",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name, response={"result": tool_result}
                            )
                        )
                    ],
                ),
            ],
        )
        return final_response.text

    # No tool call — model answered directly
    return part.text


if __name__ == "__main__":

    questions = [
        "What is the minimum investment for Brookfield?",
        "Which fund targets the highest return?",
        "If I invest $50,000 in Sequoia at 22% for 7 years, what's my return?",
        "Tell me about Blackstone's real estate fund",
    ]

    for q in questions:
        try:
            answer = ask(q)
            print(f"A: {answer}")
        except ClientError as e:
            if e.code == 429:
                print("Rate limit reached. Please wait a moment and try again.")
            else:
                raise
        print("-" * 60)
