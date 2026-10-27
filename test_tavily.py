
import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables from .env
load_dotenv()

# Read the API key without hardcoding it
api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError(
        "TAVILY_API_KEY was not found. Check your .env file."
    )

# Create a Tavily client
client = TavilyClient(api_key=api_key)

# Search the web
print("Searching the web...", flush=True)

response = client.search(
    query="Agentic AI applications in software engineering",
    max_results=5,
    search_depth="basic",
)

# Display the results
results = response.get("results", [])

print(f"\nFound {len(results)} results.\n")

for index, result in enumerate(results, start=1):
    print(f"{index}. {result.get('title', 'Untitled')}")
    print(f"URL: {result.get('url', 'No URL')}")
    print(result.get("content", "No content available"))
    print("-" * 60)
