
import json
import time
import urllib.request
import urllib.error

url = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen2.5:7b",
    "stream": False,
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a concise research writer. Use only the "
                "provided evidence. Do not invent facts or sources."
            ),
        },
        {
            "role": "user",
            "content": """
Write a Markdown report of about 200 words.

Topic: Practical applications of deep learning in healthcare.

Evidence:
[S1] FPT Software summary: Applications include drug discovery,
clinical decision support, and medical imaging. Challenges include
data quality, workflow integration, transparency, regulation, and
clinician trust.

[S2] npj Digital Medicine: A 2026 systematic review concerns
prospective real-world implementation of deep learning systems
in healthcare. The available summary does not provide its findings.

[S3] Paubox summary: Discusses healthcare AI implementation and
predictive analytics, but the supplied excerpt gives limited detail.

Use these headings:
Introduction, Applications, Limitations, Conclusion.

Cite supported statements using [S1], [S2], and [S3].
Do not invent findings from the review. Explain evidence gaps.
"""
        },
    ],
    "options": {
        "temperature": 0.1,
        "num_predict": 500,
    },
}

request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

print("Sending a direct request to Ollama...")
start = time.time()

try:
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))

    print(f"\nResponse received in {time.time() - start:.1f} seconds.")
    print("\n--- GENERATED REPORT ---\n")
    print(result["message"]["content"])

except urllib.error.URLError as error:
    print(f"\nConnection/request failed: {error}")

except Exception as error:
    print(f"\nError: {type(error).__name__}: {error}")
