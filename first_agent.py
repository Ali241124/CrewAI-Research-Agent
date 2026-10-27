
from crewai import Agent, Task, Crew, LLM

# 1. Connect CrewAI to our local Ollama model
llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.3,
)

# 2. Create our first AI agent
ai_tutor = Agent(
    role="AI Tutor",
    goal="Explain artificial intelligence concepts simply and accurately",
    backstory=(
        "You are a patient AI tutor who teaches beginners. "
        "You explain difficult technical concepts using simple "
        "language and practical examples."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# 3. Define the agent's task
learning_task = Task(
    description=(
        "Explain what an AI agent is, how it differs from a "
        "traditional chatbot, and give one practical example. "
        "Use simple language suitable for a beginner."
    ),
    expected_output=(
        "A beginner-friendly explanation with a definition, "
        "a comparison, and one practical example."
    ),
    agent=ai_tutor,
)

# 4. Create a crew containing our agent and task
crew = Crew(
    agents=[ai_tutor],
    tasks=[learning_task],
    verbose=True,
)

# 5. Execute the task and display the result
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n===== FINAL ANSWER =====\n")
    print(result)
