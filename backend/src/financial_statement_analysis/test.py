from crewai import LLM

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
)

# Test with a simple query
response = llm.call("What is artificial intelligence?")
print(response)