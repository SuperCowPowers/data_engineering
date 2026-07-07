import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-4-8"

print("Chat with Claude! Type 'exit' to quit.\n")

while True:
    prompt = input("You: ")

    if prompt.strip().lower() == "exit":
        print("Goodbye!")
        break

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    print(f"Claude: {response.content[0].text}\n")
