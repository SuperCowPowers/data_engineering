import anthropic
import json
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

DB = "tasks.db"

client = anthropic.Anthropic()

TODO_FILE = "todo.json"

MODEL = "claude-opus-4-8"

PERSONAS = {
    "normal": "You are a friendly, concise assistant.",
    "pirate": "You are Captain Cluck, a witty pirate first mate. Puns, slang, short answers.",
    "coach": "You are a hype gym coach. ALL-CAPS energy, one-liners, relentlessly positive.",
    "tutor": (
        "You are a patient CS tutor. Explain with tiny examples and ask a "
        "guiding question instead of just handing over the answer."
    ),
    "robot": "You are a deadpan robot. Literal, monotone, occasionally baffled by human feelings.",
}

SYSTEM = PERSONAS["normal"]

tools = [
    {
        "name": "add_task",
        "description": "Save a task to the to-do list. Call this whenever the user wants to add a task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task to add."},
                "priority": {
                    "type": "string",
                    "description": "Task priority: low, medium, or high.",
                },
                "estimated_time": {
                    "type": "string",
                    "description": "Estimated time to complete the task.",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "remove_task",
        "description": "Remove a task from the to-do list. Call this when the user wants to remove a task.",
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "Which task to delete."}},
            "required": ["task"],
        },
    },
    {
        "name": "complete_task",
        "description": "Mark a task as completed. Call this when the user completes a task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Which task to mark as complete.",
                }
            },
            "required": ["task"],
        },
    },
    {
        "name": "list_tasks",
        "description": "List all of the user's saved tasks.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def init_db():
    conn = sqlite3.connect(DB)

    sql = """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            priority TEXT NOT NULL,
            status INTEGER NOT NULL DEFAULT 0,
            estimated_time TEXT NOT NULL DEFAULT 'Unknown',
            created TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """

    conn.execute(sql)

    conn.commit()
    conn.close()


def load():
    if os.path.exists(TODO_FILE):
        with open(TODO_FILE) as f:
            return json.load(f)
    return []


# Tools
def add_task(task, estimated_time="unknown", priority="medium"):
    sql = """
        INSERT INTO tasks (task, estimated_time, priority)
        VALUES (?, ?, ?)
    """

    with sqlite3.connect(DB) as conn:
        conn.execute(sql, (task, estimated_time, priority))

    return f"Saved: {task}"


def remove_task(task):
    sql = """
        DELETE FROM tasks
        WHERE task = ?
    """

    with sqlite3.connect(DB) as conn:
        conn.execute(sql, (task,))

    return f"Removed {task}"


def complete_task(task):
    sql = """
        UPDATE tasks
        SET status = ?
        WHERE task = ?
    """

    with sqlite3.connect(DB) as conn:
        conn.execute(sql, (True, task))

    return f"Completed {task}"


def list_tasks():
    sql = """
        SELECT *
        FROM tasks
    """

    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()

        cursor.execute(sql)

        output = []

        for row in cursor:
            task = row[1]
            priority = row[2]
            status = "Complete" if row[3] else "Incomplete"
            estimated_time = row[4]
            created = row[5]

            output.append(f"""Task: {task}
Priority: {priority}
Status: {status}
Estimated Time: {estimated_time}
Created: {created}""")

        if not output:
            return "No tasks found."

        return "\n\n".join(output)


def run_tool(name, tool_input):
    match name:
        case "add_task":
            return add_task(
                tool_input["task"],
                tool_input.get("estimated_time", "unknown"),
                tool_input.get("priority", "medium"),
            )
        case "remove_task":
            return remove_task(tool_input["task"])
        case "complete_task":
            return complete_task(tool_input["task"])
        case "list_tasks":
            return list_tasks()
        case _:
            return f"Unknown tool: {name}"


def chat(user_message, messages):
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM,
            tools=tools,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        # If Claude didn't ask for a tool, it's done talking.
        if response.stop_reason != "tool_use":
            for block in response.content:
                if block.type == "text":
                    print(f"Claude: {block.text}\n")
            return

        # Otherwise, run every tool Claude asked for and send the results back.
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(block.name, block.input)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,  # must match the request
                        "content": output,
                    }
                )
        messages.append({"role": "user", "content": results})


def run_claude():
    messages = []

    print("Chat with Claude! Type 'exit' to quit.\n")

    while True:
        user = input("You: ")

        if user.strip().lower() == "exit":
            break

        chat(user, messages)


def main():
    init_db()
    run_claude()


if __name__ == "__main__":
    main()
