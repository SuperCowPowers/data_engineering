# Project 4 — Using Claude (LLMs) from Python

The biggest leap yet. The first three projects *analyzed* data; this one *builds*
with AI. You'll call Claude from Python, give it a personality, and then turn it
into an **agent** that can use tools — search the news, save reminders, and act
on your behalf.

Four steps, each building on the last:

1. **Hello, Claude** — a chat loop: type a prompt, get a response, repeat.
2. **Make it your own** — give it a memory, pick its model, then have fun with swappable personalities.
3. **Claude as an agent** — give it tools so it can *do* things, not just talk.
4. **Level up storage** — swap the agent's flat file for a real SQLite database.

---

## Before you start: get an API key

Talking to Claude from code requires an **API key** — a secret password that
identifies your account and bills your usage.

1. **Account** — Set up your Anthropic account.
2. **Create the key** — sign in at [console.anthropic.com](https://console.anthropic.com),
   go to **Settings → API Keys**, and click **Create Key**. Copy it immediately —
   it's shown **only once**. It looks like `sk-ant-api03-XXXX…`.
3. **Add a little credit** — under **Billing**, add a small amount of credit
   (a few dollars goes a *long* way at these prices — see the cost table in
   Step 2). API calls are pay-as-you-go.

### ⚠️ The #1 rule: never put your API key in your code

A leaked key lets anyone spend your money. So:

- **Never** paste the key into a `.py` file, and **never** commit it to git.
- Instead, put it in an **environment variable** named `ANTHROPIC_API_KEY`. The
  Anthropic library reads that variable automatically — your code never mentions
  the key at all.
- Set it in your terminal before running:

  ```bash
  export ANTHROPIC_API_KEY="sk-ant-api03-...your key..."
  ```

  That lasts for the current terminal session. To make it stick, add that line
  to your `~/.zshrc`, **or** create a `project_4/.env` file (already ignored by
  git) and load it.
- **If a key ever leaks** (committed by accident, pasted in a screenshot),
  delete it in the Console and make a new one. Don't try to "hide" it — rotate it.

> Why so strict? Secrets in git are *forever* — even after you delete them, they
> live in the repo's history. The only safe move is to keep them out entirely.

---

## Setup

The `anthropic` library is already declared in the repo's `pyproject.toml`, so:

```bash
uv sync
export ANTHROPIC_API_KEY="sk-ant-api03-...your key..."   # this terminal session
```

Now you're ready.

---

## Step 1 — Hello, Claude

Create `project_4/hello_world.py`. The smallest possible call:

```python
import anthropic

# Reads your key from the ANTHROPIC_API_KEY environment variable.
# Notice the key is NOWHERE in this file. That's the point.
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)

# response.content is a LIST of blocks; the text is on the first one.
print(response.content[0].text)
```

Run it:

```bash
uv run python project_4/hello_world.py
```

### Now make it a chat loop

Wrap the call in a `while` loop that keeps going until you type `exit`:

```python
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
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    print(f"Claude: {response.content[0].text}\n")
```

Try it and notice something: **Claude has no memory here.** Tell it your name,
then ask "what's my name?" — it won't know. Each call is a blank slate, because
we send only the latest message. We fix that in Step 2.

> **A few things to understand:**
> - `max_tokens` caps how long the *reply* can be (1024 keeps replies short and
>   cheap while you're learning — raise it for longer answers).
> - `messages` is a list of `{"role": ..., "content": ...}` turns. Right now we
>   only ever send one.
> - `response.content[0].text` — the reply comes back as a list of "blocks";
>   for a plain answer the first block holds the text.

---

## Step 2 — Make it your own

Now turn that bare chat loop into a real bot. Create `project_4/custom_claude.py`.
Three upgrades — two practical, then one for fun.

### a) Give it a memory

Right now Claude forgets everything between messages. The fix: keep a running
`messages` list and append **both** sides of the conversation — your message
*and* Claude's reply — then send the whole list every time. The API itself is
stateless; *you* hold the history.

```python
messages = []
messages.append({"role": "user", "content": prompt})       # what you said
# ...call the API, get `answer`...
messages.append({"role": "assistant", "content": answer})  # what Claude said
```

### b) Let it pick a brain (model selection)

Set a `MODEL` constant and try different ones. They trade intelligence for
speed and cost:

| Model ID | Best for | Cost (per 1M tokens, in / out) |
|---|---|---|
| `claude-haiku-4-5` | fast, cheap, simple chat | $1 / $5 |
| `claude-sonnet-4-6` | balanced | $3 / $15 |
| `claude-opus-4-8` | the smartest answers | $5 / $25 |

> 💡 **Money tip:** while you're experimenting in a chat loop, set
> `MODEL = "claude-haiku-4-5"` — it's ~5× cheaper and noticeably faster. Switch
> to Opus when you want the best reasoning. A "token" is roughly ¾ of a word, so
> a few dollars of credit is thousands of messages.

### Putting a) and b) together

Here's the working bot — it remembers the conversation and runs on whichever
model you pick. The `system` prompt (its character) is a plain placeholder for
now; we make it fun in part c.

```python
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-4-8"          # try "claude-haiku-4-5" or "claude-sonnet-4-6"

SYSTEM = "You are a friendly, concise assistant."   # the bot's character (see part c)

messages = []                      # the running conversation
print("Chat with your bot! Type 'exit' to quit.\n")
while True:
    prompt = input("You: ")
    if prompt.strip().lower() == "exit":
        break

    messages.append({"role": "user", "content": prompt})

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM,             # the character (seeded into every request)
        messages=messages,         # the whole history
    )
    answer = response.content[0].text
    print(f"Bot: {answer}\n")

    messages.append({"role": "assistant", "content": answer})   # remember the reply
```

Now it remembers — tell it your name and ask later, it knows.

### c) Give it a personality (the fun part)

That `SYSTEM` string is a **pre-prompt**: it rides along with *every* request and
steers how Claude talks, without ever showing up in the chat. It's *the* lever
for personality — the more specific you are about voice, rules, and what to
avoid, the better it sticks.

Make it swappable with a menu of characters:

```python
PERSONAS = {
    "pirate": "You are Captain Cluck, a witty pirate first mate. Puns, slang, short answers.",
    "coach":  "You are a hype gym coach. ALL-CAPS energy, one-liners, relentlessly positive.",
    "tutor":  "You are a patient CS tutor. Explain with tiny examples and ask a guiding question instead of just handing over the answer.",
    "robot":  "You are a deadpan robot. Literal, monotone, occasionally baffled by human feelings.",
}

SYSTEM = PERSONAS["pirate"]        # swap the key to change character
```

Drop that in place of the plain `SYSTEM` line above and you've got a whole cast.
A few ways to push it further:

- **Let the user choose at startup:** `SYSTEM = PERSONAS[input("Pick a persona: ")]`.
- **Edit the character without touching code:** put the text in a
  `project_4/personality.txt` file and load it with
  `SYSTEM = open("personality.txt").read()`.
- **Show the voice with an example.** If the system prompt alone doesn't quite
  nail the tone, seed the `messages` list with one short example exchange before
  the real chat — Claude mirrors it:
  ```python
  messages = [
      {"role": "user", "content": "hi"},
      {"role": "assistant", "content": "Ahoy! Cap'n Cluck at yer service, ya scallywag."},
  ]
  ```

> **Heads-up — no `temperature`.** Older tutorials make a bot "more creative" by
> raising a `temperature` setting. The current Claude models (Opus 4.8 and
> friends) don't accept it — passing it causes an error. Personality lives in
> the **system prompt**, not a dial.

**What else can you customize?** Set `max_tokens` higher for essay-length
answers, or add `output_config={"effort": "low"}` to make Opus answer faster and
cheaper on easy questions.

---

## Step 3 — Claude as an agent

So far Claude can only *talk*. An **agent** can *act* — because you give it
**tools**. There are two flavors, and this step shows both.

### a) Tools that run on Anthropic's servers — summarize the news

Some tools are built in and run on Anthropic's side. **Web search** is one: you
just declare it, and Claude searches and reads the web for you. Perfect for
"summarize today's news," which Claude otherwise couldn't do (its training has a
cutoff date).

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=2048,
    messages=[{"role": "user",
               "content": "Search the web and summarize today's top 3 technology news stories."}],
    tools=[{"type": "web_search_20260209", "name": "web_search"}],
)

# Claude searches, reads results, and writes an answer — all in one call.
for block in response.content:
    if block.type == "text":
        print(block.text)
```

You didn't write any search code — Anthropic ran the searches and handed Claude
the results. (Web search costs a small amount per search on top of tokens.)

### b) Tools that **you** run — a reminder keeper

The more powerful idea: tools *you* implement. You describe a tool, Claude
decides when to call it, **you** run the code, and you hand the result back.
This back-and-forth is called the **agent loop**.

We'll give Claude two tools — save a reminder, and list reminders — backed by a
little JSON file.

```python
import json, os
import anthropic

client = anthropic.Anthropic()
REMINDERS_FILE = "reminders.json"

# --- the tools, described so Claude knows when to use them ---
tools = [
    {
        "name": "add_reminder",
        "description": "Save a reminder. Call this whenever the user asks to be reminded of something.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "What to remember"}},
            "required": ["text"],
        },
    },
    {
        "name": "list_reminders",
        "description": "List all of the user's saved reminders.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# --- the actual code behind each tool (this is YOUR part) ---
def load():
    if os.path.exists(REMINDERS_FILE):
        with open(REMINDERS_FILE) as f:
            return json.load(f)
    return []

def run_tool(name, tool_input):
    reminders = load()
    if name == "add_reminder":
        reminders.append(tool_input["text"])
        with open(REMINDERS_FILE, "w") as f:
            json.dump(reminders, f)
        return f"Saved: {tool_input['text']}"
    if name == "list_reminders":
        return "\n".join(f"{i+1}. {r}" for i, r in enumerate(reminders)) or "No reminders yet."
    return f"Unknown tool: {name}"

# --- the agent loop ---
def chat(user_message, messages):
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
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
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,   # must match the request
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
        # loop again so Claude can use the results to write its final reply

# --- the REPL ---
messages = []
print("Reminder bot! Try: 'remind me to feed the dog', then 'what are my reminders?'  (type exit)\n")
while True:
    user = input("You: ")
    if user.strip().lower() == "exit":
        break
    chat(user, messages)
```

**Read the loop carefully — this is the whole idea of an agent:**

1. You send the conversation. Claude replies with either an answer *or* a
   request to use a tool (`stop_reason == "tool_use"`).
2. If it's a tool request, **you** run the tool and append the result.
3. You call Claude again with the result. It might answer, or ask for another
   tool. Repeat until it's done.

That cycle — *think → act → observe → think again* — is what turns a chatbot
into an agent.

### c) Other cool stuff to try

- **Combine them.** Add the `web_search` tool *and* your reminder tools to the
  same bot, so you can say "what's the weather tomorrow? remind me to bring an
  umbrella if it'll rain."
- **More tools:** a calculator, a "read a file" tool, a "what time is it?" tool
  (return `datetime.now()`), a tool that writes a file.
- **A to-do agent:** add `complete_reminder` and `delete_reminder` tools.
- **Give it a personality too** — combine Step 2's `system` prompt with Step 3's
  tools for an assistant with character *and* abilities.

> **Tool safety:** tools let Claude trigger real actions. These examples only
> touch a local file, which is safe. Be thoughtful before giving an agent tools
> that send messages, spend money, or delete things — those deserve a
> "are you sure?" confirmation step.

---

## Step 4 — Level up: a real database

Your reminder bot already has persistent storage — `reminders.json` survives
restarts. But a flat file has limits: every save rewrites the *entire* file, and
to find "just the unfinished reminders" you'd have to load everything and filter
in Python. Time to graduate to a real database.

**SQLite** is perfect for this: a complete SQL database that lives in a single
file, and it's **built into Python** (`import sqlite3` — nothing to install).
Same core ideas as the big databases (Postgres, MySQL), small enough to learn in
an afternoon. You're walking the classic progression:

> in-memory list → JSON file → **real database**

### The schema

A database needs a *schema* — a definition of your table's columns. Reminders are
tabular (each has text, a done-flag, a timestamp), so give each its own column:

```python
import sqlite3

DB = "reminders.db"

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,  -- unique id, assigned for you
            text    TEXT NOT NULL,                      -- the reminder
            done    INTEGER NOT NULL DEFAULT 0,         -- 0 = todo, 1 = done (SQLite has no bool)
            created TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()
```

Call `init_db()` once at startup. `CREATE TABLE IF NOT EXISTS` makes it safe to
run every time.

### The tool functions, now backed by SQL

Same three-beat rhythm on every write: **connect → execute → commit & close.**
These are drop-in replacements for the JSON-file versions from Step 3:

```python
def add_reminder(text):
    conn = sqlite3.connect(DB)
    conn.execute("INSERT INTO reminders (text) VALUES (?)", (text,))
    conn.commit()
    conn.close()
    return f"Saved: {text}"

def list_reminders():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row          # read columns by name, like a dict
    rows = conn.execute("SELECT id, text, done FROM reminders ORDER BY created").fetchall()
    conn.close()
    if not rows:
        return "No reminders yet."
    return "\n".join(f"{r['id']}. [{'x' if r['done'] else ' '}] {r['text']}" for r in rows)

def complete_reminder(reminder_id):
    conn = sqlite3.connect(DB)
    conn.execute("UPDATE reminders SET done = 1 WHERE id = ?", (reminder_id,))
    conn.commit()
    conn.close()
    return f"Marked reminder {reminder_id} done."
```

`list_reminders` now shows a checkbox — and it could just as easily ask the
database for *only* the unfinished ones (`WHERE done = 0`), the kind of query a
flat file can't do without loading everything.

### ⚠️ The one rule of SQL: never build queries with f-strings

See those `?` marks? That's a **parameterized query** — you pass the values
separately and SQLite inserts them safely. It's tempting to write instead:

```python
conn.execute(f"INSERT INTO reminders (text) VALUES ('{text}')")   # ❌ NEVER do this
```

…but if someone's reminder is `'); DROP TABLE reminders; --`, that f-string turns
it into a command that **deletes your whole table**. It's called *SQL injection*,
and it's one of the most common security holes on the web. The `?` placeholder
treats input as *data, never as code*, so that same evil string just gets saved
as a harmless (if weird-looking) reminder. Always use `?`.

### Wire it into the agent

Two small changes to the Step 3 bot:

1. Call `init_db()` once at startup.
2. Add a `complete_reminder` tool so Claude can check things off:

```python
tools.append({
    "name": "complete_reminder",
    "description": "Mark a reminder as done, by its id number.",
    "input_schema": {
        "type": "object",
        "properties": {"reminder_id": {"type": "integer", "description": "The reminder's id"}},
        "required": ["reminder_id"],
    },
})
```

…and add one line to `run_tool` to dispatch it. Now you can say *"remind me to
call grandma,"* later *"what are my reminders?"*, then *"mark number 2 done"* —
and it all persists in a real database between runs.

> **Columns vs. a JSON blob.** You *could* store each reminder as a blob of JSON
> text in one column. That's a real technique — but save it for data that's
> genuinely messy or nested (like a whole saved conversation), where the fields
> change from row to row. For neat, tabular data like reminders, real columns
> win: they're what let you *query* — sort, filter, count — which is the entire
> reason to use a database instead of a file.

> **Don't commit the database.** `reminders.db` is local data, not code — the repo
> already ignores `project_4/*.db`.

---

## What you'll have learned

- How to call Claude from Python with the `anthropic` library — and why your
  API key lives in an **environment variable**, never in the code.
- How to hold a conversation (system prompt + a growing `messages` history) and
  how to **choose a model** for the right cost/speed/intelligence tradeoff.
- The **agent loop**: giving Claude tools, letting it decide when to use them,
  running them yourself, and feeding the results back — the foundation of every
  AI agent, including the one helping you write this code.
- **Persistent storage done right**: moving from a flat file to a real **SQLite**
  database — schemas, columns vs. JSON blobs, and why every SQL query uses `?`
  placeholders instead of f-strings (SQL injection).
