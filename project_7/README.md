# Project 7 — Dog Breed Web App (Dash + Plotly)

The victory lap. Project 6 trained a model and saved it to a file — but a model
sitting in a `.pt` file isn't much fun. This project puts a **friendly face** on
it: a local web page where you **drag in a dog photo** and instantly see, three
different ways, how likely it is to be each breed.

This is a different kind of skill from the last six projects — not modeling, but
**turning a model into something a person can actually use.** That last mile is a
huge part of real data-engineering work.

We'll use the model computed in Project 6 (`../project_6/dog_model.pt`) — no
retraining here.

## What you're building

A little web app running on your own machine:

```
┌───────────────────────────────────────────┐
│  🐕 Dog Breed Classifier                     │
│  ┌─────────────────────────────────────┐  │
│  │  Drag & drop a dog photo here          │  │  ← you drop a photo
│  └─────────────────────────────────────┘  │
│  [ your photo ]   → Pug                      │  ← thumbnail + verdict
│                                              │
│  ▸ bar chart    (probability per breed)      │  ┐
│  ▸ radar chart  (same 5 numbers, as spokes)  │  ├ three views of ONE table
│  ▸ gauge        (confidence in the winner)   │  ┘
└───────────────────────────────────────────┘
```

## The tools: Dash + Plotly

- **Dash** lets you build a web app in **pure Python** — no HTML or JavaScript
  required. You describe the page as Python objects and wire up behavior with
  "callbacks."
- **Plotly** draws the interactive charts (they zoom and show tooltips for free).

Both came in with `uv add dash`, so a plain `uv sync` gets you set.

## How Dash works (this part is provided — read it)

`app.py` already has the plumbing written for you: it loads the model, builds the
page, and runs a callback when you drop a photo. Two concepts make the whole thing
tick.

**a) The layout — the page as Python objects.** No HTML; you build the page out of
components like `html.Div`, `html.H1`, `dcc.Upload`, and `dcc.Graph`. The two that
matter are the drop zone and an empty `result` box the callback will fill in:

```python
dcc.Upload(id="upload", ...),                 # the drag & drop target
html.Div(id="result"),                        # starts empty; the callback fills it
```

**b) The callback — the reactive heart of Dash.** A callback says *"when this input
changes, run my function and put whatever it returns into that output."*

```python
@app.callback(Output("result", "children"), Input("upload", "contents"))
def on_upload(contents):
    ...                       # decode the image, classify it, build the charts
```

Read the decorator as a sentence: **when `upload`'s `contents` change → run
`on_upload` → drop its return value into `result`'s `children`.** That's the entire
reactive model:

```
you drop a photo  ──►  contents changes  ──►  on_upload() runs  ──►  result updates
```

## The big idea: one DataFrame, many views

Here's the design worth stealing for your own projects. The model spits out five
raw numbers. Instead of juggling those numbers directly, we pour them into a tidy
**pandas DataFrame** — one row per breed — *once*:

```
      breed  probability
0       Pug     0.991531
1     Boxer     0.004548
2 Shiba Inu     0.003560
3    Beagle     0.000275
4   Samoyed     0.000085
```

Then **every chart is built from that same DataFrame.** That's the trick to keeping
this kind of code clean: get your data into a good table first, and each view
becomes two or three lines. You already know DataFrames from Projects 1–5 — this is
the same muscle, now feeding a web page.

## Write `app.py` — the four TODOs

Open `app.py`. The plumbing is done; the four functions near the top are stubbed
out with `raise NotImplementedError`. Fill them in **in order** — the app will
start but error the moment you upload a photo until all four are done. Each stub
also has the hints repeated in its comments, so you can work right in the file.

### TODO 1 — `classify(image)`: prediction → DataFrame

This is the hand-off point. Run the image through the model (exactly like Project 6:
`transform → model → softmax`, inside the `with torch.no_grad():` block), then pack
the result into a two-column DataFrame and sort it so the winner is on top:

```python
x = TRANSFORM(image).unsqueeze(0)                 # preprocess + add a batch dimension
with torch.no_grad():                             # same no-gradients block as Project 6
    probs = torch.softmax(model(x), dim=1)[0]     # 5 scores -> 5 probabilities, summing to 1
df = pd.DataFrame({"breed": BREEDS, "probability": probs.tolist()})
return df.sort_values("probability", ascending=False, ignore_index=True)
```

Everything after this touches only `df` — never a tensor again.

### TODO 2 — `bar_chart(df)`: the honest workhorse

**Plotly Express** (`px`) is the easy way to chart a DataFrame: hand it the table
and the column names and it does the rest. `px` draws the first row at the *bottom*,
so sort ascending to put the winning breed on top:

```python
fig = px.bar(df.sort_values("probability"), x="probability", y="breed",
             orientation="h", title="Breed probability")
fig.update_xaxes(tickformat=".0%", range=[0, 1])   # show 0–100% on a full scale
return fig
```

Notice you never pulled numbers out by hand — you named columns and `px` read the
DataFrame. That's the pattern for the next two.

### TODO 3 — `radar_chart(df)`: the same numbers, a different shape

`px.line_polar` is the radar cousin of `px.bar`. Same DataFrame, same idea — `r` is
how far out each spoke goes, `theta` picks the spoke:

```python
fig = px.line_polar(df, r="probability", theta="breed", line_close=True,
                    title="Probability radar", range_r=[0, 1])
fig.update_traces(fill="toself")                   # shade the enclosed area
return fig
```

A confident prediction looks like a single spike; a confused one is a lumpy blob.
The radar is genuinely good at showing *shape* at a glance.

### TODO 4 — `confidence_gauge(df)`: one number, one dial

Unlike the other two, this chart needs just **one** number — a nice reminder that a
plot can read a single cell (`df.iloc[0]`) or the whole table. Gauges live in
`plotly.graph_objects` (`go`) because Express doesn't have one:

```python
top = df.iloc[0]                                   # first row = most likely (we sorted it)
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=top["probability"] * 100,                # 0–100 for a percentage dial
    number={"suffix": "%"},
    title={"text": f"Confidence: {top['breed']}"},
    gauge={"axis": {"range": [0, 100]}},
))
return fig
```

That's all four. The provided callback already wires them together — it calls
`classify()`, then feeds the resulting `df` to each chart and shows your photo's
thumbnail next to the verdict.

## Run it

```bash
uv run python app.py          # from inside project_7/
```

You'll see `Dash is running on http://127.0.0.1:8050/`. Open that in your browser,
drag in a dog photo, and watch all three charts (and your thumbnail) appear. Change
a function, save, and Dash hot-reloads the page automatically.

## Why a French bulldog scores high on "Pug"

The model's last step is a **softmax**, which spreads a probability across *all
five* breeds that always sums to 100%. For a clear Pug, almost all the mass piles
on "Pug." But the model only *knows* these five breeds — so a photo of something
it's never seen gets sorted into the nearest look-alike:

- a **French bulldog** → high **Pug** (flat face, compact body),
- a **husky** → high **Shiba Inu**,
- a **golden retriever** → probably spread across Beagle/Boxer.

Seeing the *whole distribution* (not just the top guess) is exactly why the charts
are more honest than a single label — the radar's shape and the gauge's needle both
tell you how sure (or unsure) the model really is.

## Make it yours

Once all four work, extend it. Notice how each idea is a change to **the DataFrame
or one chart** — never a rewrite:

1. **Top 3 only.** In the bar chart, show just the three most likely breeds. Hint:
   it's a one-liner on the DataFrame — `df.head(3)`.
2. **A fourth view.** You have the DataFrame; try `px.pie(df, values="probability",
   names="breed")` for a donut.
3. **A confidence note.** If the top probability is below, say, 60%, add an
   `html.P("not sure — is this even one of my 5 breeds?")` to the result list.
4. **Grow it.** Add breeds to Project 6's `data_setup.py`, retrain, and update the
   `BREEDS` list here to match. More breeds, same app — the radar just grows more
   spokes on its own.

## What you'll have learned

- How to wrap a trained model in a **web app** with Dash — the "last mile" that
  turns a model into a usable tool.
- The **callback** model: input → function → output.
- The **one-DataFrame-many-views** pattern: shape your data into a tidy table once,
  then let each chart read from it. This scales to dashboards with a dozen panels.
- Two ways to chart: **Plotly Express** (`px`, hand it a DataFrame) and
  **graph_objects** (`go`, build a figure piece by piece).
- How projects compose: Project 6 produces an artifact; Project 7 consumes it.
