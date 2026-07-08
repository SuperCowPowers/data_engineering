# Project 7 — Dog Breed Web App (Dash + Plotly)

The victory lap. Project 6 trained a model and saved it to a file — but a model
sitting in a `.pt` file isn't much fun. This project puts a **friendly face** on
it: a local web page where you **drag in a dog photo** and instantly see, three
different ways, how likely it is to be each breed.

This is a different kind of skill from the last six projects — not modeling, but
**turning a model into something a person can actually use.** That last mile is a
huge part of real data-engineering work.

We'll use the model computed in Project 6 (`../project_6/dog_model.pt`) — no
retraining here. In fact the part that actually classifies a photo **is Project 6's
`predict.py`**, barely changed: same model, same inference. This project just wraps
it in a web page and turns its numbers into charts instead of a text printout.

## What you're building

A little web app running on your own machine:

```
┌───────────────────────────────────────────┐
│  🐕 Dog Breed Classifier                     │
│  ┌─────────────────────────────────────┐  │
│  │   Drag a dog photo here                │  │  ← one box, four ways in
│  │   (file · web image · click · paste)   │  │
│  └─────────────────────────────────────┘  │
│  [ your photo ]   → Pug                      │  ← thumbnail + verdict
│                                              │
│  ▸ bar chart    (probability per breed)      │  ┐
│  ▸ radar chart  (same 5 numbers, as spokes)  │  ├ three views of ONE table
│  ▸ gauge        (confidence in the winner)   │  ┘
└───────────────────────────────────────────┘
```

**One drop zone, four ways to give it a photo:** drag a **local file**, drag an
image straight off a **web page** (Google Images), **click** to pick a file, or
**paste** an image or an image URL. There's a subtlety worth knowing: a file drop
hands the browser the actual *bytes*, but dragging an image off a web page only
hands over its *URL* — so the app fetches that URL itself. All of this lives in the
provided plumbing (`drag_drop_utils.py` + `assets/dropzone.js`); your four TODOs
don't change.

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
components like `html.Div`, `html.H1`, and `dcc.Graph`. The drop zone comes from the
provided `drag_drop_utils.dropzone()` (a styled box plus a hidden `dcc.Store` that
the browser writes the dropped photo into), and there's an empty `result` box the
callback fills in:

```python
ddu.dropzone(),                               # the drop box + a dcc.Store("dropped-image")
html.Div(id="result"),                        # starts empty; the callback fills it
```

**b) The callback — the reactive heart of Dash.** A callback says *"when this input
changes, run my function and put whatever it returns into that output."* Here the
input is the Store the drop zone writes to:

```python
@app.callback(Output("result", "children"), Input(ddu.DROPPED_ID, "data"))
def on_drop(value):
    ...      # get a PIL image from the dropped file-or-URL, classify it, build the charts
```

Read the decorator as a sentence: **when the Store's `data` changes → run `on_drop`
→ drop its return value into `result`'s `children`.** That's the entire reactive
model:

```
you drop / paste a photo  ──►  the Store's data changes  ──►  on_drop() runs  ──►  result updates
```

(How does a *browser* drag end up in a Python `dcc.Store`? The provided
`assets/dropzone.js` listens for the raw drop/paste/click, works out the file or
URL, and hands it to Dash. That's the one job plain `dcc.Upload` couldn't do — it
only accepts local files, not images dragged off the web. Open your browser's
console to watch its `[dropzone]` debug lines.)

## The big idea: one DataFrame, many views

Here's the design worth stealing for your own projects. The model spits out one raw
number per breed. Instead of juggling those numbers directly, we pour them into a
tidy **pandas DataFrame** — one row per breed — *once* (25 rows, sorted, top few
shown):

```
                        breed  probability
0                         Pug     0.999
1                       Boxer     0.000
2  Staffordshire Bull Terrier     0.000
...                                       (22 more, all ~0)
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

### TODO 1 — `classify(image)`: this *is* your `predict.py`

Look closely: the first three lines are **exactly the inference core you wrote in
Project 6's `predict.py`** — preprocess the image, run it through the model with
gradients off, softmax the scores into probabilities:

```python
x = TRANSFORM(image).unsqueeze(0)                 # <- same as predict.py
with torch.no_grad():                             # <- same as predict.py
    probs = torch.softmax(model(x), dim=1)[0]     # <- same as predict.py: prob per breed
df = pd.DataFrame({"breed": BREEDS, "probability": probs.tolist()})   # the one new line
return df.sort_values("probability", ascending=False, ignore_index=True)
```

In `predict.py` the next step was `sorted(zip(BREEDS, probs...))` and a `print`. Here
you do the **one new thing** instead: pack those same numbers into a DataFrame and
return it. Same inference, different last mile — a table the charts can read rather
than text in a terminal. Everything after this touches only `df`, never a tensor.

### TODO 2 — `bar_chart(df)`: the honest workhorse

**Plotly Express** (`px`) is the easy way to chart a DataFrame: hand it the table
and the column names and it does the rest. With 25 breeds, though, a full bar chart
is a wall of near-zero bars — so show only the most likely `TOP_N` (a constant set
up top). `df` is already sorted, so `df.head(TOP_N)` *is* the top few. `px` draws
the first row at the *bottom*, so sort ascending to put the winner on top:

```python
top = df.head(TOP_N)
fig = px.bar(top.sort_values("probability"), x="probability", y="breed",
             orientation="h", title=f"Top {TOP_N} breeds")
fig.update_xaxes(tickformat=".0%", range=[0, 1])   # show 0–100% on a full scale
return fig
```

Notice you never pulled numbers out by hand — you named columns and `px` read the
DataFrame. That's the pattern for the next two.

### TODO 3 — `radar_chart(df)`: the same numbers, a different shape

`px.line_polar` is the radar cousin of `px.bar`. Same DataFrame, same idea — `r` is
how far out each spoke goes, `theta` picks the spoke. Feed it the **same
`df.head(TOP_N)`** as the bar chart; 25 spokes would be an unreadable snowflake:

```python
fig = px.line_polar(df.head(TOP_N), r="probability", theta="breed", line_close=True,
                    title=f"Top {TOP_N} radar", range_r=[0, 1])
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

The model's last step is a **softmax**, which spreads a probability across *all 25*
breeds that always sums to 100%. For a clear Pug, almost all the mass piles on
"Pug." But the model only *knows* these 25 breeds — so a photo of something it's
never seen gets sorted into the nearest look-alike:

- a **French bulldog** → high **Pug** (flat face, compact body),
- a **husky** → high **Shiba Inu**,
- a **golden retriever** → probably spread across Beagle/Boxer.

Seeing the *whole distribution* (not just the top guess) is exactly why the charts
are more honest than a single label — the radar's shape and the gauge's needle both
tell you how sure (or unsure) the model really is.

## Make it yours

Once all four work, extend it. Notice how each idea is a change to **the DataFrame
or one chart** — never a rewrite:

1. **Tune the shortlist.** The bar and radar already show `TOP_N = 5`. Bump it to 8,
   or make the donut below use *all* breeds. What's the right default when the model
   is unsure and the probability is spread thin across many breeds?
2. **A fourth view.** You have the DataFrame; try `px.pie(df.head(TOP_N),
   values="probability", names="breed")` for a donut.
3. **A confidence note.** If the top probability is below, say, 60%, add an
   `html.P("not sure — is this even one of my breeds?")` to the result list.
4. **Grow or shrink it.** Change the `BREEDS` list in Project 6's `data_setup.py`
   and retrain. Because Project 6 saves the breed names *inside* the model, this app
   picks them up automatically on reload — nothing to edit here. The charts just
   grow or shrink to match.

## What you'll have learned

- How to wrap a trained model in a **web app** with Dash — the "last mile" that
  turns a model into a usable tool.
- The **callback** model: input → function → output.
- The **one-DataFrame-many-views** pattern: shape your data into a tidy table once,
  then let each chart read from it. This scales to dashboards with a dozen panels.
- Two ways to chart: **Plotly Express** (`px`, hand it a DataFrame) and
  **graph_objects** (`go`, build a figure piece by piece).
- How projects compose: Project 6 produces an artifact; Project 7 consumes it.
