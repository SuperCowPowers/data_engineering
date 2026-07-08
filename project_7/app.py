"""
app.py - a local web page that classifies a dog photo you drag in and shows the
result three different ways. It loads the model computed in Project 6.

    uv run python app.py          # then open http://127.0.0.1:8050

The plumbing is done for you: loading the model, the web page, and the callback
that runs when you drop a photo. YOUR job is the four TODOs below. TODO 1 is the
exact same inference you wrote in Project 6's predict.py, just handing its result
back as a DataFrame instead of printing it; TODOs 2-4 draw three charts from that
DataFrame. Until you fill them in, the app starts but raises NotImplementedError
the moment you upload a photo. See the README for a walkthrough of each TODO.
"""

import torch
import pandas as pd  # noqa: F401  — you'll use these three once you finish the TODOs
import plotly.express as px  # noqa: F401
import plotly.graph_objects as go  # noqa: F401
from torchvision.models import ResNet18_Weights
from dash import Dash, dcc, html, Input, Output, no_update

import drag_drop_utils as ddu  # the shared drag/drop plumbing (files, web images, paste)

# --- the model + preprocessing (must match how Project 6 trained) ---
MODEL_PATH = "../project_6/dog_model.pt"
TRANSFORM = ResNet18_Weights.DEFAULT.transforms()

model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()
BREEDS = model.breeds  # Project 6 saved the class names inside the model, so this
#                        list follows however many breeds you trained on -- no edits here

TOP_N = 5  # the bar + radar show only the most likely few (all 25 would be a mess)


# ===== TODO 1: classify() -- this IS Project 6's predict.py inference =====
# The first three lines are exactly the core you wrote in predict.py: preprocess the
# image, run it through the model with gradients off, softmax the scores into
# probabilities. predict.py then sorted those into a printed text list; here you do
# the one new thing -- pack the same numbers into a DataFrame so the charts below can
# read them. Same inference, different last mile: a table instead of a printout.
#
#   x = TRANSFORM(image).unsqueeze(0)                 # <- same as predict.py
#   with torch.no_grad():                             # <- same as predict.py
#       probs = torch.softmax(model(x), dim=1)[0]     # <- same as predict.py: prob per breed
#   df = pd.DataFrame({"breed": BREEDS, "probability": probs.tolist()})   # the new bit
#   return df.sort_values("probability", ascending=False, ignore_index=True)   # winner first
def classify(image):
    """PIL image -> DataFrame[breed, probability] -- predict.py's inference, as a table."""
    raise NotImplementedError("TODO 1: build the DataFrame")


# ===== TODO 2: horizontal bar chart (top few only) =====
# Plotly Express takes a DataFrame plus column names and does the rest. With 25
# breeds a full chart is a mess, so show just the most likely `TOP_N`. `df` is
# already sorted, so `df.head(TOP_N)` IS the top few. `orientation="h"` makes the
# bars horizontal; px draws the first row at the BOTTOM, so sort ascending to get
# the winning breed on top.
#
#   top = df.head(TOP_N)
#   fig = px.bar(top.sort_values("probability"), x="probability", y="breed",
#                orientation="h", title=f"Top {TOP_N} breeds")
#   fig.update_xaxes(tickformat=".0%", range=[0, 1])   # show 0-100%, full scale
#   return fig
def bar_chart(df):
    """DataFrame -> horizontal bar chart of the TOP_N most likely breeds."""
    raise NotImplementedError("TODO 2: build the bar chart")


# ===== TODO 3: radar chart (top few only) =====
# The same TOP_N numbers, drawn as spokes on a wheel. A confident prediction is a
# single spike; a confused one is a lumpy blob. This is `px.line_polar` -- the
# radar cousin of px.bar. `r` is the distance out each spoke, `theta` picks the
# spoke, and line_close=True joins the last point back to the first. Use the same
# `df.head(TOP_N)` as the bar chart -- 25 spokes would be unreadable.
#
#   fig = px.line_polar(df.head(TOP_N), r="probability", theta="breed",
#                       line_close=True, title=f"Top {TOP_N} radar", range_r=[0, 1])
#   fig.update_traces(fill="toself")                   # shade the enclosed area
#   return fig
def radar_chart(df):
    """DataFrame -> radar (polar) chart of the TOP_N most likely breeds."""
    raise NotImplementedError("TODO 3: build the radar chart")


# ===== TODO 4: confidence gauge =====
# A single dial for the TOP breed. Unlike the other two, this chart needs just
# ONE number, not the whole table -- a nice reminder that a plot can read one
# cell (`df.iloc[0]`) or the entire DataFrame. This one uses graph_objects (go)
# because Plotly Express has no gauge.
#
#   top = df.iloc[0]                                   # first row = most likely (we sorted it)
#   fig = go.Figure(go.Indicator(
#       mode="gauge+number",
#       value=top["probability"] * 100,               # 0-100 for a percentage dial
#       number={"suffix": "%"},
#       title={"text": f"Confidence: {top['breed']}"},
#       gauge={"axis": {"range": [0, 100]}},
#   ))
#   return fig
def confidence_gauge(df):
    """DataFrame -> a single confidence dial for the top breed."""
    raise NotImplementedError("TODO 4: build the gauge")


# ----------------------------------------------------------------------------
# Everything below is provided plumbing -- read it, but you don't need to edit it.
# The drag/drop logic (files, web images, paste) lives in drag_drop_utils.py +
# assets/dropzone.js, so app.py and app_completed.py can share it.
# ----------------------------------------------------------------------------

app = Dash(__name__)

app.layout = html.Div(
    style={
        "maxWidth": "1180px",
        "margin": "40px auto",
        "fontFamily": "system-ui, sans-serif",
    },
    children=[
        html.H1("🐕 Dog Breed Classifier"),
        html.P(f"The model knows {len(BREEDS)} dog breeds."),
        ddu.dropzone(),  # one box that takes a file, a web image, a click, or a paste
        html.Div(id="result", style={"marginTop": "24px"}),
    ],
)


@app.callback(Output("result", "children"), Input(ddu.DROPPED_ID, "data"), prevent_initial_call=True)
def on_drop(value):
    """Runs whenever a photo is dropped/pasted/chosen: get the image, classify, chart."""
    if not value:
        return no_update
    try:
        # same model, same inference as Project 6's predict.py -- load_image() just
        # turns the dropped file-or-URL into a PIL image first
        image, thumb = ddu.load_image(value)
    except Exception as e:  # bad link, 404, not an image, ...
        return html.P(f"Couldn't load that image: {e}", style={"color": "crimson"})

    df = classify(image)  # TODO 1: one prediction -> one DataFrame
    top = df.iloc[0]["breed"]

    # a chart sitting in a flex cell, so the three sit side by side and wrap on
    # a narrow window instead of stacking
    def cell(figure):
        return html.Div(dcc.Graph(figure=figure), style={"flex": "1 1 320px", "minWidth": "300px"})

    return [
        # a small copy of the photo you dropped in, next to the verdict
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "16px"},
            children=[
                html.Img(src=thumb, style={"maxHeight": "160px", "borderRadius": "8px"}),
                html.H2(f"→ {top}"),
            ],
        ),
        # the three views on one line, each fed the same df
        html.Div(
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "12px",
                "marginTop": "16px",
            },
            children=[
                cell(bar_chart(df)),
                cell(radar_chart(df)),
                cell(confidence_gauge(df)),
            ],
        ),
    ]


if __name__ == "__main__":
    app.run(debug=True)
