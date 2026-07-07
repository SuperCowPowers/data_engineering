"""
app.py - a local web page that classifies a dog photo you drag in and shows the
result three different ways. It loads the model computed in Project 6.

    uv run python app.py          # then open http://127.0.0.1:8050

The plumbing is done for you: loading the model, the web page, and the callback
that runs when you drop a photo. YOUR job is the four TODOs below — turning the
prediction into a DataFrame and drawing three charts from it. Until you fill them
in, the app will start but raise NotImplementedError the moment you upload a photo.
See the README for a walkthrough of each TODO.
"""

import base64
import io

import torch
import pandas as pd  # noqa: F401  — you'll use these three once you finish the TODOs
import plotly.express as px  # noqa: F401
import plotly.graph_objects as go  # noqa: F401
from PIL import Image
from torchvision.models import ResNet18_Weights
from dash import Dash, dcc, html, Input, Output, no_update

# --- the model + preprocessing (must match how Project 6 trained) ---
MODEL_PATH = "../project_6/dog_model.pt"
BREEDS = ["Beagle", "Boxer", "Pug", "Samoyed", "Shiba Inu"]  # same order as training
TRANSFORM = ResNet18_Weights.DEFAULT.transforms()

model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()


# ===== TODO 1: turn one image into a DataFrame =====
# Run the image through the model and return a DataFrame with two columns,
# `breed` and `probability`, one row per breed. This is the hand-off point:
# every chart below reads this DataFrame, never the raw model output.
#
#   x = TRANSFORM(image).unsqueeze(0)                 # preprocess + add a batch dimension
#   with torch.no_grad():                             # same no-gradients block as Project 6
#       probs = torch.softmax(model(x), dim=1)[0]     # 5 scores -> 5 probabilities that sum to 1
#   df = pd.DataFrame({"breed": BREEDS, "probability": probs.tolist()})
#   return df.sort_values("probability", ascending=False, ignore_index=True)   # winner first
def classify(image):
    """PIL image -> DataFrame[breed, probability], most likely breed first."""
    raise NotImplementedError("TODO 1: build the DataFrame")


# ===== TODO 2: horizontal bar chart =====
# Plotly Express takes the whole DataFrame plus column names and does the rest.
# `orientation="h"` makes the bars horizontal. px draws the first row at the
# BOTTOM, so sort ascending to get the winning breed on top.
#
#   fig = px.bar(df.sort_values("probability"), x="probability", y="breed",
#                orientation="h", title="Breed probability")
#   fig.update_xaxes(tickformat=".0%", range=[0, 1])   # show 0-100%, full scale
#   return fig
def bar_chart(df):
    """DataFrame -> horizontal bar chart of probability per breed."""
    raise NotImplementedError("TODO 2: build the bar chart")


# ===== TODO 3: radar chart =====
# Same five numbers, drawn as spokes on a wheel. A confident prediction is a
# single spike; a confused one is a lumpy blob. This is `px.line_polar` -- the
# radar cousin of px.bar. `r` is the distance out each spoke, `theta` picks the
# spoke, and line_close=True joins the last point back to the first.
#
#   fig = px.line_polar(df, r="probability", theta="breed", line_close=True,
#                       title="Probability radar", range_r=[0, 1])
#   fig.update_traces(fill="toself")                   # shade the enclosed area
#   return fig
def radar_chart(df):
    """DataFrame -> radar (polar) chart of all five probabilities."""
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
# ----------------------------------------------------------------------------

# --- the web page ---
app = Dash(__name__)

app.layout = html.Div(
    style={
        "maxWidth": "1180px",
        "margin": "40px auto",
        "fontFamily": "system-ui, sans-serif",
    },
    children=[
        html.H1("🐕 Dog Breed Classifier"),
        html.P("Drag in a dog photo. The model knows 5 breeds: " "Beagle, Boxer, Pug, Samoyed, Shiba Inu."),
        dcc.Upload(
            id="upload",
            children=html.Div("Drag & drop a dog photo here, or click to choose"),
            style={
                "height": "120px",
                "lineHeight": "120px",
                "textAlign": "center",
                "border": "2px dashed #aaa",
                "borderRadius": "10px",
                "cursor": "pointer",
            },
            multiple=False,
        ),
        html.Div(id="result", style={"marginTop": "24px"}),
    ],
)


@app.callback(Output("result", "children"), Input("upload", "contents"))
def on_upload(contents):
    """Runs every time you drop a photo: decode it, classify it, draw the charts."""
    if not contents:
        return no_update

    # `contents` is a data URL: "data:image/jpeg;base64,<...>". Decode it into an image.
    _header, b64 = contents.split(",", 1)
    image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

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
                html.Img(src=contents, style={"maxHeight": "160px", "borderRadius": "8px"}),
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
