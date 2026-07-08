"""
drag_drop_utils.py - the provided plumbing that makes the drop zone accept a photo
from anywhere: a local file, an image dragged off a web page (Google Images), a
click-to-choose file, or a paste. Both app.py and app_completed.py import this so
the drag/drop logic lives in exactly one place.

How the pieces fit together:

  assets/dropzone.js  (runs in the browser)  --> writes the dropped photo into a
      dcc.Store as a string, then Dash fires the callback.
  drag_drop_utils.py  (runs in Python)       --> turns that string back into an image.

The string is one of two things:
  * a "data:" URL  -> the browser already read a local file for us (bytes inline)
  * an http(s) URL -> an image on the web that WE download here
"""

import base64
import io
import urllib.request
from urllib.parse import parse_qs, urlparse

from PIL import Image
from dash import dcc, html

# The dcc.Store the browser writes to and the callback reads from.
DROPPED_ID = "dropped-image"


def dropzone():
    """The single drop-zone component (a styled box + the hidden Store)."""
    return html.Div(
        children=[
            html.Div(
                "🐕  Drag a dog photo here — a file from your computer, or an image "
                "straight off a web page (Google Images). You can also click to pick "
                "a file, or paste an image / image URL.",
                id="dropzone",
                style={
                    "minHeight": "120px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "textAlign": "center",
                    "padding": "16px",
                    "border": "2px dashed #aaa",
                    "borderRadius": "10px",
                    "cursor": "pointer",
                    "color": "#555",
                },
            ),
            dcc.Store(id=DROPPED_ID),
        ]
    )


def resolve_image_url(url):
    """Unwrap a Google Images "Copy link" URL to the real image address.

    Google's copy-link gives a *page* URL like
    `https://www.google.com/imgres?...&imgurl=<the real .jpg>&...` — the actual
    picture is hiding in the `imgurl` query parameter. A plain image URL has no
    `imgurl`, so it passes straight through.
    """
    qs = parse_qs(urlparse(url.strip()).query)
    return qs.get("imgurl", [url.strip()])[0]


def _image_from_data_url(data_url):
    """A "data:image/...;base64,..." string (a local file) -> PIL image."""
    _header, b64 = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _image_from_web(url):
    """Download an image URL and return a PIL image.

    The User-Agent header keeps sites from rejecting the request as a bot.
    """
    req = urllib.request.Request(resolve_image_url(url), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 (local teaching app)
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def to_data_url(image):
    """PIL image -> a data URL, so the thumbnail always shows exactly what we classified."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_image(value):
    """The dropped string -> (PIL image, thumbnail data URL).

    A "data:" value is a local file the browser already read; anything else is a
    web URL we download here.
    """
    value = value.strip()
    if value.startswith("data:"):
        image = _image_from_data_url(value)
        return image, value  # the data URL is itself a perfect thumbnail
    image = _image_from_web(value)
    return image, to_data_url(image)  # rebuild a thumbnail from the fetched bytes
