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


def image_urls_to_try(url):
    """Ordered list of image addresses to attempt for a dropped/pasted link.

    A Google Images "Copy link" is a *page* URL like
    `https://www.google.com/imgres?...&imgurl=<full-res>&tbnid=<id>&...`. We try
    Google's own **thumbnail first** (built from `tbnid`): it's small but lives on
    Google's CDN and always loads, whereas the full-res `imgurl` is often on a site
    that blocks hotlinking (that's the "cannot identify image file" error). The
    model resizes everything to 224px anyway, so the thumbnail is plenty. A plain
    image URL (no Google params) is used as-is.
    """
    url = url.strip()
    qs = parse_qs(urlparse(url).query)
    candidates = []
    if "tbnid" in qs:
        candidates.append(f"https://encrypted-tbn0.gstatic.com/images?q=tbn:{qs['tbnid'][0]}")
    if "imgurl" in qs:  # fall back to the full-res original if the thumbnail is missing
        candidates.append(qs["imgurl"][0])
    if not candidates:
        candidates.append(url)
    return candidates


def _fetch(url):
    """GET the bytes at a URL. The User-Agent keeps sites from rejecting us as a bot."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 (local teaching app)
        return resp.read()


def _image_from_data_url(data_url):
    """A "data:image/...;base64,..." string (a local file) -> PIL image."""
    _header, b64 = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _image_from_web(url):
    """Download a web image (trying Google's thumbnail first) -> PIL image."""
    last_error = None
    for candidate in image_urls_to_try(url):
        try:
            return Image.open(io.BytesIO(_fetch(candidate))).convert("RGB")
        except Exception as e:  # not an image / blocked / 404 -> try the next candidate
            last_error = e
    raise last_error


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
