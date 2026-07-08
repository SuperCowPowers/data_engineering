/*
 * dropzone.js - provided plumbing for Project 7 (you don't need to edit this).
 *
 * Dash automatically loads any file in an `assets/` folder. This turns the plain
 * <div id="dropzone"> into a drop target that accepts a photo four ways:
 *
 *   1. a local file dragged in          -> read as a data: URL (bytes inline)
 *   2. an image dragged off a web page  -> grab its URL (files is empty here!)
 *   3. a click                          -> open a file picker
 *   4. a paste (Ctrl/Cmd-V)             -> an image or a URL from the clipboard
 *
 * The result is pushed into the dcc.Store "dropped-image" via Dash's
 * set_props(), which fires the Python callback. Open the browser console to see
 * the [dropzone] debug lines showing exactly what each drop delivered.
 */
(function () {
  const ZONE_ID = "dropzone";
  const STORE_ID = "dropped-image";

  const log = (...a) => console.log("[dropzone]", ...a);

  // Push a value into the dcc.Store, which triggers the Dash callback.
  function send(value) {
    if (!window.dash_clientside || !window.dash_clientside.set_props) {
      log("dash_clientside.set_props not ready");
      return;
    }
    log("sending", value.slice(0, 80), `(${value.length} chars)`);
    window.dash_clientside.set_props(STORE_ID, { data: value });
  }

  // A real file (local drop, click, or pasted image) -> data: URL.
  function sendFile(file) {
    log("file:", file.name || "(pasted)", file.type);
    const reader = new FileReader();
    reader.onload = () => send(reader.result);
    reader.readAsDataURL(file);
  }

  // Pull an image URL out of whatever a web-page drag left in the DataTransfer.
  function urlFromDataTransfer(dt) {
    let url = dt.getData("text/uri-list") || dt.getData("text/plain");
    if (!url) {
      const html = dt.getData("text/html");
      const m = html && html.match(/<img[^>]+src=["']([^"']+)["']/i);
      if (m) url = m[1];
    }
    return url ? url.trim() : null;
  }

  function handleDrop(dt) {
    log("drop types:", dt && dt.types ? Array.from(dt.types) : dt);
    if (dt.files && dt.files.length && dt.files[0].type.startsWith("image/")) {
      sendFile(dt.files[0]); // case 1: a local file
      return;
    }
    const url = urlFromDataTransfer(dt); // case 2: a web image (a URL, no file)
    if (url) {
      send(url);
      return;
    }
    log("nothing usable in this drop (no image file, no URL)");
  }

  function wire() {
    const zone = document.getElementById(ZONE_ID);
    if (!zone || zone.dataset.wired) return false;
    zone.dataset.wired = "1";
    log("wired to #" + ZONE_ID);

    // Must preventDefault on dragover, or the browser just opens the image.
    ["dragenter", "dragover"].forEach((ev) =>
      zone.addEventListener(ev, (e) => {
        e.preventDefault();
        zone.style.background = "rgba(76,120,168,0.12)";
      })
    );
    ["dragleave", "drop"].forEach((ev) =>
      zone.addEventListener(ev, (e) => {
        e.preventDefault();
        zone.style.background = "";
      })
    );
    zone.addEventListener("drop", (e) => handleDrop(e.dataTransfer));

    // case 3: click to choose a file
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = "image/*";
    picker.style.display = "none";
    document.body.appendChild(picker);
    zone.addEventListener("click", () => picker.click());
    picker.addEventListener("change", () => picker.files[0] && sendFile(picker.files[0]));

    // case 4: paste an image or a URL anywhere on the page
    document.addEventListener("paste", (e) => {
      const items = (e.clipboardData && e.clipboardData.items) || [];
      for (const it of items) {
        if (it.type.startsWith("image/")) {
          sendFile(it.getAsFile());
          return;
        }
      }
      const text = e.clipboardData && e.clipboardData.getData("text");
      if (text) send(text.trim());
    });

    return true;
  }

  // Dash renders the page asynchronously, so poll until #dropzone exists.
  const timer = setInterval(() => {
    if (wire()) clearInterval(timer);
  }, 200);
  window.addEventListener("load", wire);
})();
