/*
 * MIT License
 *
 * Copyright (c) 2026 THALES
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/* MIT License — Copyright (c) 2026 THALES
 * Zoom for Mermaid diagrams. Material renders each diagram in a closed shadow
 * root, so the rendered host element itself is moved into a full-screen viewer
 * (wheel or pinch to zoom, drag to pan, +/−/reset buttons, Esc to close) and
 * put back in place on close. Works with instant navigation (document$).
 */
(function () {
  var viewer = null;

  function open(host) {
    var placeholder = document.createComment("mermaid");
    host.parentNode.insertBefore(placeholder, host);
    var overlay = document.createElement("div");
    overlay.className = "pg-zoom";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Diagram viewer");
    overlay.innerHTML =
      '<div class="pg-zoom__bar">' +
      '<button type="button" data-z="in" aria-label="Zoom in">+</button>' +
      '<button type="button" data-z="out" aria-label="Zoom out">−</button>' +
      '<button type="button" data-z="reset" aria-label="Reset view">1:1</button>' +
      '<button type="button" data-z="close" aria-label="Close">✕</button></div>' +
      '<div class="pg-zoom__stage"><div class="pg-zoom__canvas"></div></div>' +
      '<p class="pg-zoom__hint">Scroll or pinch to zoom · drag to move · Esc to close</p>';
    document.body.appendChild(overlay);
    var stage = overlay.querySelector(".pg-zoom__stage");
    var canvas = overlay.querySelector(".pg-zoom__canvas");
    canvas.appendChild(host);
    var s = 1, x = 0, y = 0, drag = null, pointers = new Map(), pinch = null;

    function fit() {
      var sw = stage.clientWidth, sh = stage.clientHeight;
      var r = host.getBoundingClientRect(), w = r.width / s, h = r.height / s;
      s = Math.min(3, Math.max(0.2, Math.min((sw * 0.92) / w, (sh * 0.92) / h)));
      x = (sw - w * s) / 2; y = (sh - h * s) / 2; apply();
    }
    function apply() { canvas.style.transform = "translate(" + x + "px," + y + "px) scale(" + s + ")"; }
    function zoomAt(f, cx, cy) {
      var ns = Math.min(12, Math.max(0.15, s * f));
      x = cx - ((cx - x) * ns) / s; y = cy - ((cy - y) * ns) / s; s = ns; apply();
    }
    function close() {
      placeholder.parentNode.insertBefore(host, placeholder);
      placeholder.remove();
      overlay.remove();
      document.removeEventListener("keydown", onKey);
      viewer = null;
      host.focus && host.parentNode.querySelector(".pg-mermaid__zoom") && host.parentNode.querySelector(".pg-mermaid__zoom").focus();
    }
    function onKey(e) {
      if (e.key === "Escape") close();
      else if (e.key === "+" || e.key === "=") zoomAt(1.2, stage.clientWidth / 2, stage.clientHeight / 2);
      else if (e.key === "-") zoomAt(1 / 1.2, stage.clientWidth / 2, stage.clientHeight / 2);
    }
    overlay.querySelector(".pg-zoom__bar").addEventListener("click", function (e) {
      var z = e.target.getAttribute("data-z"), cx = stage.clientWidth / 2, cy = stage.clientHeight / 2;
      if (z === "in") zoomAt(1.25, cx, cy); else if (z === "out") zoomAt(0.8, cx, cy); else if (z === "reset") fit(); else if (z === "close") close();
    });
    stage.addEventListener("wheel", function (e) {
      e.preventDefault();
      var r = stage.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
    stage.addEventListener("pointerdown", function (e) {
      stage.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 1) drag = { x: e.clientX - x, y: e.clientY - y };
      if (pointers.size === 2) { var p = Array.from(pointers.values()); pinch = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y); drag = null; }
    });
    stage.addEventListener("pointermove", function (e) {
      if (!pointers.has(e.pointerId)) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2 && pinch) {
        var p = Array.from(pointers.values()), d = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y), r = stage.getBoundingClientRect();
        zoomAt(d / pinch, (p[0].x + p[1].x) / 2 - r.left, (p[0].y + p[1].y) / 2 - r.top); pinch = d;
      } else if (drag) { x = e.clientX - drag.x; y = e.clientY - drag.y; apply(); }
    });
    function up(e) { pointers.delete(e.pointerId); if (pointers.size < 2) pinch = null; if (!pointers.size) drag = null; }
    stage.addEventListener("pointerup", up);
    stage.addEventListener("pointercancel", up);
    overlay.addEventListener("click", function (e) { if (e.target === overlay) close(); });
    document.addEventListener("keydown", onKey);
    requestAnimationFrame(fit);
    overlay.querySelector('[data-z="close"]').focus();
    viewer = overlay;
  }

  function decorate() {
    document.querySelectorAll(".md-typeset div.mermaid").forEach(function (host) {
      if (host.parentNode.classList.contains("pg-mermaid")) return;
      var wrap = document.createElement("div");
      wrap.className = "pg-mermaid";
      host.parentNode.insertBefore(wrap, host);
      wrap.appendChild(host);
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pg-mermaid__zoom";
      btn.setAttribute("aria-label", "Open the diagram in the zoom viewer");
      btn.innerHTML = "⤢ Zoom";
      btn.addEventListener("click", function () { if (!viewer) open(host); });
      wrap.appendChild(btn);
      host.addEventListener("dblclick", function () { if (!viewer) open(host); });
    });
  }

  function init() {
    decorate();
    /* Material renders diagrams asynchronously: watch the content for new hosts. */
    var root = document.querySelector(".md-content");
    if (!root) return;
    var mo = new MutationObserver(function () { decorate(); });
    mo.observe(root, { childList: true, subtree: true });
    setTimeout(function () { mo.disconnect(); }, 15000);
  }
  if (typeof document$ !== "undefined" && document$.subscribe) document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
