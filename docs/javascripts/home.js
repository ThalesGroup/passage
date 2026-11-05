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
 * Home page: count the release figures up when they enter the viewport.
 * Works with Material's instant navigation (document$) and respects reduced motion.
 */
(function () {
  function animate(el) {
    var target = parseFloat(el.dataset.count);
    if (!isFinite(target)) return;
    var decimals = parseInt(el.dataset.decimals || "0", 10), suffix = el.dataset.suffix || "";
    var final = el.textContent, t0 = null, dur = 1400;
    function step(t) {
      if (t0 === null) t0 = t;
      var k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = (target * e).toFixed(decimals) + suffix;
      if (k < 1) requestAnimationFrame(step); else el.textContent = final;
    }
    requestAnimationFrame(step);
  }
  function init() {
    var items = document.querySelectorAll(".pg-stats dd[data-count]");
    if (!items.length) return;
    if (window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (!("IntersectionObserver" in window)) return;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { animate(en.target); io.unobserve(en.target); } });
    }, { threshold: 0.6 });
    items.forEach(function (el) { io.observe(el); });
  }
  if (typeof document$ !== "undefined" && document$.subscribe) document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
