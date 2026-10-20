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
