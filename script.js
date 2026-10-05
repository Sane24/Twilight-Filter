/* Twilight Filter — small progressive enhancements.
   The page is complete without this file; it only adds:
   1. current-section highlighting in the top nav
   2. click-to-enlarge for figure images
   3. eager loading of lazy images before printing */

(function () {
  "use strict";

  /* 1. Current section in the nav ------------------------------------- */

  var navLinks = document.querySelectorAll(".topbar__nav a[data-nav]");
  var targets = document.querySelectorAll("main [data-nav], footer[data-nav]");

  function updateCurrent() {
    var line = window.innerHeight * 0.3;
    var current = null;

    for (var i = 0; i < targets.length; i++) {
      if (targets[i].getBoundingClientRect().top <= line) {
        current = targets[i].getAttribute("data-nav");
      }
    }

    // at the very bottom, the footer is current even if it's short
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) {
      current = "source";
    }

    for (var j = 0; j < navLinks.length; j++) {
      if (navLinks[j].getAttribute("data-nav") === current) {
        navLinks[j].setAttribute("aria-current", "true");
      } else {
        navLinks[j].removeAttribute("aria-current");
      }
    }
  }

  var ticking = false;
  function onScroll() {
    if (!ticking) {
      ticking = true;
      window.requestAnimationFrame(function () {
        updateCurrent();
        ticking = false;
      });
    }
  }

  if (navLinks.length && targets.length) {
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    updateCurrent();
  }


  /* 2. Click to enlarge ------------------------------------------------ */

  var dialog = document.querySelector(".lightbox");
  var dialogImg = dialog ? dialog.querySelector("img") : null;
  var dialogCap = dialog ? dialog.querySelector(".lightbox__cap") : null;
  var canUseDialog = dialog && typeof dialog.showModal === "function";

  function captionFor(img) {
    var fig = img.closest("figure");
    var cap = fig ? fig.querySelector("figcaption") : null;
    var file = img.getAttribute("src").replace(/^\.\//, "");
    var text = cap ? cap.textContent.replace(/\s+/g, " ").trim() : img.alt;
    return text ? text + "  ·  " + file : file;
  }

  function openImage(img) {
    var src = img.getAttribute("src");
    if (!canUseDialog) {
      window.open(src, "_blank", "noopener");
      return;
    }
    dialogImg.src = src;
    dialogImg.alt = img.alt;
    dialogImg.classList.toggle("pixelated", img.classList.contains("pixelated"));
    dialogCap.textContent = captionFor(img);
    dialog.showModal();
  }

  var zoomable = document.querySelectorAll(".fig img, .hero__plate img, .ladder img");
  for (var k = 0; k < zoomable.length; k++) {
    zoomable[k].setAttribute("data-zoomable", "");
    zoomable[k].addEventListener("click", function (event) {
      openImage(event.currentTarget);
    });
  }

  if (canUseDialog) {
    // click on the dark backdrop closes it
    dialog.addEventListener("click", function (event) {
      if (event.target === dialog) dialog.close();
    });
    dialog.addEventListener("close", function () {
      dialogImg.removeAttribute("src");
    });
  }


  /* 3. Make sure everything is loaded before printing ------------------ */

  function loadAllImages() {
    var lazy = document.querySelectorAll('img[loading="lazy"]');
    for (var n = 0; n < lazy.length; n++) {
      lazy[n].loading = "eager";
    }
  }

  window.addEventListener("beforeprint", loadAllImages);
  window.addEventListener("load", function () {
    // after the first screen is done, quietly fetch the rest (~8 MB total)
    // so Print → Save as PDF never catches an image that hasn't loaded
    window.setTimeout(loadAllImages, 600);
  });
})();
