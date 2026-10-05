/* Twilight Filter — small progressive enhancements.
   The page is complete without this file; it only adds:
   1. scroll effects: current section in the nav, the "nightfall" bar,
      and dusk turning to night in the header
   2. ambient loops that pause off screen, and the bat-signal switching on
   3. the "step back" slider (2.2) and the hard-mask wipe (2.4)
   4. click-to-enlarge for figure images
   5. eager loading of lazy images before printing */

(function () {
  "use strict";

  var root = document.documentElement;


  /* 1. Scroll effects -------------------------------------------------- */

  var navLinks = document.querySelectorAll(".topbar__nav a[data-nav]");
  var targets = document.querySelectorAll("main [data-nav], footer[data-nav]");
  var hero = document.querySelector(".hero");
  var lastHeroP = -1;

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

  function clamp01(x) {
    return x < 0 ? 0 : x > 1 ? 1 : x;
  }

  function updateSky() {
    // whole page: how far into the night we are
    var max = root.scrollHeight - window.innerHeight;
    root.style.setProperty("--page-p", max > 0 ? clamp01(window.scrollY / max).toFixed(4) : "0");

    // header: the sun is down by the time the horizon nears the top of the screen
    if (hero) {
      var p = clamp01(window.scrollY / Math.max(1, hero.offsetHeight - 220));
      p = Math.round(p * 500) / 500;
      if (p !== lastHeroP) {
        hero.style.setProperty("--p", p);
        lastHeroP = p;
      }
    }
  }

  var ticking = false;
  function onScroll() {
    if (!ticking) {
      ticking = true;
      window.requestAnimationFrame(function () {
        updateCurrent();
        updateSky();
        ticking = false;
      });
    }
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
  updateCurrent();
  updateSky();


  /* 2. Ambient loops and the bat-signal -------------------------------- */

  var ambient = document.querySelectorAll("[data-ambient]");
  var signal = document.querySelector(".signal");

  if ("IntersectionObserver" in window) {
    var ambientObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        entry.target.classList.toggle("is-active", entry.isIntersecting);
      });
    });
    for (var a = 0; a < ambient.length; a++) ambientObserver.observe(ambient[a]);

    if (signal) {
      var signalObserver = new IntersectionObserver(function (entries) {
        if (entries[0].isIntersecting) {
          signal.classList.add("is-lit");
          signalObserver.disconnect();
        }
      }, { threshold: 0.45 });
      signalObserver.observe(signal.parentElement);
    }
  } else {
    for (var b = 0; b < ambient.length; b++) ambient[b].classList.add("is-active");
    if (signal) signal.classList.add("is-lit");
  }


  /* 3. Interactive figures --------------------------------------------- */

  // 2.2: shrink the hybrid as if stepping back from it
  var distance = document.querySelector(".distance");
  if (distance) {
    var range = distance.querySelector("input");
    var out = distance.querySelector("output");
    var hybrid = distance.parentElement.querySelector(".distance-stage img");

    var setDistance = function () {
      var size = 1 - range.value / 100;
      var reads = size > 0.5 ? "Edward" : size > 0.25 ? "a bit of both" : "Batman";
      hybrid.style.width = (size * 100).toFixed(1) + "%";
      out.textContent = Math.round(size * 100) + "% size · " + reads;
    };

    distance.hidden = false;
    range.addEventListener("input", setDistance);
    setDistance();
  }

  // 2.4: wipe between the hard-mask composite and the multiresolution blend
  var wipes = document.querySelectorAll(".wipe");
  for (var w = 0; w < wipes.length; w++) {
    (function (wipe) {
      var input = wipe.querySelector(".wipe__range");
      if (!input) return;
      var setWipe = function () {
        wipe.style.setProperty("--w", input.value + "%");
      };
      wipe.classList.add("is-live");
      input.addEventListener("input", setWipe);
      setWipe();
    })(wipes[w]);
  }


  /* 4. Click to enlarge ------------------------------------------------ */

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

  var zoomable = document.querySelectorAll(".fig img:not(.wipe__top), .hero__plate img, .ladder img");
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


  /* 5. Make sure everything is loaded before printing ------------------ */

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
