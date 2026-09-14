const PAPER_URL = "";

const paperLinks = document.querySelectorAll("[data-paper-link]");
paperLinks.forEach((link) => {
  if (PAPER_URL) {
    link.href = PAPER_URL;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.querySelector("[data-paper-label]").textContent = "Read the paper";
    link.classList.remove("is-pending");
  }
});

document.querySelectorAll("[data-compare]").forEach((comparison) => {
  const slider = comparison.querySelector('input[type="range"]');
  slider?.addEventListener("input", () => {
    comparison.style.setProperty("--position", `${slider.value}%`);
  });
});

const tokenField = document.querySelector("[data-token-field]");
if (tokenField) {
  const kept = new Set([27, 28, 39, 40, 41, 51, 52, 53, 64, 65, 76, 77]);
  for (let index = 0; index < 144; index += 1) {
    const token = document.createElement("span");
    token.className = kept.has(index) ? "kept" : "removed";
    tokenField.appendChild(token);
  }

  window.setInterval(() => {
    const removed = [...tokenField.querySelectorAll(".removed")];
    const token = removed[Math.floor(Math.random() * removed.length)];
    token?.classList.toggle("removed");
    window.setTimeout(() => token?.classList.add("removed"), 500);
  }, 750);
}

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 },
);

document.querySelectorAll(".reveal").forEach((element) => {
  revealObserver.observe(element);
});

const metricObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const element = entry.target;
      const target = Number(element.dataset.count);
      const decimals = Number(element.dataset.decimals || 0);
      const prefix = element.dataset.prefix || "";
      const suffix = element.dataset.suffix || "";
      const start = performance.now();
      const duration = 900;

      const tick = (now) => {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - (1 - progress) ** 3;
        const value = (target * eased).toFixed(decimals);
        element.textContent = `${prefix}${value}${suffix}`;
        if (progress < 1) requestAnimationFrame(tick);
      };

      requestAnimationFrame(tick);
      metricObserver.unobserve(element);
    });
  },
  { threshold: 0.6 },
);

document.querySelectorAll("[data-count]").forEach((element) => {
  metricObserver.observe(element);
});

const sections = [...document.querySelectorAll("main section[id]")];
const navLinks = [...document.querySelectorAll(".site-header nav a")];
const navObserver = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;

    navLinks.forEach((link) => {
      const selected = link.getAttribute("href") === `#${visible.target.id}`;
      if (selected) {
        link.setAttribute("aria-current", "true");
      } else {
        link.removeAttribute("aria-current");
      }
    });
  },
  { rootMargin: "-25% 0px -60% 0px", threshold: [0.05, 0.2, 0.5] },
);
sections.forEach((section) => navObserver.observe(section));

const videoObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      const video = entry.target;
      if (entry.isIntersecting) {
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  },
  { threshold: 0.15 },
);
document
  .querySelectorAll("video")
  .forEach((video) => videoObserver.observe(video));

const copyButton = document.querySelector("[data-copy-citation]");
copyButton?.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex")?.textContent ?? "";
  await navigator.clipboard.writeText(citation);
  const original = copyButton.textContent;
  copyButton.textContent = "Copied";
  window.setTimeout(() => {
    copyButton.textContent = original;
  }, 1600);
});
