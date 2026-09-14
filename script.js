const PAPER_URL = "";

document.querySelectorAll("[data-paper-link]").forEach((link) => {
  if (!PAPER_URL) return;

  link.href = PAPER_URL;
  link.target = "_blank";
  link.rel = "noreferrer";
  link.classList.remove("is-pending");
  link.querySelector("[data-paper-label]").textContent = "Paper";
});

document.querySelectorAll("[data-compare]").forEach((comparison) => {
  const slider = comparison.querySelector('input[type="range"]');
  slider?.addEventListener("input", () => {
    comparison.style.setProperty("--position", `${slider.value}%`);
  });
});

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
  { threshold: 0.2 },
);

document
  .querySelectorAll("video")
  .forEach((video) => videoObserver.observe(video));

const copyButton = document.querySelector("[data-copy-citation]");
copyButton?.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex")?.textContent ?? "";
  await navigator.clipboard.writeText(citation);
  copyButton.textContent = "Copied";
  window.setTimeout(() => {
    copyButton.textContent = "Copy BibTeX";
  }, 1600);
});
