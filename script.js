const PAPER_URL = "";

document.querySelectorAll("[data-paper-link]").forEach((link) => {
  if (!PAPER_URL) return;

  link.href = PAPER_URL;
  link.target = "_blank";
  link.rel = "noreferrer";
  link.classList.remove("is-pending");
  link.querySelector("[data-paper-label]").textContent = "Paper";
});

const copyButton = document.querySelector("[data-copy-citation]");
copyButton?.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex")?.textContent ?? "";
  await navigator.clipboard.writeText(citation);
  copyButton.textContent = "Copied";
  window.setTimeout(() => {
    copyButton.textContent = "Copy BibTeX";
  }, 1600);
});
