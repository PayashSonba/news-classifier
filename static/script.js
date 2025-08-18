// -------- Helpers --------
function $(id) { return document.getElementById(id); }

function setBadge(verdict) {
  const badge = $("verdictBadge");
  badge.textContent = verdict || "Unknown";
  badge.className = "badge " + (verdict && verdict.toLowerCase());
}

function updateProbabilityMeter(probability) {
  const fill = $("probabilityFill");
  const percent = $("probabilityPercent");
  const clamped = Math.max(0, Math.min(probability * 100, 100));
  fill.style.width = clamped + "%";
  percent.textContent = clamped.toFixed(0) + "%";
}

function updateSplit(realProb, fakeProb) {
  $("realPct").textContent = "Real: " + (realProb * 100).toFixed(1) + "%";
  $("fakePct").textContent = "Fake: " + (fakeProb * 100).toFixed(1) + "%";
}

function renderEvidenceCards(evidenceList) {
  const grid = $("evidenceContainer");
  grid.innerHTML = "";
  if (!evidenceList || evidenceList.length === 0) {
    grid.innerHTML = `<p class="muted">No evidence found.</p>`;
    return;
  }

  evidenceList.forEach(ev => {
    const card = document.createElement("article");
    card.className = "evidence-card";

    const imgUrl = ev.image ? ev.image : "";
    const badge = ev.badge === "Live"
      ? `<span class="pill live">Live</span>`
      : `<span class="pill archive">Archive</span>`;

    card.innerHTML = `
      ${imgUrl ? `<div class="thumb" style="background-image:url('${imgUrl}')"></div>` : `<div class="thumb thumb-placeholder"></div>`}
      <div class="content">
        <div class="meta">
          <span class="source">${ev.source || "Unknown"}</span>
          ${badge}
          ${ev.publishedAt ? `<span class="date">${new Date(ev.publishedAt).toLocaleDateString()}</span>` : ""}
        </div>
        <h4 class="headline"><a href="${ev.url || '#'}" target="_blank" rel="noopener noreferrer">${ev.headline || "Untitled"}</a></h4>
        ${typeof ev.similarity === "number" ? `<div class="sim">Similarity: ${(ev.similarity*100).toFixed(1)}%</div>` : ""}
      </div>
    `;
    grid.appendChild(card);
  });
}

// -------- Main action --------
$("analyzeBtn").addEventListener("click", () => {
  const text = $("newsText").value.trim();
  if (!text) {
    alert("Please paste some news text or a headline.");
    return;
  }

  $("loading").style.display = "flex";
  $("results").style.display = "none";

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  })
  .then(r => r.json())
  .then(data => {
    $("loading").style.display = "none";

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    // Verdict + location
    setBadge(data.verdict);
    $("locationHint").textContent = data.location ? `Location: ${data.location}` : "";

    // Meter + split
    const prob = (typeof data.probability === "number") ? data.probability : (data.real_prob || 0.5);
    updateProbabilityMeter(prob);
    updateSplit(data.real_prob || 0, data.fake_prob || 0);

    // Evidence
    renderEvidenceCards(data.evidence || []);

    $("results").style.display = "block";
    window.scrollTo({ top: $("results").offsetTop - 20, behavior: "smooth" });
  })
  .catch(err => {
    $("loading").style.display = "none";
    alert("Request failed: " + err);
  });
});
