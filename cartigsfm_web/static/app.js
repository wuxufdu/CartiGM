async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatHits(hits) {
  if (!hits || hits.length === 0) return "<span class='muted'>none</span>";
  return hits.map((gene) => `<span class="gene">${escapeHtml(gene)}</span>`).join(" ");
}

function renderScoreRows(rows) {
  if (!rows || rows.length === 0) return "<p class='muted'>No matching axes.</p>";
  return `
    <table>
      <thead><tr><th>Axis</th><th>Layer</th><th>Score</th><th>Marker hits</th><th>Anti hits</th></tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td><strong>${escapeHtml(row.name_en || row.axis_id)}</strong><br><small>${escapeHtml(row.axis_id)}</small></td>
            <td>${escapeHtml(row.layer)}</td>
            <td>${Number(row.combined).toFixed(3)}</td>
            <td>${formatHits(row.marker_hits)}</td>
            <td>${formatHits(row.anti_hits)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderInterpretAxisList(rows) {
  if (!rows || rows.length === 0) return "<p class='muted'>No interpreted axes.</p>";
  return rows.slice(0, 8).map((axis) => {
    const confidence = axis.confidence || {};
    return `
      <article class="axis-mini">
        <h4>${escapeHtml(axis.axis_id)}</h4>
        <p>
          <span class="badge">${escapeHtml(axis.layer)}</span>
          <span class="badge neutral">${escapeHtml(axis.safety_classification || "UNREVIEWED")}</span>
          <span class="badge amber">confidence: ${escapeHtml(confidence.label || "n/a")}</span>
        </p>
        <p><strong>Score:</strong> ${escapeHtml(axis.score)}</p>
        <p><strong>Supporting genes:</strong> ${formatHits(axis.supporting_genes)}</p>
      </article>
    `;
  }).join("");
}

function renderCannotClaim(items) {
  if (!items || items.length === 0) {
    return "<p class='muted'>No blocked claims from the submitted claim list.</p>";
  }
  return `<ul class="cannot-list">${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

async function loadHealth() {
  const data = await api("/api/health");
  document.getElementById("status").innerHTML = `
    <strong>Ready.</strong> cartigsfm ${escapeHtml(data.cartigsfm_version)} |
    dictionary ${escapeHtml(data.dictionary_version)} | ${escapeHtml(data.axis_count)} axes
  `;
}

async function runScore() {
  const output = document.getElementById("scoreOutput");
  output.innerHTML = "<p class='muted'>Scoring...</p>";
  try {
    const data = await api("/api/score", {
      method: "POST",
      body: JSON.stringify({
        genes: document.getElementById("genes").value,
        top: Number(document.getElementById("top").value || 5),
        anti_penalty: Number(document.getElementById("antiPenalty").value || 1),
      }),
    });
    const layerBlocks = Object.entries(data.by_layer || {}).map(([layer, rows]) => `
      <details open>
        <summary>${escapeHtml(layer)} top axes</summary>
        ${renderScoreRows(rows)}
      </details>
    `).join("");
    output.innerHTML = `
      <p><strong>Input genes:</strong> ${formatHits(data.input_genes)}</p>
      <h3>Overall top hits</h3>
      ${renderScoreRows(data.overall)}
      <h3>Layer-specific hits</h3>
      ${layerBlocks}
      <p class="warning">${escapeHtml(data.safety_note)}</p>
    `;
  } catch (error) {
    output.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  }
}

async function runInterpret() {
  const output = document.getElementById("interpretOutput");
  output.innerHTML = "<p class='muted'>Interpreting...</p>";
  try {
    const data = await api("/api/interpret", {
      method: "POST",
      body: JSON.stringify({
        genes: document.getElementById("genes").value,
        top_per_layer: Number(document.getElementById("interpretTopLayer").value || 3),
        overall_top: Number(document.getElementById("interpretOverallTop").value || 8),
        claims: document.getElementById("interpretClaims").value,
      }),
    });
    output.innerHTML = `
      <h3>Safety summary</h3>
      <pre class="json-lite">${escapeHtml(JSON.stringify(data.safety_summary || {}, null, 2))}</pre>
      <h3>Cannot Claim</h3>
      ${renderCannotClaim(data.cannot_claim)}
      <h3>Top interpreted axes</h3>
      <div class="axis-mini-grid">${renderInterpretAxisList(data.top_axes_per_layer)}</div>
      <details>
        <summary>Full Markdown report</summary>
        <pre>${escapeHtml(data.markdown || "")}</pre>
      </details>
    `;
  } catch (error) {
    output.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  }
}

async function runClaimCheck() {
  const output = document.getElementById("claimOutput");
  output.innerHTML = "<p class='muted'>Checking...</p>";
  try {
    const data = await api("/api/claim-check", {
      method: "POST",
      body: JSON.stringify({ claim: document.getElementById("claim").value }),
    });
    output.innerHTML = `
      <p><strong>Matched:</strong> ${data.matched ? "yes" : "no"} (${escapeHtml(data.match_type)})</p>
      <p><strong>Recommendation:</strong> ${escapeHtml(data.recommendation || data.safety_label || "Use conservative wording.")}</p>
      ${data.related_rules ? `<pre>${escapeHtml(JSON.stringify(data.related_rules, null, 2))}</pre>` : ""}
    `;
  } catch (error) {
    output.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  }
}

function renderDictionaryLayer(layerName, layer) {
  const cards = (layer.axes || []).map((axis) => `
    <article class="axis-card">
      <h3>${escapeHtml(axis.name_en || axis.axis_id)}</h3>
      <p class="muted">${escapeHtml(axis.axis_id)}</p>
      <p>${escapeHtml(axis.interpretation || "No bundled interpretation.")}</p>
      <p><strong>Core:</strong> ${formatHits(axis.core_genes)}</p>
      <p><strong>Anti:</strong> ${formatHits(axis.anti_genes)}</p>
      <p class="badge">${escapeHtml(axis.evidence_level || "evidence pending")}</p>
    </article>
  `).join("");
  document.getElementById("dictionaryOutput").innerHTML = cards;
  document.querySelectorAll("#dictionaryTabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.layer === layerName);
  });
}

async function loadDictionary() {
  const data = await api("/api/dictionary");
  const tabs = document.getElementById("dictionaryTabs");
  tabs.innerHTML = Object.entries(data.layers || {}).map(([layerName, layer]) => `
    <button data-layer="${escapeHtml(layerName)}">${escapeHtml(layerName)} (${escapeHtml(layer.count)})</button>
  `).join("");
  tabs.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => renderDictionaryLayer(button.dataset.layer, data.layers[button.dataset.layer]));
  });
  const firstLayer = Object.keys(data.layers || {})[0];
  if (firstLayer) renderDictionaryLayer(firstLayer, data.layers[firstLayer]);
}

function showCommandForm(name) {
  document.querySelectorAll("#commandTabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.command === name);
  });
  document.getElementById("p4Form").classList.toggle("hidden", name !== "p4");
  document.getElementById("inspectForm").classList.toggle("hidden", name !== "inspect");
  document.getElementById("csForm").classList.toggle("hidden", name !== "cs");
}

function optionalNumber(id) {
  const value = document.getElementById(id).value;
  return value ? Number(value) : null;
}

async function buildP4Command() {
  const output = document.getElementById("commandOutput");
  try {
    const data = await api("/api/p4-command", {
      method: "POST",
      body: JSON.stringify({
        h5ad_path: document.getElementById("h5adPath").value,
        outdir: document.getElementById("outdir").value,
        sample_col: document.getElementById("sampleCol").value,
        tissue_col: document.getElementById("tissueCol").value,
        cluster_col: document.getElementById("clusterCol").value,
        celltype_col: document.getElementById("celltypeCol").value,
        layer: document.getElementById("layer").value || null,
        min_cells: optionalNumber("minCells"),
        streaming: document.getElementById("streaming").value,
        chunk_size: optionalNumber("chunkSize"),
        no_celltype_filter: document.getElementById("noCelltypeFilter").checked,
      }),
    });
    output.textContent = `${data.command}\n\n# ${data.note}`;
  } catch (error) {
    output.textContent = error.message;
  }
}

async function buildInspectCommand() {
  const output = document.getElementById("commandOutput");
  try {
    const data = await api("/api/inspect-command", {
      method: "POST",
      body: JSON.stringify({
        h5ad_path: document.getElementById("inspectH5adPath").value,
        output_format: document.getElementById("inspectFormat").value,
      }),
    });
    output.textContent = `${data.command}\n\n# ${data.note}`;
  } catch (error) {
    output.textContent = error.message;
  }
}

async function buildCsPredictCommand() {
  const output = document.getElementById("commandOutput");
  try {
    const data = await api("/api/cs-predict-command", {
      method: "POST",
      body: JSON.stringify({
        h5ad_path: document.getElementById("csH5adPath").value,
        out: document.getElementById("csOut").value,
        mode: document.getElementById("csMode").value,
        layer: document.getElementById("csLayer").value || null,
        device: document.getElementById("csDevice").value || null,
        batch_size: Number(document.getElementById("csBatchSize").value || 4096),
      }),
    });
    output.textContent = `${data.command}\n\n# ${data.note}`;
  } catch (error) {
    output.textContent = error.message;
  }
}

document.getElementById("scoreButton").addEventListener("click", runScore);
document.getElementById("interpretButton").addEventListener("click", runInterpret);
document.getElementById("claimButton").addEventListener("click", runClaimCheck);
document.getElementById("p4Button").addEventListener("click", buildP4Command);
document.getElementById("inspectButton").addEventListener("click", buildInspectCommand);
document.getElementById("csButton").addEventListener("click", buildCsPredictCommand);
document.querySelectorAll("#commandTabs button").forEach((button) => {
  button.addEventListener("click", () => showCommandForm(button.dataset.command));
});

loadHealth().catch((error) => {
  document.getElementById("status").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
});
loadDictionary().catch((error) => {
  document.getElementById("dictionaryOutput").innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
});
