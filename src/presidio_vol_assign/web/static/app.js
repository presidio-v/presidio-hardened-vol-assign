/* Demo GUI behaviour.
 *
 * The server sends the whole trade-off set in one response — objective values
 * plus a compact allocation vector per solution — so moving the trade-off
 * slider re-renders instantly without another round trip.
 */

"use strict";

const AREA_KM = 70;
const SITE_COLOR_COUNT = 8;

const state = {
  config: null,
  scenario: null,
  knobs: {},
  result: null,
  solutions: [],
  index: 0,
  activePreset: null,
};

/* ---------------------------------------------------------------- helpers */

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "text") node.textContent = v;
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const child of children || []) {
    if (child) node.appendChild(child);
  }
  return node;
}

function svgEl(tag, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined) continue;
    node.setAttribute(k, String(v));
  }
  return node;
}

function siteColor(i) {
  return `var(--site-${i % SITE_COLOR_COUNT})`;
}

function fmt(value, digits) {
  return Number(value).toFixed(digits === undefined ? 3 : digits);
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/* -------------------------------------------------------------- bootstrap */

async function init() {
  const status = document.getElementById("run-status");
  try {
    const res = await fetch("/api/scenarios");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.config = await res.json();
  } catch (err) {
    showError(`Could not load the demo configuration: ${err.message}`);
    return;
  }

  document.getElementById("version-note").textContent =
    `presidio-hardened-vol-assign ${state.config.version} · ` +
    `runs are capped at ${state.config.limits.maxUnits} units and ` +
    `${state.config.limits.maxGenerations} generations.`;

  renderScenarioCards();
  selectScenario(state.config.scenarios[0].id);

  document.getElementById("run-btn").addEventListener("click", runScenario);
  document.getElementById("tradeoff").addEventListener("input", (e) => {
    state.activePreset = null;
    selectSolution(Number(e.target.value));
  });
  document.getElementById("download-btn").addEventListener("click", downloadCsv);

  bindOutput("generations", "generations-out");
  bindOutput("seed", "seed-out");
  status.textContent = "";
}

function bindOutput(inputId, outputId) {
  const input = document.getElementById(inputId);
  const output = document.getElementById(outputId);
  const sync = () => { output.textContent = input.value; };
  input.addEventListener("input", sync);
  sync();
}

/* ------------------------------------------------------------- scenario UI */

function renderScenarioCards() {
  const holder = document.getElementById("scenario-cards");
  clear(holder);
  for (const scenario of state.config.scenarios) {
    const card = el("button", {
      type: "button",
      class: "card",
      "aria-pressed": "false",
      "data-scenario": scenario.id,
    }, [
      el("div", { class: "card-title", text: scenario.title }),
      el("div", { class: "card-sub", text: scenario.subtitle }),
    ]);
    card.addEventListener("click", () => selectScenario(scenario.id));
    holder.appendChild(card);
  }
}

function selectScenario(id) {
  state.scenario = state.config.scenarios.find((s) => s.id === id);
  for (const card of document.querySelectorAll(".card")) {
    card.setAttribute("aria-pressed", String(card.dataset.scenario === id));
  }
  document.getElementById("scenario-detail").textContent = state.scenario.description;
  renderKnobs();
  document.getElementById("step-results").hidden = true;
  hideError();
}

function renderKnobs() {
  const holder = document.getElementById("knobs");
  clear(holder);
  state.knobs = {};

  for (const knob of state.scenario.knobs) {
    state.knobs[knob.key] = knob.default;
    const output = el("output", { text: String(knob.default) });
    const input = el("input", {
      type: "range",
      min: knob.min,
      max: knob.max,
      step: knob.step,
      value: knob.default,
    });
    input.addEventListener("input", () => {
      const value = Number(input.value);
      state.knobs[knob.key] = value;
      output.textContent = String(value);
    });
    holder.appendChild(el("label", { class: "knob" }, [
      el("span", { class: "knob-label" }, [
        el("span", { text: knob.label }),
        output,
      ]),
      input,
      el("span", { class: "knob-help", text: knob.help }),
    ]));
  }
}

/* -------------------------------------------------------------------- run */

async function runScenario() {
  const button = document.getElementById("run-btn");
  const status = document.getElementById("run-status");
  hideError();
  button.disabled = true;
  status.textContent = "Generating the situation and searching for trade-offs…";

  const body = {
    scenario: state.scenario.id,
    knobs: state.knobs,
    solver: document.getElementById("solver").value,
    seed: Number(document.getElementById("seed").value),
    generations: Number(document.getElementById("generations").value),
  };

  const started = performance.now();
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
    state.result = payload;
    const elapsed = (performance.now() - started) / 1000;
    status.textContent = `Done in ${elapsed.toFixed(1)}s.`;
    renderResults();
  } catch (err) {
    status.textContent = "";
    showError(err.message);
  } finally {
    button.disabled = false;
  }
}

function showError(message) {
  const node = document.getElementById("error");
  node.textContent = message;
  node.hidden = false;
}

function hideError() {
  document.getElementById("error").hidden = true;
}

/* ---------------------------------------------------------------- results */

function renderResults() {
  const result = state.result;
  // The map and trade-off slider always follow the first solver; when two were
  // run, the comparison lives in the run-quality table below.
  state.solutions = result.results[0].solutions
    .slice()
    .sort((a, b) => a.objectives[0] - b.objectives[0]);

  document.getElementById("step-results").hidden = false;
  document.getElementById("cli-hint").textContent = buildCliHint();

  const slider = document.getElementById("tradeoff");
  slider.max = String(Math.max(0, state.solutions.length - 1));
  slider.value = "0";

  renderPresets();
  renderMetrics();
  selectPreset("compromise");
  document.getElementById("step-results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function buildCliHint() {
  const r = state.result;
  return `${r.cliHint} \\\n  --solver ${r.solver} --seed ${r.seed} --generations ${r.generations}`;
}

/* Presets jump to the extreme of each objective, plus a balanced compromise. */
function renderPresets() {
  const holder = document.getElementById("preset-row");
  clear(holder);

  const presets = state.result.objectives.map((obj, i) => ({
    id: `obj-${i}`,
    label: `Best for: ${obj.label.toLowerCase()}`,
    pick: () => argmin(state.solutions, (s) => s.objectives[i]),
  }));
  presets.push({
    id: "compromise",
    label: "Best all-round compromise",
    pick: () => argmin(state.solutions, compromiseScore()),
  });

  for (const preset of presets) {
    const button = el("button", {
      type: "button",
      text: preset.label,
      "data-preset": preset.id,
      "aria-pressed": "false",
    });
    button.addEventListener("click", () => selectPreset(preset.id));
    holder.appendChild(button);
  }
  state.presets = presets;
}

function selectPreset(id) {
  const preset = state.presets.find((p) => p.id === id) || state.presets[state.presets.length - 1];
  state.activePreset = preset.id;
  for (const button of document.querySelectorAll("#preset-row button")) {
    button.setAttribute("aria-pressed", String(button.dataset.preset === preset.id));
  }
  const index = preset.pick();
  document.getElementById("tradeoff").value = String(index);
  selectSolution(index);
}

function argmin(items, score) {
  let best = 0;
  let bestValue = Infinity;
  items.forEach((item, i) => {
    const value = score(item);
    if (value < bestValue) { bestValue = value; best = i; }
  });
  return best;
}

/* Distance to the ideal point, with each objective normalised over the set so
   no objective dominates the comparison purely through its scale. */
function compromiseScore() {
  const ranges = objectiveRanges();
  return (solution) => {
    let total = 0;
    solution.objectives.forEach((value, i) => {
      total += Math.pow(normalise(value, ranges[i]), 2);
    });
    return total;
  };
}

function objectiveRanges() {
  return state.result.objectives.map((_, i) => {
    const values = state.solutions.map((s) => s.objectives[i]);
    return { min: Math.min(...values), max: Math.max(...values) };
  });
}

function normalise(value, range) {
  const span = range.max - range.min;
  return span > 1e-12 ? (value - range.min) / span : 0;
}

function selectSolution(index) {
  state.index = index;
  const solution = state.solutions[index];
  if (!solution) return;

  if (state.activePreset === null) {
    for (const button of document.querySelectorAll("#preset-row button")) {
      button.setAttribute("aria-pressed", "false");
    }
  }
  document.getElementById("tradeoff-out").textContent =
    `option ${index + 1} of ${state.solutions.length}`;

  renderObjectives(solution);
  renderMap(solution);
  renderFrontChart(index);
  renderLoads(solution);
}

/* ------------------------------------------------------------- objectives */

function renderObjectives(solution) {
  const holder = document.getElementById("objectives");
  clear(holder);
  const ranges = objectiveRanges();

  state.result.objectives.forEach((obj, i) => {
    const value = solution.objectives[i];
    const share = normalise(value, ranges[i]);
    const fill = el("div", { class: "obj-fill" });
    // Set via CSSOM rather than a style attribute: inline styles are blocked.
    fill.style.width = `${(share * 100).toFixed(1)}%`;
    fill.style.background = share < 0.34 ? "var(--ok)" : share > 0.66 ? "var(--danger)" : "var(--accent)";

    const position = share <= 0.001 ? "best available" : share >= 0.999 ? "worst in this set" : `${Math.round(share * 100)}% of the way to the worst option`;

    holder.appendChild(el("div", { class: "obj" }, [
      el("div", { class: "obj-head" }, [
        el("span", { text: obj.label }),
        el("span", { text: fmt(value, 4) }),
      ]),
      el("div", { class: "obj-bar" }, [fill]),
      el("div", { class: "obj-help", text: `${obj.help} — ${position}.` }),
    ]));
  });
}

/* -------------------------------------------------------------------- map */

function renderMap(solution) {
  const holder = document.getElementById("map");
  clear(holder);

  const size = 400;
  const pad = 14;
  const scale = (v) => pad + (v / AREA_KM) * (size - 2 * pad);
  const svg = svgEl("svg", {
    viewBox: `0 0 ${size} ${size}`,
    role: "img",
    "aria-label": "Map of the affected area showing where each unit is sent",
  });

  svg.appendChild(svgEl("rect", {
    x: 0, y: 0, width: size, height: size,
    fill: "var(--surface-2)", rx: 8,
  }));

  const sites = state.result.sites;
  const units = state.result.units;

  // Connection lines first so the markers sit on top of them.
  units.forEach((unit, i) => {
    const siteIdx = solution.alloc[i];
    if (siteIdx < 0) return;
    const site = sites[siteIdx];
    svg.appendChild(svgEl("line", {
      x1: scale(unit.x), y1: scale(unit.y),
      x2: scale(site.x), y2: scale(site.y),
      stroke: siteColor(siteIdx),
      "stroke-width": 0.6,
      "stroke-opacity": 0.35,
    }));
  });

  units.forEach((unit, i) => {
    const siteIdx = solution.alloc[i];
    const assigned = siteIdx >= 0;
    const radius = 2.2 + Math.min(unit.weight || 1, 5) * 0.5;
    const dot = svgEl("circle", {
      cx: scale(unit.x), cy: scale(unit.y), r: radius,
      fill: assigned ? siteColor(siteIdx) : "none",
      stroke: assigned ? "none" : "var(--text-dim)",
      "stroke-width": assigned ? 0 : 1,
      "fill-opacity": 0.85,
    });
    dot.appendChild(svgEl("title", {})).textContent =
      assigned ? `${unit.label} → ${sites[siteIdx].id}` : `${unit.label} — not assigned`;
    svg.appendChild(dot);
  });

  sites.forEach((site, j) => {
    const marker = svgEl("rect", {
      x: scale(site.x) - 7, y: scale(site.y) - 7,
      width: 14, height: 14, rx: 3,
      fill: siteColor(j),
      stroke: "var(--surface)",
      "stroke-width": 2,
    });
    marker.appendChild(svgEl("title", {})).textContent = site.label;
    svg.appendChild(marker);
    const label = svgEl("text", {
      x: scale(site.x), y: scale(site.y) - 11,
      "text-anchor": "middle",
      "font-size": 10,
      "font-weight": 600,
      fill: "var(--text)",
    });
    label.textContent = site.id;
    svg.appendChild(label);
  });

  holder.appendChild(svg);
  renderMapLegend(solution);
}

function renderMapLegend(solution) {
  const holder = document.getElementById("map-legend");
  clear(holder);
  const counts = siteLoads(solution);

  state.result.sites.forEach((site, j) => {
    const swatch = el("span", { class: "legend-swatch" });
    swatch.style.background = siteColor(j);
    holder.appendChild(el("span", { class: "legend-item" }, [
      swatch,
      el("span", { text: `${site.id} — ${counts[j]} of ${site.capacity}` }),
    ]));
  });

  const unassigned = solution.alloc.filter((v) => v < 0).length;
  if (unassigned > 0) {
    const swatch = el("span", { class: "legend-swatch" });
    swatch.style.border = "1px solid var(--text-dim)";
    holder.appendChild(el("span", { class: "legend-item" }, [
      swatch,
      el("span", { text: `${unassigned} not deployed` }),
    ]));
  }
}

function siteLoads(solution) {
  const counts = state.result.sites.map(() => 0);
  solution.alloc.forEach((siteIdx, i) => {
    if (siteIdx >= 0) counts[siteIdx] += state.result.units[i].weight || 1;
  });
  return counts;
}

/* ------------------------------------------------------------ front chart */

function renderFrontChart(activeIndex) {
  const holder = document.getElementById("front-chart");
  clear(holder);

  const objectives = state.result.objectives;
  const width = 400;
  const height = 260;
  const pad = { top: 14, right: 14, bottom: 40, left: 52 };

  const xs = state.solutions.map((s) => s.objectives[0]);
  const ys = state.solutions.map((s) => s.objectives[1]);
  const xr = { min: Math.min(...xs), max: Math.max(...xs) };
  const yr = { min: Math.min(...ys), max: Math.max(...ys) };
  const sx = (v) => pad.left + normalise(v, xr) * (width - pad.left - pad.right);
  const sy = (v) => height - pad.bottom - normalise(v, yr) * (height - pad.top - pad.bottom);

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Scatter plot of the trade-off set",
  });

  svg.appendChild(svgEl("line", {
    x1: pad.left, y1: height - pad.bottom, x2: width - pad.right, y2: height - pad.bottom,
    stroke: "var(--border)", "stroke-width": 1,
  }));
  svg.appendChild(svgEl("line", {
    x1: pad.left, y1: pad.top, x2: pad.left, y2: height - pad.bottom,
    stroke: "var(--border)", "stroke-width": 1,
  }));

  const xLabel = svgEl("text", {
    x: pad.left + (width - pad.left - pad.right) / 2, y: height - 8,
    "text-anchor": "middle", "font-size": 11, fill: "var(--text-dim)",
  });
  xLabel.textContent = `${objectives[0].label} →`;
  svg.appendChild(xLabel);

  const yLabel = svgEl("text", {
    x: 0, y: 0, "text-anchor": "middle", "font-size": 11, fill: "var(--text-dim)",
    transform: `translate(14, ${pad.top + (height - pad.top - pad.bottom) / 2}) rotate(-90)`,
  });
  yLabel.textContent = `${objectives[1].label} →`;
  svg.appendChild(yLabel);

  // With three objectives the third is shown as marker size, so the plot still
  // says something about the objective that has no axis.
  const zr = objectives.length > 2
    ? { min: Math.min(...state.solutions.map((s) => s.objectives[2])),
        max: Math.max(...state.solutions.map((s) => s.objectives[2])) }
    : null;

  state.solutions.forEach((solution, i) => {
    const active = i === activeIndex;
    const r = zr ? 2.5 + normalise(solution.objectives[2], zr) * 4 : 3.4;
    const dot = svgEl("circle", {
      cx: sx(solution.objectives[0]),
      cy: sy(solution.objectives[1]),
      r: active ? Math.max(r, 5) : r,
      fill: active ? "var(--accent)" : "var(--text-dim)",
      "fill-opacity": active ? 1 : 0.45,
      stroke: active ? "var(--surface)" : "none",
      "stroke-width": active ? 2 : 0,
    });
    dot.appendChild(svgEl("title", {})).textContent =
      objectives.map((o, k) => `${o.label}: ${fmt(solution.objectives[k], 4)}`).join("\n");
    svg.appendChild(dot);
  });

  holder.appendChild(svg);

  document.getElementById("front-caption").textContent = zr
    ? `Each dot is one allocation you could choose. Larger dots mean more ${objectives[2].label.toLowerCase()}. The highlighted dot is the one shown on the map.`
    : "Each dot is one allocation you could choose. The highlighted dot is the one shown on the map.";
}

/* ------------------------------------------------------------------ loads */

function renderLoads(solution) {
  const table = document.getElementById("loads-table");
  clear(table);
  const counts = siteLoads(solution);
  const isHumanitarian = state.result.model === "humanitarian";
  const unitWord = isHumanitarian ? "People allocated" : "Roles filled";
  const capWord = isHumanitarian ? "Capacity" : "Roles open";

  const head = el("tr", {}, [
    el("th", { text: state.result.model === "humanitarian" ? "Centre" : "Department" }),
    el("th", { text: unitWord }),
    el("th", { text: capWord }),
    el("th", { text: "Utilisation" }),
  ]);
  table.appendChild(el("thead", {}, [head]));

  const body = el("tbody", {});
  state.result.sites.forEach((site, j) => {
    const capacity = site.capacity || 0;
    const ratio = capacity > 0 ? counts[j] / capacity : 0;
    const over = counts[j] > capacity;
    body.appendChild(el("tr", { class: over ? "over" : null }, [
      el("td", { text: site.id }),
      el("td", { text: String(counts[j]) }),
      el("td", { text: String(capacity) }),
      el("td", {
        class: !over && ratio > 0 ? "good" : null,
        text: capacity > 0 ? `${Math.round(ratio * 100)}%${over ? " — over capacity" : ""}` : "—",
      }),
    ]));
  });
  table.appendChild(body);
}

function renderMetrics() {
  const table = document.getElementById("metrics-table");
  clear(table);
  table.appendChild(el("thead", {}, [
    el("tr", {}, [
      el("th", { text: "Algorithm" }),
      el("th", { text: "Options found" }),
      el("th", { text: "Hypervolume" }),
      el("th", { text: "Spacing" }),
      el("th", { text: "Solver time" }),
    ]),
  ]));
  const body = el("tbody", {});
  for (const result of state.result.results) {
    body.appendChild(el("tr", {}, [
      el("td", { text: result.solver.toUpperCase() }),
      el("td", { text: String(result.metrics.nns) }),
      el("td", { text: fmt(result.metrics.hv, 4) }),
      el("td", { text: fmt(result.metrics.sm, 4) }),
      el("td", { text: `${fmt(result.metrics.cpuTimeSec, 2)}s` }),
    ]));
  }
  table.appendChild(body);

  document.getElementById("metrics-note").textContent =
    "Hypervolume measures how much of the trade-off space the set covers — higher is better. " +
    "Spacing measures how evenly the options are spread — lower is better. " +
    (state.result.results.length > 1
      ? "The map above shows the first algorithm's result."
      : "");
}

/* --------------------------------------------------------------- download */

function downloadCsv() {
  const solution = state.solutions[state.index];
  if (!solution) return;

  const isHumanitarian = state.result.model === "humanitarian";
  const header = isHumanitarian ? ["person_id", "center_id"] : ["volunteer_id", "ed_id"];
  const rows = [header.join(",")];

  solution.alloc.forEach((siteIdx, i) => {
    if (siteIdx < 0) return;
    rows.push(`${state.result.units[i].id},${state.result.sites[siteIdx].id}`);
  });

  const meta = [
    "",
    `# scenario,${state.result.scenario}`,
    `# solver,${state.result.results[0].solver}`,
    `# seed,${state.result.seed}`,
    `# generations,${state.result.generations}`,
    ...state.result.objectives.map(
      (obj, k) => `# ${obj.key} (${obj.label}),${solution.objectives[k]}`
    ),
  ];

  const blob = new Blob([rows.concat(meta).join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = el("a", {
    href: url,
    download: `allocation_${state.result.scenario}_seed${state.result.seed}.csv`,
  });
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

document.addEventListener("DOMContentLoaded", init);
