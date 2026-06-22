/**
 * Initializes filtering and sorting for the Available Environments card grid.
 *
 * Styles are inlined as a sibling <style> block. Fern's MDX-component bundler
 * does not resolve `import "./*.css"` side-effects, and the `nvidia` global
 * theme owns the docs.yml `css:` field, so per-component styles live here.
 */

import { useEffect } from "react";

const envFilterCss = `
.env-filter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1.5rem;
  align-items: center;
  margin: 1rem 0 0.75rem;
}
.env-filter-group {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  align-items: center;
}
.env-filter-group-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  opacity: 0.6;
  margin-right: 0.25rem;
  white-space: nowrap;
}
.env-filter-btn {
  padding: 0.2rem 0.65rem;
  border-radius: 1rem;
  border: 1px solid var(--color-foreground-border, #ccc);
  background: transparent;
  cursor: pointer;
  font-size: 0.8rem;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
  color: inherit;
}
.env-filter-btn:hover {
  border-color: var(--color-brand-primary, #76b900);
}
.env-filter-btn.active {
  background: var(--color-brand-primary, #76b900);
  border-color: var(--color-brand-primary, #76b900);
  color: #fff;
}
.env-sort-select {
  padding: 0.2rem 0.5rem;
  border-radius: 0.4rem;
  border: 1px solid var(--color-foreground-border, #ccc);
  background: transparent;
  font-size: 0.8rem;
  cursor: pointer;
  color: inherit;
}
.env-count-label {
  font-size: 0.82rem;
  opacity: 0.65;
  margin: 0.25rem 0 1rem;
}
.env-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 1rem;
  margin-top: 0.5rem;
}
.env-card {
  border: 1px solid var(--color-foreground-border, #ddd);
  border-radius: 0.5rem;
  padding: 0.85rem 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  background: var(--color-background-secondary, transparent);
  transition: box-shadow 0.15s;
}
.env-card:hover {
  box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
.env-card[style*="display: none"] {
  display: none !important;
}
.env-card-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.5rem;
}
.env-card-name {
  font-weight: 600;
  font-size: 0.95rem;
  line-height: 1.3;
}
.env-badge {
  display: inline-block;
  padding: 0.1rem 0.5rem;
  border-radius: 0.8rem;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  white-space: nowrap;
  flex-shrink: 0;
}
.env-badge-math { background: #dbeafe; color: #1d4ed8; }
.env-badge-coding { background: #dcfce7; color: #166534; }
.env-badge-agent { background: #ffedd5; color: #9a3412; }
.env-badge-knowledge { background: #ede9fe; color: #5b21b6; }
.env-badge-safety { background: #fee2e2; color: #991b1b; }
.env-badge-instruction_following { background: #cffafe; color: #164e63; }
.env-badge-rlhf { background: #fce7f3; color: #9d174d; }
.env-badge-other { background: #f3f4f6; color: #374151; }
.env-badge-games { background: #fef9c3; color: #854d0e; }
@media (prefers-color-scheme: dark) {
  .env-badge-math { background: #1e3a5f; color: #93c5fd; }
  .env-badge-coding { background: #14532d; color: #86efac; }
  .env-badge-agent { background: #431407; color: #fdba74; }
  .env-badge-knowledge { background: #2e1065; color: #c4b5fd; }
  .env-badge-safety { background: #450a0a; color: #fca5a5; }
  .env-badge-instruction_following { background: #083344; color: #67e8f9; }
  .env-badge-rlhf { background: #500724; color: #f9a8d4; }
  .env-badge-other { background: #1f2937; color: #d1d5db; }
  .env-badge-games { background: #422006; color: #fde68a; }
}
.env-card-desc {
  font-size: 0.82rem;
  opacity: 0.8;
  margin: 0;
  line-height: 1.45;
  flex: 1;
}
.env-card-footer {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 0.75rem;
  align-items: center;
  margin-top: 0.3rem;
  font-size: 0.78rem;
}
.env-avail {
  display: inline-block;
  padding: 0.1rem 0.45rem;
  border-radius: 0.8rem;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.env-avail.yes { background: #dcfce7; color: #166534; }
.env-avail.no { background: #f3f4f6; color: #9ca3af; }
@media (prefers-color-scheme: dark) {
  .env-avail.yes { background: #14532d; color: #86efac; }
  .env-avail.no { background: #1f2937; color: #6b7280; }
}
.env-card-footer a {
  font-size: 0.78rem;
}
.env-license {
  opacity: 0.55;
  font-size: 0.72rem;
}
.env-grid-examples {
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
}
`;

function initEnvFilter() {
  const grid = document.getElementById("env-cards-grid");
  if (!grid) return;

  const cards = Array.from(grid.querySelectorAll(".env-card"));
  const filterBar = document.getElementById("env-filter-bar");
  const countLabel = document.getElementById("env-count");

  const domains = [...new Set(cards.map((c) => (c as HTMLElement).dataset.domain).filter(Boolean))].sort();

  let activeDomain = "all";
  let filterTrain = false;
  let filterVal = false;
  let sortKey = "name";

  function makeBtn(label: string, extraClass?: string) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.className = "env-filter-btn" + (extraClass ? " " + extraClass : "");
    return btn;
  }

  function makeGroup(labelText?: string) {
    const wrap = document.createElement("div");
    wrap.className = "env-filter-group";
    if (labelText) {
      const lbl = document.createElement("span");
      lbl.className = "env-filter-group-label";
      lbl.textContent = labelText;
      wrap.appendChild(lbl);
    }
    return wrap;
  }

  if (filterBar) {
    filterBar.className = "env-filter-bar";

    const domainGroup = makeGroup("Domain:");
    const allBtn = makeBtn("All", "active");
    allBtn.addEventListener("click", () => {
      activeDomain = "all";
      domainGroup.querySelectorAll(".env-filter-btn").forEach((b) => b.classList.remove("active"));
      allBtn.classList.add("active");
      apply();
    });
    domainGroup.appendChild(allBtn);

    domains.forEach((domain) => {
      const btn = makeBtn(domain);
      btn.addEventListener("click", () => {
        activeDomain = domain;
        domainGroup.querySelectorAll(".env-filter-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        apply();
      });
      domainGroup.appendChild(btn);
    });
    filterBar.appendChild(domainGroup);

    const typeGroup = makeGroup("Data:");
    const trainBtn = makeBtn("Train");
    trainBtn.addEventListener("click", () => {
      filterTrain = !filterTrain;
      trainBtn.classList.toggle("active", filterTrain);
      apply();
    });
    const valBtn = makeBtn("Validation");
    valBtn.addEventListener("click", () => {
      filterVal = !filterVal;
      valBtn.classList.toggle("active", filterVal);
      apply();
    });
    typeGroup.appendChild(trainBtn);
    typeGroup.appendChild(valBtn);
    filterBar.appendChild(typeGroup);

    const sortGroup = makeGroup("Sort:");
    const sortSel = document.createElement("select");
    sortSel.className = "env-sort-select";
    [
      ["name", "A → Z"],
      ["domain", "Domain"],
    ].forEach(([val, label]) => {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = label;
      sortSel.appendChild(opt);
    });
    sortSel.addEventListener("change", () => {
      sortKey = sortSel.value;
      apply();
    });
    sortGroup.appendChild(sortSel);
    filterBar.appendChild(sortGroup);
  }

  function apply() {
    cards.forEach((card) => {
      const el = card as HTMLElement;
      const domainMatch = activeDomain === "all" || el.dataset.domain === activeDomain;
      const trainMatch = !filterTrain || el.dataset.train === "1";
      const valMatch = !filterVal || el.dataset.val === "1";
      el.style.display = domainMatch && trainMatch && valMatch ? "" : "none";
    });

    const visible = cards.filter((c) => (c as HTMLElement).style.display !== "none");
    visible.sort((a, b) => {
      const aEl = a as HTMLElement;
      const bEl = b as HTMLElement;
      if (sortKey === "domain") {
        const d = (aEl.dataset.domain || "").localeCompare(bEl.dataset.domain || "");
        if (d !== 0) return d;
      }
      return (aEl.dataset.name || "").localeCompare(bEl.dataset.name || "");
    });
    visible.forEach((c) => grid.appendChild(c));

    if (countLabel) {
      countLabel.textContent = `${visible.length} environment${visible.length !== 1 ? "s" : ""}`;
    }
  }

  apply();
}

export function EnvFilterInit() {
  useEffect(() => {
    initEnvFilter();
  }, []);

  return <style dangerouslySetInnerHTML={{ __html: envFilterCss }} />;
}
