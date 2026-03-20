/* Environment card filtering and sorting for docs/environments/index.html */
(function () {
  "use strict";

  function init() {
    const grid = document.getElementById("env-cards-grid");
    if (!grid) return;

    const cards = Array.from(grid.querySelectorAll(".env-card"));
    const filterBar = document.getElementById("env-filter-bar");
    const countLabel = document.getElementById("env-count");

    // ── Collect unique domains ───────────────────────────────────────
    const domains = [...new Set(cards.map((c) => c.dataset.domain).filter(Boolean))].sort();

    // ── State ────────────────────────────────────────────────────────
    let activeDomain = "all";
    let filterTrain = false;
    let filterVal = false;
    let sortKey = "name"; // "name" | "domain"

    // ── Helpers ──────────────────────────────────────────────────────
    function makeBtn(label, extraClass) {
      const btn = document.createElement("button");
      btn.textContent = label;
      btn.className = "env-filter-btn" + (extraClass ? " " + extraClass : "");
      return btn;
    }

    function makeGroup(labelText) {
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

    // ── Build filter bar ─────────────────────────────────────────────
    if (filterBar) {
      filterBar.className = "env-filter-bar";

      // Domain buttons
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

      // Data type toggles
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

      // Sort
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

    // ── Apply filters + sort ─────────────────────────────────────────
    function apply() {
      // Filter
      cards.forEach((card) => {
        const domainMatch = activeDomain === "all" || card.dataset.domain === activeDomain;
        const trainMatch = !filterTrain || card.dataset.train === "1";
        const valMatch = !filterVal || card.dataset.val === "1";
        card.style.display = domainMatch && trainMatch && valMatch ? "" : "none";
      });

      // Sort visible cards in place
      const visible = cards.filter((c) => c.style.display !== "none");
      visible.sort((a, b) => {
        if (sortKey === "domain") {
          const d = (a.dataset.domain || "").localeCompare(b.dataset.domain || "");
          if (d !== 0) return d;
        }
        return (a.dataset.name || "").localeCompare(b.dataset.name || "");
      });
      visible.forEach((c) => grid.appendChild(c));

      // Count
      if (countLabel) {
        countLabel.textContent = `${visible.length} environment${visible.length !== 1 ? "s" : ""}`;
      }
    }

    // Initial render
    apply();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
