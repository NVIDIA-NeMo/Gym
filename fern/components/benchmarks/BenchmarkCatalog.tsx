import type { ReactNode } from "react";
import { useMemo, useState } from "react";

import {
  BENCHMARKS,
  DOMAIN_LABELS,
  PATTERN_LABELS,
  benchmarkSourceUrl,
} from "./benchmarks-data";
import type { Benchmark, Domain, ScorerPattern } from "./benchmarks-data";

/**
 * BenchmarkCatalog — filterable grid of every shipped resources server.
 *
 * Reads from a typed source-of-truth (./benchmarks-data.ts). Adding a
 * benchmark = appending an entry. Filter by domain, scorer pattern,
 * verified flag.
 *
 * Usage in MDX:
 *   import { BenchmarkCatalog } from "@/components/benchmarks/BenchmarkCatalog";
 *
 *   <BenchmarkCatalog />
 */

export interface BenchmarkCatalogProps {
  domain?: Domain;
  pattern?: ScorerPattern;
}

export function BenchmarkCatalog({ domain, pattern }: BenchmarkCatalogProps): ReactNode {
  const [activeDomain, setActiveDomain] = useState<Domain | "all">(domain ?? "all");
  const [activePattern, setActivePattern] = useState<ScorerPattern | "all">(pattern ?? "all");
  const [verifiedOnly, setVerifiedOnly] = useState(false);
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return BENCHMARKS.filter((b) => {
      if (activeDomain !== "all" && !b.domain.includes(activeDomain)) return false;
      if (activePattern !== "all" && b.pattern !== activePattern) return false;
      if (verifiedOnly && !b.verified) return false;
      if (q && !b.name.toLowerCase().includes(q) && !b.task.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [activeDomain, activePattern, verifiedOnly, search]);

  return (
    <div className="ng-bench-catalog">
      <div className="ng-bench-catalog__filters" role="region" aria-label="Filter benchmarks">
        <input
          type="search"
          className="ng-bench-catalog__search"
          placeholder={`Search ${BENCHMARKS.length} benchmarks…`}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />

        <div className="ng-bench-catalog__filter-group">
          <span className="ng-bench-catalog__filter-label">Domain</span>
          <button
            type="button"
            className={`ng-bench-chip ${activeDomain === "all" ? "ng-bench-chip--active" : ""}`}
            onClick={() => setActiveDomain("all")}
          >
            Any
          </button>
          {(Object.keys(DOMAIN_LABELS) as Domain[]).map((d) => (
            <button
              key={d}
              type="button"
              className={`ng-bench-chip ${activeDomain === d ? "ng-bench-chip--active" : ""}`}
              onClick={() => setActiveDomain(d)}
            >
              {DOMAIN_LABELS[d]}
            </button>
          ))}
        </div>

        <div className="ng-bench-catalog__filter-group">
          <span className="ng-bench-catalog__filter-label">Pattern</span>
          <button
            type="button"
            className={`ng-bench-chip ${activePattern === "all" ? "ng-bench-chip--active" : ""}`}
            onClick={() => setActivePattern("all")}
          >
            All
          </button>
          {(Object.keys(PATTERN_LABELS) as ScorerPattern[]).map((p) => (
            <button
              key={p}
              type="button"
              className={`ng-bench-chip ${activePattern === p ? "ng-bench-chip--active" : ""}`}
              onClick={() => setActivePattern(p)}
            >
              {PATTERN_LABELS[p]}
            </button>
          ))}
        </div>

        <div className="ng-bench-catalog__filter-group">
          <label className="ng-bench-catalog__toggle">
            <input
              type="checkbox"
              checked={verifiedOnly}
              onChange={(e) => setVerifiedOnly(e.target.checked)}
            />
            Verified only
          </label>
        </div>

        <div className="ng-bench-catalog__count">
          {filtered.length} of {BENCHMARKS.length}
        </div>
      </div>

      {filtered.length === 0 ? (
        <p className="ng-bench-catalog__empty">No benchmarks match the current filters.</p>
      ) : (
        <div className="ng-bench-catalog__grid">
          {filtered.map((benchmark) => (
            <BenchmarkCard key={benchmark.id} benchmark={benchmark} />
          ))}
        </div>
      )}
    </div>
  );
}

interface BenchmarkCardProps {
  benchmark: Benchmark;
}

function BenchmarkCard({ benchmark }: BenchmarkCardProps): ReactNode {
  return (
    <a
      className="ng-bench-card"
      href={benchmarkSourceUrl(benchmark)}
      target="_blank"
      rel="noopener noreferrer"
    >
      <div className="ng-bench-card__header">
        <code className="ng-bench-card__name">{benchmark.name}</code>
        {benchmark.verified ? (
          <span className="ng-bench-card__verified ng-bench-card__verified--yes" title="verified: true in YAML">
            ✓ Verified
          </span>
        ) : (
          <span className="ng-bench-card__verified ng-bench-card__verified--no" title="verified: false (default)">
            Unverified
          </span>
        )}
      </div>

      <p className="ng-bench-card__task">{benchmark.task}</p>

      <div className="ng-bench-card__tags">
        <span className={`ng-bench-card__pattern ng-bench-card__pattern--${benchmark.pattern}`}>
          {PATTERN_LABELS[benchmark.pattern]}
        </span>
        {benchmark.domain.map((d) => (
          <span key={d} className="ng-bench-card__domain">
            {DOMAIN_LABELS[d]}
          </span>
        ))}
      </div>
    </a>
  );
}
