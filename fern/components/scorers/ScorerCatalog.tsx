import type { ReactNode } from "react";
import { useMemo, useState } from "react";

import { PATTERN_LABELS, SCORERS, sourceUrl } from "./scorers-data";
import type { Scorer, ScorerPattern } from "./scorers-data";

/**
 * ScorerCatalog — filterable grid of every shipped `verify()` pattern.
 *
 * Reads from a typed source-of-truth (./scorers-data.ts) so claims can't drift:
 * every card cites a source_path + verify_line that the next Content Audit
 * can re-verify with grep. New scorers land by editing scorers-data.ts.
 *
 * Usage in MDX:
 *   import { ScorerCatalog } from "@/components/scorers/ScorerCatalog";
 *
 *   <ScorerCatalog />
 *
 * Or filtered to a single pattern (used inline on category sub-sections):
 *   <ScorerCatalog pattern="llm-judge" />
 */

export interface ScorerCatalogProps {
  /** Optional — restrict the catalog to a single pattern. */
  pattern?: ScorerPattern;
  /** Optional — restrict to scorers that match all of these domain tags. */
  domain?: string;
}

const ALL_DOMAINS = ["math", "coding", "agent", "knowledge", "instruction_following", "safety", "other"] as const;

export function ScorerCatalog({ pattern, domain }: ScorerCatalogProps): ReactNode {
  const [activePattern, setActivePattern] = useState<ScorerPattern | "all">(pattern ?? "all");
  const [activeDomain, setActiveDomain] = useState<string | "all">(domain ?? "all");
  const [judgeOnly, setJudgeOnly] = useState(false);
  const [deterministicOnly, setDeterministicOnly] = useState(false);

  const filtered = useMemo(() => {
    return SCORERS.filter((s) => {
      if (activePattern !== "all" && s.pattern !== activePattern) return false;
      if (activeDomain !== "all" && !s.domain.includes(activeDomain)) return false;
      if (judgeOnly && !s.judge_required) return false;
      if (deterministicOnly && !s.deterministic) return false;
      return true;
    });
  }, [activePattern, activeDomain, judgeOnly, deterministicOnly]);

  return (
    <div className="ng-scorer-catalog">
      <div className="ng-scorer-catalog__filters" role="region" aria-label="Filter scorers">
        <div className="ng-scorer-catalog__filter-group">
          <span className="ng-scorer-catalog__filter-label">Pattern</span>
          <button
            type="button"
            className={`ng-scorer-chip ${activePattern === "all" ? "ng-scorer-chip--active" : ""}`}
            onClick={() => setActivePattern("all")}
          >
            All
          </button>
          {(Object.keys(PATTERN_LABELS) as ScorerPattern[]).map((p) => (
            <button
              key={p}
              type="button"
              className={`ng-scorer-chip ${activePattern === p ? "ng-scorer-chip--active" : ""}`}
              onClick={() => setActivePattern(p)}
            >
              {PATTERN_LABELS[p]}
            </button>
          ))}
        </div>

        <div className="ng-scorer-catalog__filter-group">
          <span className="ng-scorer-catalog__filter-label">Domain</span>
          <button
            type="button"
            className={`ng-scorer-chip ${activeDomain === "all" ? "ng-scorer-chip--active" : ""}`}
            onClick={() => setActiveDomain("all")}
          >
            Any
          </button>
          {ALL_DOMAINS.map((d) => (
            <button
              key={d}
              type="button"
              className={`ng-scorer-chip ${activeDomain === d ? "ng-scorer-chip--active" : ""}`}
              onClick={() => setActiveDomain(d)}
            >
              {d.replace("_", " ")}
            </button>
          ))}
        </div>

        <div className="ng-scorer-catalog__filter-group">
          <label className="ng-scorer-catalog__toggle">
            <input
              type="checkbox"
              checked={judgeOnly}
              onChange={(e) => setJudgeOnly(e.target.checked)}
            />
            Judge required
          </label>
          <label className="ng-scorer-catalog__toggle">
            <input
              type="checkbox"
              checked={deterministicOnly}
              onChange={(e) => setDeterministicOnly(e.target.checked)}
            />
            Deterministic
          </label>
        </div>

        <div className="ng-scorer-catalog__count">
          {filtered.length} of {SCORERS.length}
        </div>
      </div>

      {filtered.length === 0 ? (
        <p className="ng-scorer-catalog__empty">
          No scorers match the current filters.
        </p>
      ) : (
        <div className="ng-scorer-catalog__grid">
          {filtered.map((scorer) => (
            <ScorerCard key={scorer.id} scorer={scorer} />
          ))}
        </div>
      )}
    </div>
  );
}

interface ScorerCardProps {
  scorer: Scorer;
}

function ScorerCard({ scorer }: ScorerCardProps): ReactNode {
  return (
    <div className="ng-scorer-card">
      <div className="ng-scorer-card__header">
        <code className="ng-scorer-card__name">{scorer.name}</code>
        <span className={`ng-scorer-card__pattern ng-scorer-card__pattern--${scorer.pattern}`}>
          {PATTERN_LABELS[scorer.pattern]}
        </span>
      </div>

      <p className="ng-scorer-card__description">{scorer.description}</p>

      <div className="ng-scorer-card__use-when">
        <span className="ng-scorer-card__field-label">Use when</span>
        <span>{scorer.use_when}</span>
      </div>

      <div className="ng-scorer-card__tags">
        {scorer.domain.map((d) => (
          <span key={d} className="ng-scorer-card__tag">
            {d.replace("_", " ")}
          </span>
        ))}
        {scorer.judge_required && (
          <span className="ng-scorer-card__tag ng-scorer-card__tag--judge">judge</span>
        )}
        {scorer.deterministic && (
          <span className="ng-scorer-card__tag ng-scorer-card__tag--deterministic">deterministic</span>
        )}
      </div>

      {scorer.config_knobs.length > 0 && (
        <div className="ng-scorer-card__knobs">
          <span className="ng-scorer-card__field-label">Config</span>
          <span className="ng-scorer-card__knobs-list">
            {scorer.config_knobs.map((k, i) => (
              <code key={k}>
                {k}
                {i < scorer.config_knobs.length - 1 ? ", " : ""}
              </code>
            ))}
          </span>
        </div>
      )}

      <a
        className="ng-scorer-card__source"
        href={sourceUrl(scorer)}
        target="_blank"
        rel="noopener noreferrer"
      >
        {scorer.source_path}:{scorer.verify_line} ↗
      </a>
    </div>
  );
}
