import type { ReactNode } from "react";
import { useMemo, useState } from "react";

import { AGENTS, SHAPE_LABELS, agentSourceUrl } from "./agents-data";
import type { Agent, AgentShape } from "./agents-data";

/**
 * AgentCatalog — filterable grid of every shipped agent harness.
 *
 * Reads from a typed source-of-truth (./agents-data.ts) so claims can't drift.
 * Adding a harness lands by editing agents-data.ts.
 *
 * Usage in MDX:
 *   import { AgentCatalog } from "@/components/agents/AgentCatalog";
 *
 *   <AgentCatalog />
 */

export interface AgentCatalogProps {
  shape?: AgentShape;
}

export function AgentCatalog({ shape }: AgentCatalogProps): ReactNode {
  const [activeShape, setActiveShape] = useState<AgentShape | "all">(shape ?? "all");
  const [trainableOnly, setTrainableOnly] = useState(false);
  const [usesVerifyOnly, setUsesVerifyOnly] = useState(false);

  const filtered = useMemo(() => {
    return AGENTS.filter((a) => {
      if (activeShape !== "all" && a.shape !== activeShape) return false;
      if (trainableOnly && !a.trainable_multi_turn) return false;
      if (usesVerifyOnly && !a.uses_gym_verify) return false;
      return true;
    });
  }, [activeShape, trainableOnly, usesVerifyOnly]);

  return (
    <div className="ng-agent-catalog">
      <div className="ng-agent-catalog__filters" role="region" aria-label="Filter agents">
        <div className="ng-agent-catalog__filter-group">
          <span className="ng-agent-catalog__filter-label">Shape</span>
          <button
            type="button"
            className={`ng-agent-chip ${activeShape === "all" ? "ng-agent-chip--active" : ""}`}
            onClick={() => setActiveShape("all")}
          >
            All
          </button>
          {(Object.keys(SHAPE_LABELS) as AgentShape[]).map((s) => (
            <button
              key={s}
              type="button"
              className={`ng-agent-chip ${activeShape === s ? "ng-agent-chip--active" : ""}`}
              onClick={() => setActiveShape(s)}
            >
              {SHAPE_LABELS[s]}
            </button>
          ))}
        </div>

        <div className="ng-agent-catalog__filter-group">
          <label className="ng-agent-catalog__toggle">
            <input
              type="checkbox"
              checked={trainableOnly}
              onChange={(e) => setTrainableOnly(e.target.checked)}
            />
            Trainable multi-turn
          </label>
          <label className="ng-agent-catalog__toggle">
            <input
              type="checkbox"
              checked={usesVerifyOnly}
              onChange={(e) => setUsesVerifyOnly(e.target.checked)}
            />
            Uses Gym verify()
          </label>
        </div>

        <div className="ng-agent-catalog__count">
          {filtered.length} of {AGENTS.length}
        </div>
      </div>

      {filtered.length === 0 ? (
        <p className="ng-agent-catalog__empty">No agents match the current filters.</p>
      ) : (
        <div className="ng-agent-catalog__grid">
          {filtered.map((agent) => (
            <AgentCard key={agent.id} agent={agent} />
          ))}
        </div>
      )}
    </div>
  );
}

interface AgentCardProps {
  agent: Agent;
}

function AgentCard({ agent }: AgentCardProps): ReactNode {
  return (
    <div
      className={`ng-agent-card ${agent.recommended_default ? "ng-agent-card--recommended" : ""}`}
    >
      <div className="ng-agent-card__header">
        <code className="ng-agent-card__name">{agent.name}</code>
        <span className={`ng-agent-card__shape ng-agent-card__shape--${agent.shape}`}>
          {SHAPE_LABELS[agent.shape]}
        </span>
      </div>

      {agent.recommended_default && (
        <div className="ng-agent-card__default-banner">Default — start here</div>
      )}

      <p className="ng-agent-card__description">{agent.description}</p>

      <div className="ng-agent-card__use-when">
        <span className="ng-agent-card__field-label">Use when</span>
        <span>{agent.use_when}</span>
      </div>

      <div className="ng-agent-card__tags">
        {agent.tags.map((t) => (
          <span key={t} className="ng-agent-card__tag">
            {t.replace("_", " ")}
          </span>
        ))}
      </div>

      <div className="ng-agent-card__capabilities">
        <span
          className={`ng-agent-card__capability ${agent.trainable_multi_turn ? "is-yes" : "is-no"}`}
          title={
            agent.trainable_multi_turn
              ? "Rollouts preserve token IDs + context across turns; trainable as a multi-turn trajectory."
              : "Rollouts reset context per turn or break monotonic-token training; eval-only."
          }
        >
          {agent.trainable_multi_turn ? "✓" : "—"} Trainable multi-turn
        </span>
        <span
          className={`ng-agent-card__capability ${agent.uses_gym_verify ? "is-yes" : "is-no"}`}
          title={
            agent.uses_gym_verify
              ? "Pairs with a Gym resources server — verify() returns reward."
              : "External library owns scoring; bypasses Gym verify()."
          }
        >
          {agent.uses_gym_verify ? "✓" : "—"} Uses Gym verify()
        </span>
      </div>

      <a
        className="ng-agent-card__source"
        href={agentSourceUrl(agent)}
        target="_blank"
        rel="noopener noreferrer"
      >
        {agent.source_path} ↗
      </a>
    </div>
  );
}
