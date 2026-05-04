import type { ReactNode } from "react";

/**
 * VerifiedBadge — surfaces a resources server's `verified: true|false` flag in docs.
 *
 * The `verified` flag in resources_servers/<name>/configs/<name>.yaml is a load-bearing
 * signal: it asserts the env has been baselined on at least one instruct + one thinking
 * model with run-to-run reward variance < 1%, has documented validation recipes, and
 * has reviewer signoff. The pre-commit `add-verified-flag` hook auto-injects
 * `verified: false` on new servers; flipping to `true` requires team review.
 *
 * This component surfaces that flag visually so doc readers see the same assertion
 * the codebase enforces.
 *
 * Usage in MDX:
 *   import { VerifiedBadge } from "@/components/VerifiedBadge";
 *
 *   <VerifiedBadge verified baselineModels={["gpt-4.1", "Qwen3-30B"]} />
 *   <VerifiedBadge verified={false} />
 */

export interface VerifiedBadgeProps {
  /** Whether the resources server's YAML has `verified: true`. */
  verified: boolean;
  /** Optional list of models the verified flag was baselined against. Shown on hover/title. */
  baselineModels?: string[];
}

export function VerifiedBadge({ verified, baselineModels }: VerifiedBadgeProps): ReactNode {
  const baselineText =
    baselineModels && baselineModels.length > 0
      ? `Baselined on: ${baselineModels.join(", ")}`
      : verified
        ? "Baselined per the verified-flag contract"
        : "Example data only — not yet baselined for production use";

  if (verified) {
    return (
      <span className="ng-verified-badge ng-verified-badge--yes" title={baselineText}>
        <span className="ng-verified-badge__check" aria-hidden="true">
          ✓
        </span>
        <span className="ng-verified-badge__label">Verified</span>
        {baselineModels && baselineModels.length > 0 && (
          <span className="ng-verified-badge__detail">
            on {baselineModels.length} model{baselineModels.length === 1 ? "" : "s"}
          </span>
        )}
      </span>
    );
  }

  return (
    <span className="ng-verified-badge ng-verified-badge--no" title={baselineText}>
      <span className="ng-verified-badge__check" aria-hidden="true">
        ◯
      </span>
      <span className="ng-verified-badge__label">Unverified</span>
      <span className="ng-verified-badge__detail">example data only</span>
    </span>
  );
}
