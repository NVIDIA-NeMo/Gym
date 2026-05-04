import type { ReactNode } from "react";

import { PROVIDER_MATRIX, RL_FRAMEWORK_MATRIX, SANDBOX_MATRIX } from "./matrix-data";
import type { MatrixCell, MatrixDefinition, SupportStatus } from "./matrix-data";

/**
 * CompatibilityMatrix — render a typed matrix from a single source of truth.
 *
 * Three datasets ship today (provider, RL framework, sandbox); pick which one
 * to render via the `kind` prop. New rows or columns land by editing
 * ./matrix-data.ts — there's no per-page table to drift independently.
 *
 * Usage in MDX:
 *   import { CompatibilityMatrix } from "@/components/compatibility/CompatibilityMatrix";
 *
 *   <CompatibilityMatrix kind="provider" />
 *   <CompatibilityMatrix kind="rl-framework" />
 *   <CompatibilityMatrix kind="sandbox" />
 */

export interface CompatibilityMatrixProps {
  kind: "provider" | "rl-framework" | "sandbox";
  /** Hide the title and caption — useful when embedding inside a custom heading. */
  bare?: boolean;
}

const STATUS_LABEL: Record<SupportStatus, string> = {
  supported: "Supported",
  partial: "Partial",
  unsupported: "Not supported",
  planned: "Planned",
};

const STATUS_GLYPH: Record<SupportStatus, string> = {
  supported: "✓",
  partial: "◐",
  unsupported: "—",
  planned: "◯",
};

function pickMatrix(kind: CompatibilityMatrixProps["kind"]): MatrixDefinition {
  switch (kind) {
    case "provider":
      return PROVIDER_MATRIX;
    case "rl-framework":
      return RL_FRAMEWORK_MATRIX;
    case "sandbox":
      return SANDBOX_MATRIX;
  }
}

export function CompatibilityMatrix({ kind, bare }: CompatibilityMatrixProps): ReactNode {
  const matrix = pickMatrix(kind);

  return (
    <div className="ng-compat-matrix">
      {!bare && (
        <div className="ng-compat-matrix__header">
          <h4 className="ng-compat-matrix__title">{matrix.title}</h4>
          <p className="ng-compat-matrix__caption">{matrix.caption}</p>
        </div>
      )}

      <div className="ng-compat-matrix__scroll">
        <table className="ng-compat-matrix__table">
          <thead>
            <tr>
              <th scope="col" className="ng-compat-matrix__row-header">
                {/* Row label header */}
              </th>
              {matrix.columns.map((col) => (
                <th key={col.key} scope="col">
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.rows.map((row) => (
              <tr key={row.id}>
                <th scope="row" className="ng-compat-matrix__row-header">
                  {row.href ? (
                    <a href={row.href}>
                      <code>{row.label}</code>
                    </a>
                  ) : (
                    <code>{row.label}</code>
                  )}
                </th>
                {matrix.columns.map((col) => (
                  <td key={col.key}>
                    <CellRender cell={row.cells[col.key]} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="ng-compat-matrix__legend" aria-label="Status legend">
        {(["supported", "partial", "unsupported", "planned"] as SupportStatus[]).map((s) => (
          <span key={s} className={`ng-compat-matrix__legend-item ng-compat-matrix__cell--${s}`}>
            <span className="ng-compat-matrix__cell-glyph">{STATUS_GLYPH[s]}</span>
            {STATUS_LABEL[s]}
          </span>
        ))}
      </div>
    </div>
  );
}

interface CellRenderProps {
  cell: MatrixCell | undefined;
}

function CellRender({ cell }: CellRenderProps): ReactNode {
  if (!cell) {
    return <span className="ng-compat-matrix__cell ng-compat-matrix__cell--unsupported">—</span>;
  }

  const title = cell.detail
    ? cell.source
      ? `${cell.detail} (source: ${cell.source})`
      : cell.detail
    : cell.source
      ? `Source: ${cell.source}`
      : STATUS_LABEL[cell.status];

  return (
    <span
      className={`ng-compat-matrix__cell ng-compat-matrix__cell--${cell.status}`}
      title={title}
    >
      <span className="ng-compat-matrix__cell-glyph">{STATUS_GLYPH[cell.status]}</span>
      {cell.detail && <span className="ng-compat-matrix__cell-detail">{cell.detail}</span>}
    </span>
  );
}
