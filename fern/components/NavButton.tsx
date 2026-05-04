/**
 * Wayfinding navigation button for tutorials.
 * Renders a styled pill button for prev/next/back navigation.
 * Requires the `.nav-button` CSS rules from main.css.
 *
 * Usage:
 *   <NavButton href="/latest/environment-tutorials" label="Back to Environment Tutorials" direction="back" />
 *   <NavButton href="/latest/environment-tutorials/multi-step-environment" label="Multi-Step Environment" direction="next" />
 *   <NavButton href="/latest/environment-tutorials/single-step-environment" label="Single-Step Environment" direction="prev" />
 */
export function NavButton({
  href,
  label,
  direction = "back",
}: {
  href: string;
  label: string;
  direction?: "back" | "prev" | "next";
}) {
  const arrow = direction === "next" ? "\u2192" : "\u2190";
  const text = direction === "next" ? `${label} ${arrow}` : `${arrow} ${label}`;

  return (
    <a href={href} className={`nav-button nav-button-${direction}`}>
      {text}
    </a>
  );
}
