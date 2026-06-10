/**
 * Wayfinding navigation button for tutorials.
 * Renders a styled pill button for prev/next/back navigation.
 *
 * Styles are inlined as a sibling <style> block. Fern's MDX-component bundler
 * does not resolve `import "./*.css"` side-effects, and the `nvidia` global
 * theme owns the docs.yml `css:` field, so per-component styles live here.
 * Brand variables (--nv-*, --rounded) come from the global theme's :root.
 *
 * Usage:
 *   <NavButton href="/latest/environment-tutorials" label="Back to Environment Tutorials" direction="back" />
 *   <NavButton href="/latest/environment-tutorials/multi-step-environment" label="Multi-Step Environment" direction="next" />
 *   <NavButton href="/latest/environment-tutorials/single-step-environment" label="Single-Step Environment" direction="prev" />
 */

const navButtonCss = `
.nav-button {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.4rem 1rem;
    border-radius: var(--rounded, 999px);
    font-size: 0.875rem;
    font-weight: 500;
    text-decoration: none;
    border: 1px solid var(--nv-light-grey-3, #DDDDDD);
    color: var(--nv-color-black, #000000);
    background: transparent;
    transition: border-color 0.2s ease, color 0.2s ease, background-color 0.2s ease;
    white-space: nowrap;
}
.nav-button:hover {
    border-color: var(--nv-color-green, #74B900);
    color: var(--nv-color-green, #74B900);
}
.dark .nav-button {
    color: var(--nv-color-white, #FFFFFF);
    border-color: var(--nv-dark-grey-4, #333333);
}
.dark .nav-button:hover {
    border-color: var(--nv-color-green, #74B900);
    color: var(--nv-color-green, #74B900);
}
`;

export function NavButton({
  href,
  label,
  direction = "back",
}: {
  href: string;
  label: string;
  direction?: "back" | "prev" | "next";
}) {
  const arrow = direction === "next" ? "→" : "←";
  const text = direction === "next" ? `${label} ${arrow}` : `${arrow} ${label}`;

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: navButtonCss }} />
      <a href={href} className={`nav-button nav-button-${direction}`}>
        {text}
      </a>
    </>
  );
}
