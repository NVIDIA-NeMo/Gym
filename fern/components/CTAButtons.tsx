/**
 * CTA button group for landing/index pages.
 * Renders a row of pill-shaped action buttons with NVIDIA green accent.
 * Requires the `.cta-buttons` CSS rules from main.css.
 */
export type CTAItem = {
  label: string;
  href: string;
  variant?: "primary" | "secondary";
};

export function CTAButtons({ items }: { items: CTAItem[] }) {
  return (
    <div className="cta-buttons">
      {items.map((item) => (
        <a
          key={item.href}
          href={item.href}
          className={`cta-button ${item.variant === "secondary" ? "cta-secondary" : "cta-primary"}`}
        >
          {item.label}
        </a>
      ))}
    </div>
  );
}
