# Broadsheet design system

Broadsheet is newsprint set for the web: near-black Source Serif 4 on paper white, with the process inks — cyan and magenta, completed by a print yellow in the press treatments — used small and deliberately, like spot color. Page structure uses no boxes and no dividers between sections; hierarchy comes from the serif scale and negative space alone. The one place rules print is as front-page furniture: the thick-thin head pair around a dateline rail, in full-strength ink. Photographs print as their own misregistered process plates.

The system's one product surface is **Daybreak**, an invented five-minute morning news digest used to demonstrate the system in its own voice (landing page, deck). There is no logo: the brand is set in type — the name in Source Serif 4 Semibold wherever a mark would go.

## Sources

- Mounted codebase `Broadsheet/` (read-only, the prior-account design system this project rebuilds): `styles.css` (tokens + component classes), `theme.json` (the parameters it was derived from), `readme.md`, `foundations/*.html`, `components/*.html`, `templates/landing/index.html`, `templates/deck/Deck.dc.html`, `print-plates.js`, `assets/photo.jpg`, `assets/photo-color.jpg`.
- Uploaded fonts: `uploads/dejavu-sans/`, `uploads/Geologica,Golos_Text,Hanken_Grotesk,Inter,Montserrat,etc/` — none of these is Source Serif 4, the system's only family; they are not used (see Caveats).
- User note: "Snorkel company branding tends to be navy blue and white." Not applied — the Broadsheet source defines its own palette and no Snorkel material was provided (see Caveats).

## Content fundamentals

- **Voice**: a newspaper's — dry, declarative, lightly wry. Sentences are complete and plain; jokes land deadpan ("edited by people who went to bed angry about adjectives", "Push notifications, ever — 0", "the pressroom's polite word for the waste basket").
- **Person**: second person to the reader ("your morning", "your inbox by six"); first person plural for the paper ("we consider that a feature of the news"). Never "I".
- **Casing**: sentence case everywhere — headlines, buttons ("Read today's", "Get started"), labels. Uppercase only for furniture: kickers, datelines, table headers, tags' context.
- **Headlines break per sentence**, one clause per line: "All the news. / None of the feed."
- **Numbers as furniture**: issue numbers with the numero sign and thousands comma ("№ 4,218"), times as clock figures ("6:00"), dot-leader indexes pointing at a figure.
- **Punctuation**: real typographic marks — curly quotes, em dashes with spaces (" — "), the middle dot as a separator ("Your name · July 2026"). Pull quotes hang their opening quote in the margin; attributions begin with "— ".
- **No emoji, no exclamation marks, no marketing superlatives.** Button labels are verbs or short nouns (Subscribe, Preview, Learn more, Publish, Cancel).
- **Microcopy examples**: dialog "Publish this page? — It goes live at its current URL. You can unpublish at any time and nothing else on the site changes." Footer: "Built from this project's theme.json — see readme.md."

## Visual foundations

- **Color**: paper `#f3f2f2`, surface `#eae9e9`, ink `#201e1d`; accents cyan `#0088b0` and magenta `#d6006c`; process yellow `#edbb00` for print treatments only. Each role has a 100–900 OKLCH ramp on one shared lightness scale: 100–300 tints/hovers/hairlines, 500 base, 600 hover, 700 pressed and small accent text, 800 text on tints. Cyan marks the interactive; magenta is the rarer second spot; never both in one small component. No dark surfaces anywhere — even deck dividers stay paper.
- **Type**: Source Serif 4 for everything (Regular 400, Semibold 600, true Italic 400); the serif is the chrome — no sans anywhere. Headings 42/32/25/20/16/13 at 1.12 and -0.015em; body 15px/1.55 in components, 15.5–17px on a 28px rhythm on pages; display heads clamp(42px, 6vw, 80px) at ≈1.08, -0.02em, optically shifted -0.035em left. Body columns are justified with real hyphenation. Figures proportional (`pnum`), tabular in datelines.
- **Spacing**: 5·10·15·20·30·40px (4px × 1.25 density — airy; never tighten). Pages run a 28px leading unit with a 14px half-step; heads are cap-trimmed (`text-box: trim-both cap alphabetic`) so baselines sit on the grid.
- **Layout**: left-aligned, asymmetric; content hugs the left edge with whitespace on the right. Max width 1200px, gutter clamp(20px, 5vw, 72px). Sections are separated by air alone. Nothing is fixed/sticky; the nav is a plain bar with no bottom rule.
- **Backgrounds**: flat paper. No gradients, no patterns, no illustrations. The only textures are the print treatments on imagery.
- **Imagery**: never a raw image. Content photographs print as four misregistered CMYK plates (`.cmyk`, via SVG feColorMatrix filters in `assets/print-plates.js`), screened with a 3px halftone dot; hovering gathers the plates into register and the pointer leans them a breath. Interface imagery takes `.halftone` (35% grayscale, contrast 1.15, dot screen). Image color is therefore press-like: multiplied inks on paper.
- **Borders & rules**: hairline `--color-divider` (ink 16%) on inputs, secondary buttons and table rows only. Head furniture is full ink (2px over 1px). `.hr` exists but is discouraged.
- **Corner radii**: near-square — 1 / 2 / 4px. Tags at 1.5px, dialog 4px.
- **Shadows**: three ink-tinted levels (`--shadow-sm/md/lg`: 0 1px 2px 14%, 0 3px 10px 16%, 0 12px 32px 22%). Cards are flat by default; elevation is opt-in. No inner shadows.
- **Cards**: surface fill, 15px padding, 2px radius, no border — and the one boxed component; for discrete items only, never layout.
- **Hover**: primary → accent-600; pressed → accent-700; outlined → 7% ink tint (14% pressed); ghost → 10% accent tint (18% pressed); table rows 4% ink; links → cyan. **Focus**: 2px accent ring, offset 2. **Disabled**: 45% opacity. **Selection**: 30% accent.
- **Transparency & blur**: none as surfaces. Dialog scrim is neutral-900 at 50%, no blur. Muted text is ink at 78/70/55% (contrast-checked: 70% = 5.8:1 for small text).
- **Animation**: essentially none in the interface — no transitions on hover, no bounces. Motion lives only in the press treatments (plate registration easing on hover) and the deck's `.rise` entrances; the landing uses smooth scroll only. Respect `prefers-reduced-motion`.
- **Deck furniture** (templates/deck): corner crop marks and true registration targets at 65% ink, a 16px four-patch color bar on cover/close, dividers with a press field, B/W wedge, color scale and star target, plate numerals.

## Iconography

Phosphor icons (phosphoricons.com) in the **duotone** weight, inlined as SVG (viewBox 0 0 256 256) on `currentColor`: a 25%-opacity fill plate under the solid plate. Sizes: 13px in card meta and segmented options, 14px in text buttons, 16px in icon buttons, 20px standalone. The twelve glyphs the source ships are copied to `assets/icons/*-duotone.svg` and exposed via the `Icon` component (sparkle, layers, circle, arrow, search, settings, user, heart, bell, calendar, image, folder). No icon font, no PNG icons, no emoji. Unicode is used typographically (№, ·, —, “ ”), never as icons.

## Components

Built from the source's inventory (its CSS classes), exposed as React under `window.BroadsheetDesignSystem_b6f617`. Every component renders the source's class names; styling lives in `tokens/components.css`.

- `components/actions/` — **Button** (primary / secondary / ghost; icon; block), **Tag** (accent / accent-2 / neutral / outline)
- `components/forms/` — **Field**, **Input** (multiline), **Radio** + **RadioGroup**, **Segmented**
- `components/surfaces/` — **Card** (kicker, title, body, meta; elevation sm/md/lg), **Dialog**
- `components/navigation/` — **Nav**
- `components/data/` — **Table**
- `components/print/` — **Cmyk** (four-plate photograph), **Halftone**, **PlateText** (`.cmyk-num` / `.cmyk-head`), **HeadRule** + **Dateline**
- `components/icons/` — **Icon** (+ `PHOSPHOR_DUOTONE` path data)

Intentional additions (no class of their own in the source): `Icon` wraps the inlined Phosphor SVGs; `RadioGroup`, `Dateline` and `HeadRule` name markup patterns the source's landing template uses without a component class.

## Do / Don't

Do: separate sections with whitespace; set everything in the serif; use cyan for interactive, magenta rarely; print photographs as plates.
Don't: structure with rules, borders or boxes; use both accents in one small component; tighten the spacing scale; introduce a sans-serif; place an untreated image; use dark surfaces.

## Index

- `styles.css` — global entry (imports only) → `tokens/fonts.css`, `colors.css`, `typography.css`, `spacing.css`, `base.css`, `print.css`, `components.css`
- `assets/` — `photo.jpg`, `photo-color.jpg` (reference photographs), `print-plates.js` (CMYK filter defs + press driver; load beside the stylesheet wherever `.cmyk` is used), `icons/` (12 Phosphor duotone SVGs)
- `guidelines/` — foundation cards (Colors ×6, Type ×5, Spacing ×4, Interaction, Brand ×2) and `theme.json` (the source's parameters)
- `components/<group>/` — see above; each folder has a `*.card.html`
- `ui_kits/daybreak/` — click-through Daybreak site: front page, back issues, preferences
- `templates/landing/` — the source's landing starter (plain HTML), `templates/deck/` — the source's 23-slide deck starter (Design Component)
- `thumbnail.html` — project tile; `SKILL.md` — agent skill entry

## Caveats

- **Fonts**: Source Serif 4 loads from Google Fonts; no binaries were supplied. The uploaded fonts (DejaVu Sans, Geologica, Golos Text, Hanken Grotesk, Inter, Montserrat, Orbitron, Quantico, Schibsted Grotesk, Sora, Space Grotesk) are all sans-serifs and do not belong to this system, so they are unused.
- **Snorkel navy/white note** not applied: the source is Broadsheet (cyan/magenta on paper). Say so if this system should be re-skinned.
- Back-issues and preferences screens in the UI kit are compositions of source components, not recreations of source screens (the source has only the landing and the deck).
