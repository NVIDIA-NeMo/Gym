# NVIDIA Fern Theme

Shared branding assets for NVIDIA Fern documentation sites.

## Contents

| File | Purpose |
|------|---------|
| `assets/NVIDIA_dark.svg` | Header logo (dark mode) |
| `assets/NVIDIA_light.svg` | Header logo (light mode) |
| `assets/NVIDIA_symbol.svg` | Favicon |
| `main.css` | Full NVIDIA brand CSS (colors, typography, layout, dark/light, footer, cards, landing) |
| `components/CustomFooter.tsx` | NVIDIA footer with privacy links and Built with Fern badge |
| `components/BadgeLinks.tsx` | Horizontal badge layout (replaces stacked Markdown badge images) |
| `docs-theme.yml` | Template `docs.yml` theme config — merge into your project's `docs.yml` (includes Adobe Launch `js` for docs.nvidia.com) |

## Usage

1. Copy `assets/`, `main.css`, and `components/CustomFooter.tsx` into your `fern/` directory.
2. Merge the settings from `docs-theme.yml` into your `fern/docs.yml`, replacing placeholders.
3. The `instances` URL must follow the format `https://<projname>.docs.buildwithfern.com`.

## Origin

Extracted from the [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) Fern docs.
