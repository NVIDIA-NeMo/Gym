# GDP.pdf agent

Single-turn agent that materializes GDP.pdf's long-document input immediately before the policy call. It always includes LiteParse-extracted text for every page. For vision-capable policies it also supplies ordered page images.

The AA v4.3 delivery controls are explicit configuration:

- `image_dpi`: 150 by default; may be reduced no lower than 72
- `pages_per_image`: 1 by default; 2 or 4 creates page-labeled composites
- `max_images`: optional endpoint image-count cap; only leading pages lose image coverage and all page text remains
- `image_format`: the benchmark uses JPEG for opaque payload reduction; PNG remains available

The policy gets one user turn with no tools. Base64 images are stripped from persisted rollout artifacts after the model call.
