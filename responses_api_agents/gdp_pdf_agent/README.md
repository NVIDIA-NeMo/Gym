# GDP.pdf agent

Single-turn agent that materializes GDP.pdf's long-document input immediately before the policy call. It always includes LiteParse-extracted text for every page. For vision-capable policies it also supplies ordered page images.

Delivery adapts separately for each request following AA v4.3:

- Start at 150 DPI. On an explicit context or payload rejection, retry at 120, 96, 76, then 72 DPI. AA specifies the 150 DPI starting point and 72 DPI floor, but does not publish the decrement schedule; this implementation uses 20% reductions.
- For an endpoint image-count cap, use one page per image where possible, then two, then four. If four-up still exceeds the cap, retain leading-page images and every page's text. Composite cells are labeled and the prompt states image coverage.
- `max_images` optionally declares the endpoint cap in advance. Otherwise, explicit image-count rejections trigger adaptation; reported numeric caps are used directly, and an unknown cap is probed by lowering the image count.
- `image_format` selects JPEG (benchmark default) or PNG. JPEG quality is controlled by `jpeg_quality`.

Only explicit input-limit errors trigger adaptation. Authentication, rate-limit, transport, and judge failures do not change delivery. If the input still fails at 72 DPI, the attempt receives zero through the normal empty-answer verifier. Text is never truncated to make an input fit. Accepted answers are generated once; rejected input requests are not additional rollouts.

Each result records `document_delivery`: DPI, pages per image, image count, page coverage, and rejected profiles with their limit category. The endpoint is the authority on whether the complete request (including its output allowance and model-specific image processing) fits; no approximate tokenizer is used.

The policy gets one user turn with no tools. Seed and verifier requests retain the original lightweight inputs. Full text and page images are materialized only for the policy call and redacted from returned trajectory artifacts.
