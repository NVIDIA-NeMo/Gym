"""Opt-in local vision judge transport; originals and rubric stay unchanged."""
import asyncio
import base64
import copy
import io
import json
import os
import random
import re
from pathlib import Path

from gdpval_upstream_media import pdf_bytes_to_blocks

REVISION = 'c40eb47f74f105239eabb93cb6876d0ce603b8c4'


def open_judge_image(data):
    """Read JPEG header, request reduced DCT decoding, then enforce pixel budget.

    Header-only exception is JPEG-specific. No full-resolution pixels are loaded
    with Pillow's guard suspended. Progressive huge JPEGs are rejected because
    their coefficient storage can still scale with the original dimensions.
    """
    from PIL import Image
    if data.startswith(b'\xff\xd8\xff'):
        old_limit = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = None
            source = Image.open(io.BytesIO(data), formats=['JPEG'])
        finally:
            Image.MAX_IMAGE_PIXELS = old_limit
    else:
        source = Image.open(io.BytesIO(data))
    original_size = source.size
    try:
        if source.format == 'JPEG':
            if source.info.get('progressive') or source.info.get('progression'):
                if original_size[0] * original_size[1] > 50_000_000:
                    raise ValueError('Huge progressive JPEG requires offline conversion')
            source.draft('RGB', (2048, 2048))
        if source.width * source.height > 50_000_000:
            raise ValueError(f'Image still exceeds 50 million pixels after decoder reduction: {source.size}')
        return source, original_size
    except BaseException:
        source.close()
        raise


# Page/raster budget for PDF deliverables. Each rasterized page costs ~2k prompt
# tokens on the hub VLMs at 144 DPI (measured 17.5k tokens for 8 pages), so long
# documents are rendered at a lower DPI instead of being refused outright.
#   JUDGE_MAX_PAGES   hard cap; a longer PDF is still refused (default 120, was 50)
#   JUDGE_DPI         DPI up to JUDGE_DPI_PAGES pages (default 144)
#   JUDGE_LOW_DPI     DPI beyond that (default 96)
#   JUDGE_DPI_PAGES   page count where the DPI steps down (default 50)
def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def page_budget(pages):
    """(dpi, max_pages) for a PDF with *pages* pages under the env budget."""
    max_pages = _env_int('JUDGE_MAX_PAGES', 120)
    dpi = _env_int('JUDGE_DPI', 144) if pages <= _env_int('JUDGE_DPI_PAGES', 50) else _env_int('JUDGE_LOW_DPI', 96)
    return dpi, max_pages


def prepare_request(request, model, *, max_tokens=8192, dpi=None, max_pages=None):
    import fitz
    from PIL import Image, ImageOps
    req = copy.deepcopy(request)
    req['model'] = model
    req['max_tokens'] = max_tokens
    req.pop('max_completion_tokens', None)
    req.pop('reasoning_effort', None)
    req.pop('extra_body', None)  # Gemini-specific settings are not sent to MiniMax.
    audit = []
    for message in req['messages']:
        if not isinstance(message.get('content'), list):
            continue
        blocks = []
        for part in message['content']:
            if part['type'] == 'text':
                if re.match(r'^\s*[^\n]+: \[Error:', part.get('text', '')):
                    raise ValueError('Unreadable judge deliverable; refusing incomplete grading')
                blocks.append(part)
                continue
            if part['type'] == 'video_url':
                # Only video-capable judges get the clip; others get a note so the
                # request stays valid and the judge knows a video was delivered.
                url = part['video_url']['url']
                if not url.startswith('data:video/') or ';base64,' not in url:
                    raise ValueError('Only saved inline video is supported')
                size = len(url.split(',', 1)[1]) * 3 // 4
                if not judge_supports_video(model):
                    blocks.append({'type': 'text', 'text': f"[video deliverable, {size / 1e6:.1f} MB — this judge cannot view video]"})
                    audit.append({'kind': 'video', 'bytes': size, 'attached': False})
                    continue
                blocks.append(part)
                audit.append({'kind': 'video', 'bytes': size, 'attached': True})
                continue
            if part['type'] == 'input_audio':
                # Emitted by the shared reader only when audio_capable; forward to
                # judges with an audio tower, otherwise name the file so the judge
                # does not grade it as missing.
                if judge_supports_audio(model):
                    blocks.append(part)
                    audit.append({'kind': 'audio', 'attached': True})
                else:
                    blocks.append({'type': 'text', 'text': '[audio deliverable — this judge cannot listen to audio]'})
                    audit.append({'kind': 'audio', 'attached': False})
                continue
            if part['type'] != 'image_url':
                raise ValueError(f"Unsupported local-judge media: {part['type']}")
            url = part['image_url']['url']
            if not url.startswith('data:') or ';base64,' not in url:
                raise ValueError('Only saved inline media is supported')
            header, encoded = url.split(',', 1)
            data = base64.b64decode(encoded, validate=True)
            if data.startswith(b'%PDF-'):
                with fitz.open(stream=data, filetype='pdf') as doc:
                    pages = doc.page_count
                    page_dpi, page_cap = page_budget(pages)
                    if dpi is not None:
                        page_dpi = dpi
                    if max_pages is not None:
                        page_cap = max_pages
                    if doc.needs_pass or not 0 < pages <= page_cap:
                        raise ValueError(f'PDF cannot be fully judged: {pages} pages or encrypted (cap {page_cap})')
                    for page in doc:
                        if page.rect.width * page.rect.height * (page_dpi / 72)**2 > 40_000_000:
                            raise ValueError('PDF page exceeds raster pixel budget')
                rendered = pdf_bytes_to_blocks(data, dpi=page_dpi, max_pages=page_cap, include_text=True)
                if sum(b['type'] == 'image_url' for b in rendered) != pages:
                    raise ValueError('Incomplete PDF rendering; refusing partial judgment')
                blocks.extend(rendered)
                audit.append({'kind': 'pdf', 'bytes': len(data), 'pages': pages, 'dpi': page_dpi})
            elif header.startswith('data:image/'):
                source, original_size = open_judge_image(data)
                with source:
                    decoded_size = source.size
                    source.load()
                    img = ImageOps.exif_transpose(source).convert('RGB')
                    img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                    out = io.BytesIO()
                    img.save(out, format='JPEG', quality=90, optimize=True)
                    encoded_image = out.getvalue()
                    audit.append({'kind': 'image', 'original_bytes': len(data),
                                  'original_size': original_size, 'decoded_size': decoded_size, 'judge_size': img.size,
                                  'judge_bytes': len(encoded_image)})
                if len(encoded_image) > 8_000_000:
                    raise ValueError('Judge image exceeds byte budget')
                blocks.append({'type': 'image_url', 'image_url': {'url':
                    'data:image/jpeg;base64,' + base64.b64encode(encoded_image).decode()}})
            else:
                raise ValueError(f'Unsupported media header: {header}')
        message['content'] = blocks
    total_images = sum(p['type'] == 'image_url' for m in req['messages']
                       if isinstance(m.get('content'), list) for p in m['content'])
    image_cap = _env_int('JUDGE_MAX_IMAGES', 160)
    if total_images > image_cap:
        raise ValueError(f'Request exceeds {image_cap} images; no pages will be silently dropped')
    video_bytes = sum(a['bytes'] for a in audit if a.get('kind') == 'video' and a.get('attached'))
    if len(json.dumps(req).encode()) - video_bytes * 4 // 3 > 32_000_000:
        raise ValueError('Converted request exceeds 32 MB excluding video; adjust rendering explicitly')
    if video_bytes > 160_000_000:
        raise ValueError('Attached video exceeds 160 MB; refusing oversized judge request')
    return req, audit


async def send_request(request, endpoint, key, *, deadline=600):
    from openai import AsyncOpenAI, APIStatusError, APITimeoutError, APIConnectionError
    # One attempt may use the whole deadline: a 180 s per-attempt timeout cut off
    # long verdicts (Qwen3.8 thinking took 160 s median, 775 s p90 under load on
    # 2026-10-05), and each silent retry restarted generation until the deadline
    # expired -> 32 % judge_unavailable. Retries are for 429/5xx only.
    async def attempt():
        async with AsyncOpenAI(base_url=endpoint, api_key=key, max_retries=0, timeout=deadline) as client:
            for n in range(4):
                try:
                    return (await client.chat.completions.create(**request)).model_dump(mode='json')
                except (APIStatusError, APITimeoutError, APIConnectionError) as exc:
                    status = getattr(exc, 'status_code', None)
                    if isinstance(exc, APITimeoutError) or (status is not None and status not in (429, 500, 502, 503, 504)) or n == 3:
                        raise
                    delay = min(60, 5 * 2**n) + random.uniform(0, 3)
                    response = getattr(exc, 'response', None)
                    if response is not None:
                        try:
                            delay = max(delay, float(response.headers.get('retry-after', 0)))
                        except ValueError:
                            pass
                    await asyncio.sleep(delay)
    return await asyncio.wait_for(attempt(), timeout=deadline)


def judge_supports_video(model):
    """Judges that accept OpenAI/vLLM ``video_url`` data URLs (verified on the
    inference hub for the Qwen3.8 family, 2026-10-05; MiMo-V2.6 declares video input).
    GPT/Claude do not."""
    m = model.lower()
    return 'qwen' in m or 'gemini' in m or 'mimo' in m


def judge_supports_audio(model):
    """Judges with an audio tower served behind vLLM ``input_audio`` blocks.
    Qwen3.8-Flash-Next has none (text+vision only); MiMo-V2.6 has one."""
    return 'mimo' in model.lower()


def resolve_judge_endpoint(config):
    """(url, model, api_key) for this verdict.

    JUDGE_ENDPOINT_FILE (a file of `export KEY=value` lines written by a locally
    hosted judge server, see gdpval_rl_v2/judge/serve_qwen3_vl.sbatch) overrides
    the configured judge_url/judge_model when present and complete. It is read
    per verdict -- this module runs in a fresh subprocess each time -- so a
    judge server that restarts on another node is picked up without restarting
    the training job. Missing/partial file -> fall back to the config.
    """
    url, model, key = config['judge_url'], config['judge_model'], os.environ['JUDGE_API_KEY']
    path = os.environ.get('JUDGE_ENDPOINT_FILE')
    if path and os.path.isfile(path):
        values = {}
        for line in open(path):
            line = line.strip()
            if line.startswith('export ') and '=' in line:
                k, v = line[len('export '):].split('=', 1)
                values[k] = v.strip().strip('"').strip("'")
        if values.get('JUDGE_BASE_URL') and values.get('JUDGE_MODEL_NAME'):
            url, model = values['JUDGE_BASE_URL'], values['JUDGE_MODEL_NAME']
            k = values.get('JUDGE_API_KEY', '')
            if k and not k.startswith('$'):
                key = k
    return url, model, key


async def local_completion(body, config, task):
    from nemo_gym.openai_utils import NeMoGymChatCompletion
    judge_url, model, judge_key = resolve_judge_endpoint(config)
    is_gpt55 = model == 'openai/openai/gpt-5.5'
    is_qwen = 'qwen' in model.lower()
    is_mimo = 'mimo' in model.lower()
    # Qwen3.8 on the hub thinks by default and the hub caps max_output_tokens at
    # 32768; 16384 leaves room for reasoning plus the structured JSON verdict.
    request, audit = prepare_request(body.model_dump(mode='json', exclude_unset=True), model,
                                     max_tokens=32768 if is_gpt55 else (16384 if (is_qwen or is_mimo) else 8192))
    if is_gpt55:
        request.pop('temperature', None)
        request['reasoning_effort'] = 'high'
    elif (is_qwen or is_mimo) and os.environ.get('JUDGE_THINKING', '0') in ('0', 'false', 'False'):
        # MiMo-V2.6 with thinking on spent the whole 16k budget reasoning on 13% of verdicts
        # (finish_reason=length, empty content -> invalid judgement); its template honours
        # enable_thinking=false like Qwen's, so both local judges run thinking-off by default.
        # Thinking off by default: with it on, verdicts ran ~10k completion tokens
        # and 160 s median / 775 s p90 on the shared hub, blowing the judge
        # deadline for a third of rollouts (2026-10-05). JUDGE_THINKING=1 re-enables.
        # vLLM chat-template switch; non-standard, so it rides in extra_body.
        request['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
    with (Path(task) / 'local-judge-media.jsonl').open('a') as stream:
        stream.write(json.dumps({'revision': REVISION, 'media': audit}) + '\n')
    with (Path(task) / 'local-judge-media.jsonl').open('a') as stream:
        stream.write(json.dumps({'revision': REVISION, 'judge_url': judge_url, 'judge_model': model}) + '\n')
    response = await send_request(request, judge_url, judge_key,
                                  deadline=max(60, min(config['judge_timeout'] - 60, 1140)))
    return NeMoGymChatCompletion.model_validate(response)
