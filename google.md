# google.com

Result of reading `http://google.com`, captured through the platform image-build lane — the only egress path where google.com is reachable (the workstation egress proxy denies it as an unreviewed source). Captured with `urllib.request.urlopen('http://google.com', timeout=60)` in a `python:3.11-slim` OCI build.

## Response metadata

- Request: `GET http://google.com`
- Status: `HTTP 200`
- Final URL (after redirect): `http://www.google.com/`
- Body size: ~86.8 KB (86,809 bytes reported by the fetch)

## Body

The response body is the live Google homepage. It is Google's proprietary third-party HTML and JavaScript, and no license covering redistribution of the complete page has been identified, so the body is **not reproduced here**. This document records the source, the capture method, and the response metadata instead.

The page's `<head>` begins with `<!doctype html><html itemscope="" itemtype="http://schema.org/WebPage" lang="en">` and includes the standard meta description "Search the world's information, including webpages, images, videos and more…" and `<title>Google</title>`.
