# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CapSolver integration boundary for visual public-site browser sessions.

The browser driver calls this hook at the same lifecycle points as the maintained
runner. The default implementation deliberately supports explicit page hooks
and fails closed when a challenge is detected but no reviewed solver is
installed; provider-specific challenge code stays replaceable.
"""

from __future__ import annotations

import hashlib
import importlib
import ipaddress
import logging
import os
import time
from typing import Any, Protocol
from urllib.parse import parse_qs, unquote, urlparse

import httpx


LOG = logging.getLogger("nemo_gym.resources_servers.visual_browser.captcha")


BROWSER_PROXY_CONFIG_ATTR = "_nemo_gym_browser_proxy_config"
CHALLENGE_TITLE_MARKERS = (
    "just a moment",
    "attention required",
    "checking your browser",
    "verify you are human",
    "human verification",
    "security check",
    "captcha",
)
CHALLENGE_TEXT_MARKERS = (
    "verify you are human",
    "performing security verification",
    "checking if the site connection is secure",
    "complete the security check",
    "complete the captcha",
    "please verify that you are not a robot",
    "protected by cloudflare",
)
CAPTCHA_FRAME_SELECTORS = (
    'iframe[src*="challenges.cloudflare.com"]',
    'iframe[src*="turnstile"]',
    'iframe[src*="google.com/recaptcha"]',
    'iframe[src*="recaptcha.net/recaptcha"]',
    'iframe[title*="Cloudflare"]',
    'iframe[title*="captcha"]',
    'iframe[title*="CAPTCHA"]',
)
CAPTCHA_INTERCEPT_SCRIPT = """(() => {
    if (window.__nemoGymTurnstileHookInstalled) return;
    window.__nemoGymTurnstileHookInstalled = true;
    window.__nemoGymTurnstileParams = null;
    window.__nemoGymTurnstileCallback = null;
    const capture = (params) => {
        if (!params || !params.sitekey) return;
        window.__nemoGymTurnstileParams = {
            websiteKey: params.sitekey,
            action: params.action || null,
            cdata: params.cData || params.cdata || null,
        };
        if (typeof params.callback === 'function') {
            window.__nemoGymTurnstileCallback = params.callback;
        }
    };
    const wrap = (turnstile) => {
        if (!turnstile || turnstile.__nemoGymWrapped || typeof turnstile.render !== 'function') {
            return turnstile;
        }
        const render = turnstile.render.bind(turnstile);
        turnstile.render = (container, params = {}) => {
            capture(params);
            return render(container, params);
        };
        turnstile.__nemoGymWrapped = true;
        return turnstile;
    };
    let value = window.turnstile;
    Object.defineProperty(window, 'turnstile', {
        configurable: true,
        get() { return value; },
        set(next) { value = wrap(next); },
    });
    if (value) value = wrap(value);
})();"""


def _origin(url: str) -> str:
    """Return a log-safe origin without query parameters or credentials."""

    parsed = urlparse(url)
    if not parsed.hostname:
        return "unknown"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme or 'unknown'}://{parsed.hostname}{port}"


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


class CaptchaSolver(Protocol):
    def maybe_solve(self, page: Any, *, phase: str) -> bool: ...


class NoopCaptchaSolver:
    def maybe_solve(self, page: Any, *, phase: str) -> bool:
        LOG.debug(
            "event=captcha_skipped provider=none phase=%s origin=%s",
            phase,
            _origin(getattr(page, "url", "")),
        )
        return False


class ModuleCaptchaSolver:
    """Load an operator-reviewed solver without coupling Gym to its secrets."""

    def __init__(self, spec: str) -> None:
        module_name, separator, attribute = spec.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("WA_CAPTCHA_SOLVER must use module.path:factory format")
        factory = getattr(importlib.import_module(module_name), attribute)
        self._solver = factory()
        if not hasattr(self._solver, "maybe_solve"):
            raise TypeError("captcha solver factory must return an object with maybe_solve(page, phase=...)")

    def maybe_solve(self, page: Any, *, phase: str) -> bool:
        started = time.monotonic()
        try:
            solved = bool(self._solver.maybe_solve(page, phase=phase))
        except Exception:
            LOG.exception(
                "event=captcha_solver_failed provider=custom phase=%s origin=%s elapsed_seconds=%.3f",
                phase,
                _origin(getattr(page, "url", "")),
                time.monotonic() - started,
            )
            raise
        LOG.info(
            "event=captcha_solver_complete provider=custom phase=%s origin=%s solved=%s elapsed_seconds=%.3f",
            phase,
            _origin(getattr(page, "url", "")),
            solved,
            time.monotonic() - started,
        )
        return solved


class CapSolverBrowserSolver:
    """Solve visible Turnstile/reCAPTCHA v2 and managed Cloudflare challenges."""

    CREATE_URL = "https://api.capsolver.com/createTask"
    RESULT_URL = "https://api.capsolver.com/getTaskResult"

    def __init__(self, api_key: str, *, timeout: float = 45.0, proxy_server: str = "") -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._solver_proxy_config = self._parse_proxy_server(proxy_server) if proxy_server else None
        self._completed_challenges: set[tuple[str, str, str]] = set()

    def maybe_solve(self, page: Any, *, phase: str) -> bool:
        started = time.monotonic()
        origin = _origin(getattr(page, "url", ""))
        challenge_signal = self._challenge_signal(page)
        blocking_challenge = challenge_signal is not None
        challenge = self._challenge(page)
        if not blocking_challenge:
            if challenge is None:
                LOG.debug(
                    "event=captcha_scan provider=capsolver phase=%s origin=%s challenge=none status=clear",
                    phase,
                    origin,
                )
            else:
                kind, site_key = challenge
                LOG.debug(
                    "event=captcha_scan provider=capsolver phase=%s origin=%s challenge=%s "
                    "status=nonblocking_background site_key_sha256=%s",
                    phase,
                    origin,
                    kind,
                    _fingerprint(site_key),
                )
            return False
        if challenge is None and challenge_signal is not None and self._is_managed_cloudflare_signal(challenge_signal):
            # A managed Cloudflare page can hide its Turnstile site key.  The
            # maintained reference runner falls back to CapSolver's proxy-bound
            # AntiCloudflareTask in this case, so preserve that behavior here.
            challenge = ("cloudflare", challenge_signal)
        if challenge is None:
            LOG.error(
                "event=captcha_unresolved provider=capsolver phase=%s origin=%s reason=site_key_missing signal=%s",
                phase,
                origin,
                challenge_signal,
            )
            raise RuntimeError("CAPTCHA challenge detected but no supported site key was found")
        kind, site_key = challenge
        identity = (origin, kind, _fingerprint(site_key))
        if identity in self._completed_challenges:
            LOG.warning(
                "event=captcha_rechallenged provider=capsolver phase=%s origin=%s challenge=%s "
                "action=retry site_key_sha256=%s",
                phase,
                origin,
                kind,
                identity[2],
            )
            self._completed_challenges.discard(identity)
        task = self._build_task(page, kind, site_key)
        task_type = str(task["type"])
        LOG.info(
            "event=captcha_detected provider=capsolver phase=%s origin=%s challenge=%s "
            "signal=%s site_key_sha256=%s task_type=%s",
            phase,
            origin,
            kind,
            challenge_signal,
            _fingerprint(site_key),
            task_type,
        )
        try:
            with httpx.Client(timeout=min(30.0, self._timeout)) as client:
                response = client.post(
                    self.CREATE_URL,
                    json={
                        "clientKey": self._api_key,
                        "task": task,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                task_id = payload.get("taskId")
                if not task_id:
                    raise RuntimeError(
                        f"CapSolver createTask failed: {payload.get('errorDescription', 'unknown error')}"
                    )
                task_fingerprint = _fingerprint(str(task_id))
                LOG.info(
                    "event=captcha_task_created provider=capsolver phase=%s origin=%s "
                    "challenge=%s provider_task_sha256=%s",
                    phase,
                    origin,
                    kind,
                    task_fingerprint,
                )
                deadline = time.monotonic() + self._timeout
                polls = 0
                while time.monotonic() < deadline:
                    time.sleep(1.0)
                    polls += 1
                    result = client.post(self.RESULT_URL, json={"clientKey": self._api_key, "taskId": task_id})
                    result.raise_for_status()
                    result_payload = result.json()
                    if result_payload.get("status") == "processing":
                        if polls == 1 or polls % 10 == 0:
                            LOG.debug(
                                "event=captcha_poll provider=capsolver phase=%s origin=%s "
                                "provider_task_sha256=%s polls=%d elapsed_seconds=%.3f status=processing",
                                phase,
                                origin,
                                task_fingerprint,
                                polls,
                                time.monotonic() - started,
                            )
                        continue
                    if result_payload.get("status") != "ready":
                        raise RuntimeError(
                            f"CapSolver task failed: {result_payload.get('errorDescription', 'unknown error')}"
                        )
                    solution = result_payload.get("solution") or {}
                    if kind == "cloudflare":
                        injection = self._apply_cloudflare_solution(page, solution)
                        try:
                            page.reload(wait_until="domcontentloaded")
                        except Exception:
                            pass
                    else:
                        token = solution.get("token") or solution.get("gRecaptchaResponse")
                        if not token:
                            raise RuntimeError("CapSolver returned no browser token")
                        injection = self._inject(page, kind, str(token))
                    if blocking_challenge and not self._wait_for_challenge_clear(page):
                        raise RuntimeError("CAPTCHA solution was injected but the challenge page did not clear")
                    self._completed_challenges.add(identity)
                    LOG.info(
                        "event=captcha_solved provider=capsolver phase=%s origin=%s challenge=%s "
                        "provider_task_sha256=%s polls=%d fields=%d callbacks=%d elapsed_seconds=%.3f",
                        phase,
                        origin,
                        kind,
                        task_fingerprint,
                        polls,
                        int(injection.get("fieldCount", 0)),
                        int(injection.get("callbacksCalled", 0)),
                        time.monotonic() - started,
                    )
                    return True
        except Exception:
            LOG.exception(
                "event=captcha_solver_failed provider=capsolver phase=%s origin=%s challenge=%s elapsed_seconds=%.3f",
                phase,
                origin,
                kind,
                time.monotonic() - started,
            )
            raise
        LOG.error(
            "event=captcha_solver_timeout provider=capsolver phase=%s origin=%s challenge=%s "
            "elapsed_seconds=%.3f timeout_seconds=%.1f",
            phase,
            origin,
            kind,
            time.monotonic() - started,
            self._timeout,
        )
        raise TimeoutError(f"CapSolver did not finish within {self._timeout:.1f}s")

    @staticmethod
    def _challenge(page: Any) -> tuple[str, str] | None:
        candidates: list[tuple[str, str]] = []
        for selector, kind in ((".cf-turnstile[data-sitekey]", "turnstile"), ("[data-sitekey]", "recaptcha")):
            locator = page.locator(selector)
            if locator.count():
                value = locator.first.get_attribute("data-sitekey")
                if value:
                    candidates.append((kind, value))
        for frame in page.frames:
            parsed = urlparse(frame.url)
            query = parse_qs(parsed.query)
            value = (query.get("k") or query.get("sitekey") or [None])[0]
            if not value:
                continue
            kind = "turnstile" if "cloudflare" in parsed.netloc or "turnstile" in frame.url else "recaptcha"
            candidates.append((kind, value))
        try:
            captured = page.evaluate(
                """() => {
                    const params = window.__nemoGymTurnstileParams;
                    return params && params.websiteKey ? params.websiteKey : null;
                }"""
            )
            if captured:
                candidates.insert(0, ("turnstile", str(captured)))
        except Exception:
            pass
        return candidates[0] if candidates else None

    @staticmethod
    def _challenge_signal(page: Any) -> str | None:
        try:
            title = str(page.title() or "").lower()
            for marker in CHALLENGE_TITLE_MARKERS:
                if marker in title:
                    return f"title:{marker}"
        except Exception:
            pass
        try:
            body = str(page.locator("body").inner_text(timeout=1_000) or "").lower()
            for marker in CHALLENGE_TEXT_MARKERS:
                if marker in body:
                    return f"body:{marker}"
        except Exception:
            pass
        for selector in CAPTCHA_FRAME_SELECTORS:
            try:
                locator = page.locator(selector)
                for index in range(min(locator.count(), 5)):
                    box = locator.nth(index).bounding_box()
                    if box and box.get("width", 0) > 20 and box.get("height", 0) > 20:
                        return f"visible_frame:{selector}"
            except Exception:
                pass
        return None

    @staticmethod
    def _is_managed_cloudflare_signal(signal: str) -> bool:
        return any(
            marker in signal
            for marker in (
                "title:just a moment",
                "title:attention required",
                "title:checking your browser",
                "body:checking if the site connection is secure",
                "body:performing security verification",
                "body:protected by cloudflare",
                "challenges.cloudflare.com",
                "turnstile",
            )
        )

    @classmethod
    def _is_challenge_page(cls, page: Any) -> bool:
        return cls._challenge_signal(page) is not None

    @staticmethod
    def _parse_proxy_server(proxy_server: str) -> dict[str, str]:
        parsed = urlparse(proxy_server if "://" in proxy_server else f"http://{proxy_server}")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("WA_CAPTCHA_PROXY_SERVER has an invalid port") from exc
        if parsed.scheme.lower() not in {"http", "https", "socks5"} or not parsed.hostname or not port:
            raise ValueError("WA_CAPTCHA_PROXY_SERVER is not a valid proxy URL")
        config = {"server": f"{parsed.scheme.lower()}://{parsed.hostname}:{port}"}
        if parsed.username:
            config["username"] = unquote(parsed.username)
        if parsed.password:
            config["password"] = unquote(parsed.password)
        return config

    @staticmethod
    def _is_loopback_proxy(config: dict[str, str]) -> bool:
        parsed = urlparse(config.get("server", ""))
        if parsed.hostname == "localhost":
            return True
        try:
            return ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            return False

    def _proxy_config(self, page: Any) -> dict[str, str] | None:
        # CapSolver's proxy identity must match the browser context that
        # received the challenge.  ``WA_CAPTCHA_PROXY_SERVER`` is only a
        # public, provider-reachable replacement for an actual browser proxy
        # (for example, when Chromium connects through a loopback tunnel).  It
        # must not turn a direct browser session into a proxy-bound solver
        # request: reCAPTCHA tokens are commonly tied to the request IP and
        # such a mismatch can leave otherwise valid tasks polling forever.
        try:
            browser_config = getattr(page.context, BROWSER_PROXY_CONFIG_ATTR, None)
        except Exception:
            browser_config = None
        if not browser_config:
            return None
        config = self._solver_proxy_config or browser_config
        if self._is_loopback_proxy(config):
            raise RuntimeError(
                "CapSolver cannot reach the browser's loopback proxy; set "
                "WA_CAPTCHA_PROXY_SERVER to the public endpoint for the same proxy"
            )
        return dict(config)

    def _proxy_fields(self, page: Any) -> dict[str, Any]:
        config = self._proxy_config(page)
        if config is None:
            return {}
        parsed = urlparse(str(config.get("server", "")))
        try:
            port = parsed.port
        except ValueError:
            port = None
        if parsed.scheme.lower() not in {"http", "https", "socks5"} or not parsed.hostname or not port:
            return {}
        fields: dict[str, Any] = {
            "proxyType": parsed.scheme.lower(),
            "proxyAddress": parsed.hostname,
            "proxyPort": port,
        }
        if config.get("username"):
            fields["proxyLogin"] = config["username"]
        if config.get("password"):
            fields["proxyPassword"] = config["password"]
        return fields

    def _build_task(self, page: Any, kind: str, site_key: str) -> dict[str, Any]:
        if kind == "cloudflare":
            config = self._proxy_config(page)
            if config is None:
                raise RuntimeError("CapSolver AntiCloudflareTask requires the browser's public proxy")
            parsed = urlparse(str(config.get("server", "")))
            try:
                port = parsed.port
            except ValueError:
                port = None
            if not parsed.hostname or not port:
                raise RuntimeError("CapSolver AntiCloudflareTask requires a valid proxy host and port")
            proxy = f"{parsed.hostname}:{port}"
            if config.get("username") and config.get("password"):
                proxy = f"{proxy}:{config['username']}:{config['password']}"
            task: dict[str, Any] = {
                "type": "AntiCloudflareTask",
                "websiteURL": page.url,
                "proxy": proxy,
            }
            try:
                user_agent = page.evaluate("() => navigator.userAgent")
                if user_agent:
                    task["userAgent"] = str(user_agent)
            except Exception:
                pass
            try:
                html = page.content()
                if html:
                    task["html"] = str(html)
            except Exception:
                pass
            return task

        proxy_fields = self._proxy_fields(page)
        if kind == "turnstile":
            task_type = "AntiTurnstileTask" if proxy_fields else "AntiTurnstileTaskProxyLess"
        else:
            task_type = "ReCaptchaV2Task" if proxy_fields else "ReCaptchaV2TaskProxyLess"
        return {
            "type": task_type,
            "websiteURL": page.url,
            "websiteKey": site_key,
            **proxy_fields,
        }

    @staticmethod
    def _apply_cloudflare_solution(page: Any, solution: dict[str, Any]) -> dict[str, int]:
        raw_cookies = solution.get("cookies") or {}
        token = solution.get("token")
        if isinstance(raw_cookies, dict):
            cookies = dict(raw_cookies)
            if token and "cf_clearance" not in cookies:
                cookies["cf_clearance"] = token
            parsed = urlparse(page.url)
            origin = f"{parsed.scheme}://{parsed.netloc}/"
            records = [
                {"name": str(name), "value": str(value), "url": origin} for name, value in cookies.items() if value
            ]
        elif isinstance(raw_cookies, list):
            records = [dict(cookie) for cookie in raw_cookies if isinstance(cookie, dict)]
        else:
            records = []
        if not records:
            raise RuntimeError("CapSolver Cloudflare solution contained no cookies")
        page.context.add_cookies(records)
        return {"fieldCount": len(records), "callbacksCalled": 0}

    @classmethod
    def _wait_for_challenge_clear(cls, page: Any, *, timeout: float = 12.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not cls._is_challenge_page(page):
                return True
            try:
                page.wait_for_timeout(1_000)
            except Exception:
                time.sleep(1.0)
        return not cls._is_challenge_page(page)

    @staticmethod
    def _inject(page: Any, kind: str, token: str) -> dict[str, int]:
        field_name = "cf-turnstile-response" if kind == "turnstile" else "g-recaptcha-response"
        result = page.evaluate(
            """([name, token, kind]) => {
                let fields = Array.from(document.querySelectorAll(
                    `textarea[name="${name}"], input[name="${name}"], ` +
                    `textarea[name^="${name}"], input[name^="${name}"]`
                ));
                if (!fields.length) {
                    const field = document.createElement('textarea');
                    field.name = name;
                    field.style.display = 'none';
                    document.body.appendChild(field);
                    fields = [field];
                }
                for (const field of fields) {
                    field.value = token;
                    field.dispatchEvent(new Event('input', {bubbles: true}));
                    field.dispatchEvent(new Event('change', {bubbles: true}));
                }
                let callbacksCalled = 0;

                for (const element of document.querySelectorAll('[data-callback]')) {
                    const callbackName = element.getAttribute('data-callback');
                    const callback = callbackName && window[callbackName];
                    if (typeof callback === 'function') {
                        try { callback(token); callbacksCalled += 1; } catch (_) {}
                    }
                }

                if (kind === 'recaptcha') {
                    const callbacks = [];
                    const seen = new Set();
                    const scan = (object, depth = 0) => {
                        if (!object || depth > 8 || seen.has(object)) return;
                        if (typeof object !== 'object' && typeof object !== 'function') return;
                        seen.add(object);
                        for (const key of Object.keys(object)) {
                            let value;
                            try { value = object[key]; } catch (_) { continue; }
                            if ((key === 'callback' || key === 'promise-callback') &&
                                typeof value === 'function') {
                                callbacks.push(value);
                            } else if (value &&
                                       (typeof value === 'object' || typeof value === 'function')) {
                                scan(value, depth + 1);
                            }
                        }
                    };
                    if (window.___grecaptcha_cfg && window.___grecaptcha_cfg.clients) {
                        scan(window.___grecaptcha_cfg.clients);
                    }
                    for (const callback of callbacks) {
                        try { callback(token); callbacksCalled += 1; } catch (_) {}
                    }
                }
                if (typeof window.__nemoGymTurnstileCallback === 'function') {
                    try {
                        window.__nemoGymTurnstileCallback(token);
                        callbacksCalled += 1;
                    } catch (_) {}
                }
                return {fieldCount: fields.length, callbacksCalled};
            }""",
            [field_name, token, kind],
        )
        if not isinstance(result, dict):
            return {"fieldCount": 0, "callbacksCalled": 0}
        return {
            "fieldCount": int(result.get("fieldCount", 0)),
            "callbacksCalled": int(result.get("callbacksCalled", 0)),
        }


def captcha_solver_from_environment() -> CaptchaSolver:
    """Resolve the approved solver implementation for a run.

    The built-in provider is selected by a CapSolver key. An explicit module
    remains available when an operator needs a reviewed browser-specific
    implementation.
    """

    spec = os.environ.get("WA_CAPTCHA_SOLVER", "").strip()
    if spec:
        LOG.info("event=captcha_solver_configured provider=custom spec=%s", spec)
        return ModuleCaptchaSolver(spec)
    api_key = os.environ.get("CAPSOLVER_API_KEY", "").strip()
    if api_key and os.environ.get("WA_CAPTCHA_PROVIDER", "capsolver").lower() == "capsolver":
        timeout = float(os.environ.get("WA_CAPTCHA_TIMEOUT", "45"))
        proxy_server = os.environ.get("WA_CAPTCHA_PROXY_SERVER", "").strip()
        LOG.info(
            "event=captcha_solver_configured provider=capsolver key_present=true "
            "timeout_seconds=%.1f solver_proxy_present=%s",
            timeout,
            bool(proxy_server),
        )
        return CapSolverBrowserSolver(api_key, timeout=timeout, proxy_server=proxy_server)
    LOG.info("event=captcha_solver_configured provider=none key_present=%s", bool(api_key))
    return NoopCaptchaSolver()
