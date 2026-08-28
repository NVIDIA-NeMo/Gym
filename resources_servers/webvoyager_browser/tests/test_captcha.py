# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest

from resources_servers.webvoyager_browser import captcha


class _Locator:
    def __init__(self, site_key: str | None) -> None:
        self._site_key = site_key
        self.first = self

    def count(self) -> int:
        return int(self._site_key is not None)

    def get_attribute(self, name: str) -> str | None:
        assert name == "data-sitekey"
        return self._site_key

    def nth(self, _index: int):
        return self

    def bounding_box(self):
        return None


class _Page:
    url = "https://example.test/form?private=query"
    frames: list = []

    def __init__(self, site_key: str | None = "public-site-key") -> None:
        self._site_key = site_key
        self.injected_token: str | None = None
        self.context = type("Context", (), {})()

    def locator(self, selector: str) -> _Locator:
        if selector.startswith(".cf-turnstile"):
            return _Locator(self._site_key)
        return _Locator(None)

    def evaluate(self, script: str, arguments: list[str]):
        _field_name, token, kind = arguments
        assert kind == "turnstile"
        assert "data-callback" in script
        assert "___grecaptcha_cfg" in script
        self.injected_token = token
        return {"fieldCount": 1, "callbacksCalled": 1}


class _BlockingPage(_Page):
    def title(self) -> str:
        return "Welcome" if self.injected_token else "Just a moment..."


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout
        self.requests: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def post(self, url: str, *, json: dict) -> _Response:
        self.requests.append((url, json))
        if url.endswith("createTask"):
            return _Response({"taskId": "provider-task-secret"})
        return _Response(
            {
                "status": "ready",
                "solution": {"token": "captcha-solution-secret"},
            }
        )


def test_capsolver_success_logs_lifecycle_without_secrets(monkeypatch, caplog) -> None:
    client = _Client(timeout=30.0)
    monkeypatch.setattr(captcha.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr(captcha.time, "sleep", lambda _seconds: None)
    page = _BlockingPage()
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)

    with caplog.at_level(logging.DEBUG, logger="nemo_gym.resources_servers.webvoyager_browser.captcha"):
        assert solver.maybe_solve(page, phase="initial") is True

    assert page.injected_token == "captcha-solution-secret"
    assert [url.rsplit("/", 1)[-1] for url, _payload in client.requests] == ["createTask", "getTaskResult"]
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=captcha_detected" in messages
    assert "event=captcha_task_created" in messages
    assert "event=captcha_solved" in messages
    assert "origin=https://example.test" in messages
    assert "fields=1" in messages
    assert "callbacks=1" in messages
    for secret in (
        "CAP-private-key",
        "captcha-solution-secret",
        "provider-task-secret",
        "public-site-key",
        "private=query",
    ):
        assert secret not in messages


def test_capsolver_no_challenge_emits_debug_scan(caplog) -> None:
    page = _Page(site_key=None)
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)

    with caplog.at_level(logging.DEBUG, logger="nemo_gym.resources_servers.webvoyager_browser.captcha"):
        assert solver.maybe_solve(page, phase="after wait") is False

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=captcha_scan" in messages
    assert "challenge=none" in messages
    assert "CAP-private-key" not in messages


def test_capsolver_ignores_nonblocking_widget_without_provider_request(monkeypatch, caplog) -> None:
    client = _Client(timeout=30.0)
    monkeypatch.setattr(captcha.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr(captcha.time, "sleep", lambda _seconds: None)
    blocking_page = _BlockingPage()
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)

    assert solver.maybe_solve(blocking_page, phase="after navigate") is True
    requests_after_real_challenge = list(client.requests)
    with caplog.at_level(logging.DEBUG, logger="nemo_gym.resources_servers.webvoyager_browser.captcha"):
        assert solver.maybe_solve(_Page(), phase="before post-action screenshot") is False

    assert client.requests == requests_after_real_challenge
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "status=nonblocking_background" in messages
    assert "event=captcha_unresolved" not in messages


def test_capsolver_retries_completed_challenge_that_reappears(monkeypatch, caplog) -> None:
    client = _Client(timeout=30.0)
    monkeypatch.setattr(captcha.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr(captcha.time, "sleep", lambda _seconds: None)
    page = _BlockingPage()
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)
    solver._completed_challenges.add(("https://example.test", "turnstile", captcha._fingerprint("public-site-key")))

    with caplog.at_level(logging.WARNING, logger="nemo_gym.resources_servers.webvoyager_browser.captcha"):
        assert solver.maybe_solve(page, phase="before post-action screenshot") is True

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=captcha_rechallenged" in messages
    assert "action=retry" in messages
    assert [url.rsplit("/", 1)[-1] for url, _payload in client.requests] == ["createTask", "getTaskResult"]
    assert "CAP-private-key" not in messages


def test_capsolver_failed_clear_is_not_cached_and_can_be_retried(monkeypatch) -> None:
    class _PersistentBlockingPage(_BlockingPage):
        def title(self) -> str:
            return "Just a moment..."

    client = _Client(timeout=30.0)
    monkeypatch.setattr(captcha.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr(captcha.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(captcha.CapSolverBrowserSolver, "_wait_for_challenge_clear", lambda *_args: False)
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)
    page = _PersistentBlockingPage()

    for phase in ("after computer", "before post-action screenshot"):
        with pytest.raises(RuntimeError, match="challenge page did not clear"):
            solver.maybe_solve(page, phase=phase)

    assert solver._completed_challenges == set()
    assert [url.rsplit("/", 1)[-1] for url, _payload in client.requests] == [
        "createTask",
        "getTaskResult",
        "createTask",
        "getTaskResult",
    ]


def test_capsolver_environment_selection_logs_presence_not_value(monkeypatch, caplog) -> None:
    monkeypatch.setenv("CAPSOLVER_API_KEY", "CAP-private-key")
    monkeypatch.setenv("WA_CAPTCHA_PROVIDER", "capsolver")
    monkeypatch.setenv(
        "WA_CAPTCHA_PROXY_SERVER",
        "http://proxy-user:proxy-password@proxy.example:19407",  # pragma: allowlist secret
    )

    with caplog.at_level(logging.INFO, logger="nemo_gym.resources_servers.webvoyager_browser.captcha"):
        solver = captcha.captcha_solver_from_environment()

    assert isinstance(solver, captcha.CapSolverBrowserSolver)
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "provider=capsolver" in messages
    assert "key_present=true" in messages
    assert "solver_proxy_present=True" in messages
    assert "CAP-private-key" not in messages
    assert "proxy-password" not in messages


def test_capsolver_task_uses_the_browser_proxy_without_logging_credentials() -> None:
    page = _Page()
    setattr(
        page.context,
        captcha.BROWSER_PROXY_CONFIG_ATTR,
        {
            "server": "http://proxy.example:19407",
            "username": "proxy-user",
            "password": "proxy-password",  # pragma: allowlist secret
        },
    )

    task = captcha.CapSolverBrowserSolver("CAP-private-key")._build_task(page, "turnstile", "public-site-key")

    assert task == {
        "type": "AntiTurnstileTask",
        "websiteURL": page.url,
        "websiteKey": "public-site-key",
        "proxyType": "http",
        "proxyAddress": "proxy.example",
        "proxyPort": 19407,
        "proxyLogin": "proxy-user",
        "proxyPassword": "proxy-password",  # pragma: allowlist secret
    }


def test_capsolver_explicit_public_proxy_overrides_browser_loopback_proxy() -> None:
    page = _Page()
    setattr(
        page.context,
        captcha.BROWSER_PROXY_CONFIG_ATTR,
        {"server": "http://127.0.0.1:19407"},
    )

    solver = captcha.CapSolverBrowserSolver(
        "CAP-private-key",
        proxy_server="http://proxy-user:proxy-password@proxy.example:29407",  # pragma: allowlist secret
    )
    task = solver._build_task(page, "recaptcha", "public-site-key")

    assert task["type"] == "ReCaptchaV2Task"
    assert task["proxyAddress"] == "proxy.example"
    assert task["proxyPort"] == 29407
    assert task["proxyLogin"] == "proxy-user"
    assert task["proxyPassword"] == "proxy-password"  # pragma: allowlist secret


def test_capsolver_explicit_proxy_is_ignored_for_direct_browser_context() -> None:
    page = _Page()
    solver = captcha.CapSolverBrowserSolver(
        "CAP-private-key",
        proxy_server="http://proxy-user:proxy-password@proxy.example:29407",  # pragma: allowlist secret
    )

    task = solver._build_task(page, "recaptcha", "public-site-key")

    assert task == {
        "type": "ReCaptchaV2TaskProxyLess",
        "websiteURL": page.url,
        "websiteKey": "public-site-key",
    }


def test_capsolver_rejects_browser_loopback_proxy_without_public_override() -> None:
    page = _Page()
    setattr(
        page.context,
        captcha.BROWSER_PROXY_CONFIG_ATTR,
        {"server": "http://127.0.0.1:19407"},
    )

    with pytest.raises(RuntimeError, match="WA_CAPTCHA_PROXY_SERVER"):
        captcha.CapSolverBrowserSolver("CAP-private-key")._build_task(page, "recaptcha", "public-site-key")


class _ChallengeBody:
    def inner_text(self, *, timeout: int) -> str:
        assert timeout == 1_000
        return "Checking if the site connection is secure"


class _GenericChallengeBody:
    def inner_text(self, *, timeout: int) -> str:
        assert timeout == 1_000
        return "Complete the captcha to continue"


class _NormalBody:
    def inner_text(self, *, timeout: int) -> str:
        assert timeout == 1_000
        return "Sports news, scores, schedules, and highlights"


class _ArticleBody:
    def inner_text(self, *, timeout: int) -> str:
        assert timeout == 1_000
        return "The interviewee said, ‘I am not a robot,’ while describing the performance."


class _FrameLocator(_Locator):
    def __init__(self, *, visible: bool) -> None:
        super().__init__(None)
        self._visible = visible

    def count(self) -> int:
        return 1

    def bounding_box(self):
        if self._visible:
            return {"width": 300, "height": 80}
        return None


class _BackgroundCaptchaFramePage(_Page):
    frames = [type("Frame", (), {"url": "https://google.com/recaptcha/api2/bframe"})()]

    def __init__(self, *, visible: bool) -> None:
        super().__init__(site_key=None)
        self._visible = visible

    def title(self) -> str:
        return "Sports homepage"

    def locator(self, selector: str):
        if selector == "body":
            return _NormalBody()
        if selector.startswith("iframe") and "recaptcha" in selector:
            return _FrameLocator(visible=self._visible)
        return _Locator(None)


class _ChallengePage(_Page):
    def __init__(self) -> None:
        super().__init__(site_key=None)

    def title(self) -> str:
        return "Captcha required"

    def locator(self, selector: str):
        if selector == "body":
            return _GenericChallengeBody()
        return _Locator(None)


class _ArticlePage(_Page):
    def __init__(self) -> None:
        super().__init__(site_key=None)

    def title(self) -> str:
        return "BBC interview"

    def locator(self, selector: str):
        if selector == "body":
            return _ArticleBody()
        return _Locator(None)


class _CloudflareContext:
    def __init__(self) -> None:
        self.cookies: list[dict] = []

    def add_cookies(self, cookies: list[dict]) -> None:
        self.cookies.extend(cookies)


class _CloudflarePage(_ChallengePage):
    def __init__(self) -> None:
        super().__init__()
        self.context = _CloudflareContext()
        setattr(
            self.context,
            captcha.BROWSER_PROXY_CONFIG_ATTR,
            {"server": "http://proxy.example:19407"},
        )
        self.reloaded = False

    def title(self) -> str:
        return "Welcome" if self.reloaded else "Just a moment..."

    def locator(self, selector: str):
        if selector == "body":
            return _NormalBody() if self.reloaded else _ChallengeBody()
        return _Locator(None)

    def evaluate(self, script: str):
        if "navigator.userAgent" in script:
            return "test-browser-agent"
        return None

    def content(self) -> str:
        return "<html><title>Just a moment...</title></html>"

    def reload(self, *, wait_until: str) -> None:
        assert wait_until == "domcontentloaded"
        self.reloaded = True


class _CloudflareClient(_Client):
    def post(self, url: str, *, json: dict) -> _Response:
        self.requests.append((url, json))
        if url.endswith("createTask"):
            return _Response({"taskId": "provider-task-secret"})
        return _Response(
            {
                "status": "ready",
                "solution": {"cookies": {"cf_clearance": "clearance-secret"}},
            }
        )


def test_article_phrase_is_not_treated_as_a_blocking_captcha(caplog) -> None:
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)

    with caplog.at_level(logging.DEBUG, logger="nemo_gym.resources_servers.webvoyager_browser.captcha"):
        assert solver.maybe_solve(_ArticlePage(), phase="after navigate") is False

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "status=clear" in messages
    assert "event=captcha_unresolved" not in messages


def test_capsolver_uses_proxy_bound_cloudflare_fallback_without_site_key(monkeypatch, caplog) -> None:
    client = _CloudflareClient(timeout=30.0)
    monkeypatch.setattr(captcha.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr(captcha.time, "sleep", lambda _seconds: None)
    page = _CloudflarePage()
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)

    with caplog.at_level(logging.INFO, logger="nemo_gym.resources_servers.webvoyager_browser.captcha"):
        assert solver.maybe_solve(page, phase="after navigate") is True

    create_task = client.requests[0][1]["task"]
    assert create_task["type"] == "AntiCloudflareTask"
    assert create_task["websiteURL"] == page.url
    assert create_task["proxy"] == "proxy.example:19407"
    assert create_task["userAgent"] == "test-browser-agent"
    assert page.reloaded is True
    assert page.context.cookies == [
        {
            "name": "cf_clearance",
            "value": "clearance-secret",
            "url": "https://example.test/",
        }
    ]
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "challenge=cloudflare" in messages
    assert "task_type=AntiCloudflareTask" in messages
    for secret in ("CAP-private-key", "provider-task-secret", "clearance-secret"):
        assert secret not in messages


def test_capsolver_fails_closed_for_blocking_challenge_without_site_key(caplog) -> None:
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)

    with caplog.at_level(logging.INFO, logger="nemo_gym.resources_servers.webvoyager_browser.captcha"):
        with pytest.raises(RuntimeError, match="no supported site key"):
            solver.maybe_solve(_ChallengePage(), phase="initial")

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=captcha_unresolved" in messages
    assert "reason=site_key_missing" in messages
    assert "CAP-private-key" not in messages


def test_capsolver_ignores_hidden_background_captcha_frame(caplog) -> None:
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)

    with caplog.at_level(logging.DEBUG, logger="nemo_gym.resources_servers.webvoyager_browser.captcha"):
        assert solver.maybe_solve(_BackgroundCaptchaFramePage(visible=False), phase="after navigate") is False

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=captcha_scan" in messages
    assert "challenge=none" in messages
    assert "event=captcha_unresolved" not in messages


def test_capsolver_treats_visible_captcha_frame_as_blocking(caplog) -> None:
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)

    with caplog.at_level(logging.INFO, logger="nemo_gym.resources_servers.webvoyager_browser.captcha"):
        with pytest.raises(RuntimeError, match="no supported site key"):
            solver.maybe_solve(_BackgroundCaptchaFramePage(visible=True), phase="after navigate")

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=captcha_unresolved" in messages
    assert "signal=visible_frame:" in messages
