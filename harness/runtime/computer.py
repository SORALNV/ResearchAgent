from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse


@runtime_checkable
class ComputerUseDriver(Protocol):
    """Minimal driver used by the OpenAI computer-use loop."""

    def apply(self, action: dict[str, Any]) -> None:
        """Apply one model-requested action."""

    def screenshot_base64(self) -> str:
        """Return a PNG screenshot encoded as base64."""

    def current_url(self) -> str:
        """Return the current browser URL, if applicable."""

    def close(self) -> None:
        """Release browser and OS resources."""


@dataclass(frozen=True)
class ComputerUsePolicy:
    allowed_domains: tuple[str, ...] = ()
    max_actions: int = 80
    allow_downloads: bool = False
    allow_clipboard: bool = False
    allow_file_upload: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_url(self, url: str) -> None:
        if not self.allowed_domains or not url:
            return
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise PermissionError(f"computer-use URL has no hostname: {url}")
        allowed = any(
            hostname == domain.lower()
            or hostname.endswith("." + domain.lower())
            for domain in self.allowed_domains
        )
        if not allowed:
            raise PermissionError(f"computer-use domain is not allowed: {hostname}")


class PlaywrightComputerDriver:
    """Optional Chromium driver for approved computer-use sessions.

    Importing this module does not require Playwright. The dependency and browser
    are loaded only when a session is explicitly approved and started.
    """

    def __init__(
        self,
        *,
        start_url: str,
        width: int = 1280,
        height: int = 900,
        headless: bool = True,
        policy: ComputerUsePolicy | None = None,
    ) -> None:
        self.policy = policy or ComputerUsePolicy()
        self.policy.validate_url(start_url)
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional computer dependency and Chromium: "
                "pip install -e '.[computer]' && playwright install chromium"
            ) from exc
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=headless,
            downloads_path=None if not self.policy.allow_downloads else "downloads",
        )
        self._context = self._browser.new_context(
            viewport={"width": width, "height": height},
            accept_downloads=self.policy.allow_downloads,
        )
        self._page = self._context.new_page()
        self._page.goto(start_url, wait_until="domcontentloaded")
        self._validate_current_url()

    def apply(self, action: dict[str, Any]) -> None:
        action_type = str(action.get("type") or action.get("action") or "").lower()
        if action_type == "click":
            self._page.mouse.click(float(action["x"]), float(action["y"]))
        elif action_type in {"double_click", "doubleclick"}:
            self._page.mouse.dblclick(float(action["x"]), float(action["y"]))
        elif action_type in {"move", "move_mouse"}:
            self._page.mouse.move(float(action["x"]), float(action["y"]))
        elif action_type == "scroll":
            self._page.mouse.wheel(
                float(action.get("scroll_x") or action.get("delta_x") or 0),
                float(action.get("scroll_y") or action.get("delta_y") or 0),
            )
        elif action_type in {"type", "input_text"}:
            self._page.keyboard.type(str(action.get("text") or ""))
        elif action_type in {"keypress", "key"}:
            keys = action.get("keys") or action.get("key") or []
            if isinstance(keys, str):
                keys = [keys]
            for key in keys:
                self._page.keyboard.press(str(key))
        elif action_type == "drag":
            path = action.get("path") or []
            if not isinstance(path, list) or len(path) < 2:
                raise ValueError("drag action requires at least two path points")
            first = path[0]
            self._page.mouse.move(float(first["x"]), float(first["y"]))
            self._page.mouse.down()
            try:
                for point in path[1:]:
                    self._page.mouse.move(float(point["x"]), float(point["y"]))
            finally:
                self._page.mouse.up()
        elif action_type in {"wait", "sleep"}:
            time.sleep(min(10.0, max(0.0, float(action.get("seconds") or 1.0))))
        elif action_type in {"screenshot", "observe"}:
            pass
        else:
            raise ValueError(f"unsupported computer-use action: {action_type}")
        self._page.wait_for_timeout(150)
        self._validate_current_url()

    def screenshot_base64(self) -> str:
        image = self._page.screenshot(type="png", full_page=False)
        return base64.b64encode(image).decode("ascii")

    def current_url(self) -> str:
        return str(self._page.url)

    def close(self) -> None:
        for resource in (
            getattr(self, "_context", None),
            getattr(self, "_browser", None),
            getattr(self, "_playwright", None),
        ):
            if resource is None:
                continue
            try:
                if resource is self._playwright:
                    resource.stop()
                else:
                    resource.close()
            except Exception:
                pass

    def _validate_current_url(self) -> None:
        self.policy.validate_url(self.current_url())
