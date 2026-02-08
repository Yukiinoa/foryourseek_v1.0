from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class HttpClient:
    user_agent: str
    timeout_sec: int = 20
    retries: int = 3

    _browser_headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }

    def __post_init__(self) -> None:
        self.session = requests.Session()
        retry = Retry(
            total=self.retries,
            read=self.retries,
            connect=self.retries,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD", "POST"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get(self, url: str, *, headers: Optional[dict] = None) -> requests.Response:
        h = {"User-Agent": self.user_agent}
        if headers:
            h.update(headers)
        return self.session.get(url, headers=h, timeout=self.timeout_sec)

    def get_text(self, url: str, *, headers: Optional[dict] = None) -> str:
        resp = self.get(url, headers=headers)
        resp.raise_for_status()
        resp.encoding = resp.encoding or "utf-8"
        return resp.text

    def post(
        self, url: str, *, data: Optional[dict] = None, headers: Optional[dict] = None
    ) -> requests.Response:
        h = {"User-Agent": self.user_agent}
        if headers:
            h.update(headers)
        return self.session.post(url, data=data or {}, headers=h, timeout=self.timeout_sec)

    def post_text(
        self, url: str, *, data: Optional[dict] = None, headers: Optional[dict] = None
    ) -> str:
        resp = self.post(url, data=data, headers=headers)
        resp.raise_for_status()
        resp.encoding = resp.encoding or "utf-8"
        return resp.text

    def get_text_smart(self, url: str, *, headers_list: Optional[list[dict]] = None) -> str:
        """Try with default headers first, then a browser-like header set."""
        tries = headers_list or [None, self._browser_headers]
        last_exc = None
        for h in tries:
            try:
                return self.get_text(url, headers=h)
            except Exception as ex:
                last_exc = ex
        if last_exc:
            raise last_exc
        raise RuntimeError("get_text_smart failed without exception")
