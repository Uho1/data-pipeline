from __future__ import annotations

import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests

from market_data.utils import retry_call


_BASE_URL = "https://opendart.fss.or.kr/api"

# Environment variable names for API key rotation (checked in order)
_ENV_KEY_NAMES = (
    "DART_API_KEY_1",
    "DART_API_KEY_2",
    "DART_API_KEY_3",
    "DART_API_KEY_4",
    "DART_API_KEY_5",
    "DART_API_KEY_6",
    "DART_API_KEY_7",
)

_PREFERRED_KEY_ENV = "DART_API_KEY_PREFERRED"

# macOS Keychain service names (legacy, for backward compatibility)
_KEYCHAIN_SERVICES = (
    "market-data-dart-api",
    "market-data-dart-api-2",
    "market-data-dart-api-3",
)


def _load_dotenv_if_available() -> None:
    """Load .env file from project root if python-dotenv is available."""
    root = Path(__file__).resolve().parents[3]  # src/market_data/kr_dart -> project root
    env_file = root / ".env"
    try:
        from dotenv import load_dotenv
        if env_file.exists():
            load_dotenv(env_file, override=False)
    except ImportError:
        if not env_file.exists():
            return
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


def _load_all_api_keys() -> list[str]:
    """Return all non-empty DART API keys from env vars, .env file, or macOS Keychain.

    Priority order:
    1. Preferred env key declared via DART_API_KEY_PREFERRED (optional)
    2. Environment variables: DART_API_KEY_1 .. DART_API_KEY_7
    3. Legacy env var: dart_api
    4. macOS Keychain (only on macOS)
    """
    # Ensure .env is loaded
    _load_dotenv_if_available()

    keys: list[str] = []

    # 1. Check numbered env vars (from .env or shell environment)
    for env_name in _ENV_KEY_NAMES:
        key = os.getenv(env_name, "").strip()
        if key and key not in keys and not key.startswith("여기에"):
            keys.append(key)

    preferred_name = os.getenv(_PREFERRED_KEY_ENV, "").strip()
    if preferred_name:
        preferred_value = os.getenv(preferred_name, "").strip()
        if preferred_value and preferred_value in keys:
            keys = [preferred_value] + [value for value in keys if value != preferred_value]
        elif preferred_value and preferred_value not in keys and not preferred_value.startswith("여기에"):
            keys = [preferred_value, *keys]

    # 2. Check legacy single-key env var
    legacy_key = os.getenv("dart_api", "").strip()
    if legacy_key and legacy_key not in keys and not legacy_key.startswith("여기에"):
        keys.append(legacy_key)

    # 3. Fallback: macOS Keychain (backward compatibility)
    if not keys and sys.platform == "darwin":
        for service in _KEYCHAIN_SERVICES:
            try:
                result = subprocess.run(
                    ["security", "find-generic-password", "-s", service, "-w"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    key = result.stdout.strip()
                    if key and key not in keys:
                        keys.append(key)
            except Exception:
                continue

    return keys


def _load_all_api_keys_from_keychain() -> list[str]:
    """Legacy alias — now delegates to _load_all_api_keys()."""
    return _load_all_api_keys()


def _load_api_key_from_keychain() -> str:
    """Return the first non-empty DART API key."""
    keys = _load_all_api_keys()
    return keys[0] if keys else ""


class DartRateLimitError(RuntimeError):
    """Raised when Open DART reports quota exhaustion."""


class DartClient:
    """Thin Open DART API wrapper with automatic key rotation on rate limit."""

    def __init__(self, api_key: str | None = None, *, timeout: int = 30) -> None:
        explicit_key = str(api_key or "").strip()
        if explicit_key and not explicit_key.startswith("여기에"):
            self._keys: list[str] = [explicit_key]
        else:
            self._keys = _load_all_api_keys()
        if not self._keys:
            raise RuntimeError(
                "DART API key not found. Set it in .env file:\n"
                "  DART_API_KEY_1=your_key_here\n"
                "Or via environment variable: export dart_api='<YOUR_KEY>'"
            )
        self._key_idx: int = 0
        self.timeout = int(timeout)

    @property
    def api_key(self) -> str:
        return self._keys[self._key_idx]

    def _rotate_key(self) -> bool:
        """Advance to the next key. Returns True if a new key is available."""
        if self._key_idx + 1 < len(self._keys):
            self._key_idx += 1
            return True
        return False

    def _request(self, path: str, *, params: dict[str, object] | None = None) -> requests.Response:
        payload = {"crtfc_key": self.api_key}
        if params:
            payload.update(params)
        return retry_call(
            lambda: requests.get(f"{_BASE_URL}/{path}", params=payload, timeout=self.timeout),
            retries=3,
            backoff_base=1.0,
            label=f"dart:{path}",
        )

    def _get_json(self, path: str, *, params: dict[str, object] | None = None) -> dict[str, object]:
        response = self._request(path, params=params)
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("status", "000"))
        if status == "020":
            if self._rotate_key():
                return self._get_json(path, params=params)
            raise DartRateLimitError(
                f"DART API rate limit (all keys exhausted) path={path}"
            )
        if status not in {"000", ""}:
            raise RuntimeError(
                f"DART API error path={path} status={payload.get('status')} message={payload.get('message')}"
            )
        return payload

    def get_corp_codes(self) -> list[dict[str, str]]:
        response = self._request("corpCode.xml")
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            xml_name = next((name for name in archive.namelist() if name.lower().endswith(".xml")), None)
            if xml_name is None:
                raise RuntimeError("DART corpCode archive did not contain an XML payload")
            raw = archive.read(xml_name).decode("utf-8")

        import xml.etree.ElementTree as ET

        root = ET.fromstring(raw)
        rows: list[dict[str, str]] = []
        for item in root.findall("list"):
            row = {child.tag: str(child.text or "").strip() for child in item}
            if row:
                rows.append(row)
        return rows

    def company(self, corp_code: str) -> dict[str, object]:
        return self._get_json("company.json", params={"corp_code": corp_code})

    def list_filings(
        self,
        *,
        corp_code: str | None = None,
        bgn_de: str,
        end_de: str | None = None,
        page_no: int = 1,
        page_count: int = 100,
    ) -> dict[str, object]:
        params: dict[str, object] = {
            "bgn_de": bgn_de,
            "page_no": page_no,
            "page_count": page_count,
        }
        if corp_code:
            params["corp_code"] = corp_code
        if end_de:
            params["end_de"] = end_de
        return self._get_json("list.json", params=params)

    def financials_single_account(
        self,
        *,
        corp_code: str,
        bsns_year: int,
        reprt_code: str,
        fs_div: str = "CFS",
    ) -> dict[str, object]:
        return self._get_json(
            "fnlttSinglAcnt.json",
            params={
                "corp_code": corp_code,
                "bsns_year": int(bsns_year),
                "reprt_code": reprt_code,
                "fs_div": fs_div,
            },
        )

    def financials_all_accounts(
        self,
        *,
        corp_code: str,
        bsns_year: int,
        reprt_code: str,
        fs_div: str = "CFS",
    ) -> dict[str, object]:
        return self._get_json(
            "fnlttSinglAcntAll.json",
            params={
                "corp_code": corp_code,
                "bsns_year": int(bsns_year),
                "reprt_code": reprt_code,
                "fs_div": fs_div,
            },
        )

    def financials_xbrl(self, *, rcept_no: str, reprt_code: str) -> bytes:
        response = self._request(
            "fnlttXbrl.xml",
            params={
                "rcept_no": str(rcept_no).strip(),
                "reprt_code": str(reprt_code).strip(),
            },
        )
        response.raise_for_status()
        content = response.content
        # Detect rate-limit XML response (not a ZIP)
        if content.lstrip()[:5] == b"<?xml" and b"<status>020</status>" in content:
            if self._rotate_key():
                return self.financials_xbrl(rcept_no=rcept_no, reprt_code=reprt_code)
            raise DartRateLimitError(
                f"DART XBRL rate limit (all keys exhausted) rcept_no={rcept_no}"
            )
        return content


def dumps_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
