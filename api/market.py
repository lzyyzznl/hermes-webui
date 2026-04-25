"""
Hermes Web UI -- Skills Market backend module.
Handles search, install, uninstall, check-updates, and upgrade
for skills from the ZTE internal Skills Market.
"""

import json
import logging
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_version(version_str):
    """Parse a version string into a comparable tuple.

    Supports formats like:
    - 1.2.3
    - 1.2.3-alpha
    - v1.2.3
    - 1.2
    - 1

    Returns a tuple of ints/strings for comparison.
    """
    if not version_str:
        return (0,)

    # Remove leading 'v' or 'V'
    version_str = version_str.lstrip('vV')

    # Split by '-' first (for pre-release like 1.2.3-alpha)
    if '-' in version_str:
        main_part, pre_release = version_str.split('-', 1)
    else:
        main_part, pre_release = version_str, None

    # Split main part by '.'
    parts = []
    for part in main_part.split('.'):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(part)

    # Normalize length: pad with 0s so (1,2) and (1,2,1) can be compared
    # Use 4 parts to handle versions like 1.0.0.1
    while len(parts) < 4:
        parts.append(0)

    # Add pre-release info (pre-release versions are lower than release)
    if pre_release:
        parts.append(pre_release)
    else:
        # Release versions are higher than pre-release
        # Use '~' which sorts after all letters in ASCII
        parts.append('~')

    return tuple(parts)


def _version_lt(local_ver, remote_ver):
    """Return True if local version is strictly less than remote version."""
    if not local_ver and not remote_ver:
        return False
    if not local_ver:
        return True
    if not remote_ver:
        return False

    local_parsed = _parse_version(local_ver)
    remote_parsed = _parse_version(remote_ver)

    # Compare element by element to handle different lengths
    for i in range(max(len(local_parsed), len(remote_parsed))):
        local_part = local_parsed[i] if i < len(local_parsed) else 0
        remote_part = remote_parsed[i] if i < len(remote_parsed) else 0

        # Handle mixed types (int vs str) by converting int to str for comparison
        if isinstance(local_part, int) and isinstance(remote_part, int):
            if local_part < remote_part:
                return True
            elif local_part > remote_part:
                return False
        else:
            # Convert both to strings for comparison
            # Pre-release (non-empty string) should be less than release (empty string)
            local_str = str(local_part)
            remote_str = str(remote_part)
            if local_str < remote_str:
                return True
            elif local_str > remote_str:
                return False

    return False


# ── Constants ──────────────────────────────────────────────────────────────────

_MARKET_BASE = "https://market.zte.com.cn/zte-paas-market-bff/zte-paas-market-api"
_MARKET_LIST_URL = f"{_MARKET_BASE}/asset/browsing/getAssetListByCategoryCode"
_MARKET_DOWNLOAD_URL = f"{_MARKET_BASE}/asset/skill/download"

_SKILLS_ROOT = Path.home() / ".hermes" / "skills" / "co-claw"
_METADATA_PATH = _SKILLS_ROOT / ".market_installed.json"

_CLI_UAC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cli-uac")

_CACHE_TTL = 300  # 5 minutes

# ── Module-level state ────────────────────────────────────────────────────────

_uac_cache = {"empno": "", "token": "", "name": "", "ts": 0.0}
_market_lock = threading.Lock()


# ── SSL context (shared) ─────────────────────────────────────────────────────

def _ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ── UAC Authentication ────────────────────────────────────────────────────────

def _get_uac():
    """Return (empno, token, name) from cli-uac, with in-memory 5-min cache."""
    now = time.time()
    if _uac_cache["token"] and (now - _uac_cache["ts"]) < _CACHE_TTL:
        return _uac_cache["empno"], _uac_cache["token"], _uac_cache["name"]

    if not os.path.isfile(_CLI_UAC):
        raise RuntimeError("cli-uac binary not found")

    try:
        result = subprocess.run(
            [_CLI_UAC],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("cli-uac timed out")

    if result.returncode != 0:
        raise RuntimeError(f"cli-uac failed: {result.stderr.strip() or 'unknown error'}")

    try:
        data = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"cli-uac returned invalid JSON: {exc}")

    empno = data.get("coclaw_empno", "").strip()
    token = data.get("coclaw_token", "").strip()
    name = data.get("coclaw_name", "").strip()

    if not empno or not token:
        raise RuntimeError("cli-uac returned incomplete credentials")

    _uac_cache["empno"] = empno
    _uac_cache["token"] = token
    _uac_cache["name"] = name
    _uac_cache["ts"] = now

    return empno, token, name


# ── Market API request helper ─────────────────────────────────────────────────

def _market_request(url, data=None, method="GET", timeout=15):
    """Generic authenticated request to the Market API."""
    empno, token, _name = _get_uac()
    cookie = f"iAuthTid_prod={token}; iAuthUid_prod={empno}; UCSSSOLanguage=zh_CN"

    headers = {
        "Cookie": cookie,
        "Content-Type": "application/json; charset=utf-8",
    }

    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Metadata management ──────────────────────────────────────────────────────

def _load_metadata():
    """Load installed-skills metadata. Returns dict keyed by skill name."""
    if not _METADATA_PATH.exists():
        return {}
    try:
        text = _METADATA_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    # Corrupt or wrong type -- rebuild
    return {}


def _save_metadata(data):
    """Atomic write of metadata dict to disk."""
    _SKILLS_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(_SKILLS_ROOT), suffix=".tmp", prefix=".market_meta_"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(_METADATA_PATH))
    except BaseException:
        # Clean up tmp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── SKILL.md parsing ─────────────────────────────────────────────────────────

def _parse_skill_md(content, fallback_name):
    """Extract name and version from YAML frontmatter (between --- markers)."""
    name = fallback_name
    version = ""

    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            frontmatter = content[3:end]
            for line in frontmatter.splitlines():
                line = line.strip()
                if line.startswith("name:"):
                    val = line.split(":", 1)[1].strip().strip("\"'")
                    if val:
                        name = val
                elif line.startswith("version:"):
                    val = line.split(":", 1)[1].strip().strip("\"'")
                    if val:
                        version = val

    return name, version


# ── Internal helpers ──────────────────────────────────────────────────────────

def _download_skill(skill_id):
    """Download a skill ZIP from the market. Returns raw bytes."""
    download_url = f"{_MARKET_DOWNLOAD_URL}/{skill_id}"
    empno, token, _name = _get_uac()
    cookie = f"iAuthTid_prod={token}; iAuthUid_prod={empno}; UCSSSOLanguage=zh_CN"
    headers = {
        "Cookie": cookie,
        "Content-Type": "application/json; charset=utf-8",
    }
    req = urllib.request.Request(download_url, headers=headers, method="GET")

    with urllib.request.urlopen(req, timeout=30, context=_ssl_ctx()) as resp:
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read()

    if not raw:
        raise RuntimeError("Empty download response")

    # Binary response (ZIP file) — return directly
    if "application/json" not in content_type:
        return raw

    # JSON wrapper with download URL in `bo`
    resp_json = json.loads(raw.decode("utf-8"))
    if isinstance(resp_json, dict) and "bo" in resp_json:
        file_url = resp_json["bo"]
        if isinstance(file_url, str) and file_url.startswith("http"):
            req2 = urllib.request.Request(file_url, headers={"Cookie": cookie}, method="GET")
            with urllib.request.urlopen(req2, timeout=30, context=_ssl_ctx()) as r:
                data = r.read()
            if not data:
                raise RuntimeError("Empty download data")
            return data

    raise RuntimeError("Unexpected download response format")


def _extract_and_install(zip_bytes, fallback_name):
    """Extract ZIP, parse SKILL.md, move to skills dir. Returns (name, version)."""
    tmp_dir = tempfile.mkdtemp(prefix="hermes_market_")
    try:
        zip_path = os.path.join(tmp_dir, "skill.zip")
        with open(zip_path, "wb") as zf:
            zf.write(zip_bytes)

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Guard against zip slip (path traversal in archive members)
            abs_tmp = os.path.abspath(tmp_dir)
            for member in zf.infolist():
                member_path = os.path.abspath(os.path.join(tmp_dir, member.filename))
                if not member_path.startswith(abs_tmp + os.sep) and member_path != abs_tmp:
                    raise RuntimeError(f"Unsafe path in archive: {member.filename}")
            zf.extractall(tmp_dir)

        # Find SKILL.md
        skill_md_path = None
        for root, _dirs, files in os.walk(tmp_dir):
            if "SKILL.md" in files:
                skill_md_path = os.path.join(root, "SKILL.md")
                break

        if not skill_md_path:
            raise RuntimeError("SKILL.md not found in downloaded archive")

        content = Path(skill_md_path).read_text(encoding="utf-8")
        parsed_name, parsed_version = _parse_skill_md(content, fallback_name)

        # Sanitize: reject names with path traversal or unsafe characters
        if not re.match(r'^[a-zA-Z0-9_][a-zA-Z0-9_.\-/]*$', parsed_name):
            parsed_name = fallback_name

        dest_dir = _SKILLS_ROOT / parsed_name
        if dest_dir.exists():
            shutil.rmtree(str(dest_dir))

        src_dir = os.path.dirname(skill_md_path)
        shutil.move(src_dir, str(dest_dir))

        return parsed_name, parsed_version
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Exported functions (called from routes.py) ────────────────────────────────

def market_installed():
    """Return installed market skills from metadata (fast, no filesystem scan)."""
    meta = _load_metadata()
    return {"installed": [
        {"name": name, "assetId": info.get("assetId", ""),
         "skillId": info.get("skillId", ""), "version": info.get("version", "")}
        for name, info in meta.items()
    ]}


def market_search(query="", page=1, rows=16):
    """Search the Skills Market. Returns parsed API response."""
    payload = {
        "categoryCode": "coclaw_skill",
        "pageNo": page,
        "pageSize": rows,
        "bo": {
            "queryInfo": query,
        },
    }
    return _market_request(_MARKET_LIST_URL, data=payload, method="POST")


def market_install(asset_id, skill_id, name):
    """Download and install a skill from the Market."""
    with _market_lock:
        zip_bytes = _download_skill(skill_id)
        parsed_name, parsed_version = _extract_and_install(zip_bytes, name)

        meta = _load_metadata()
        meta[parsed_name] = {
            "assetId": asset_id,
            "skillId": skill_id,
            "version": parsed_version,
            "installedAt": time.time(),
        }
        _save_metadata(meta)

        return {"ok": True, "name": parsed_name, "version": parsed_version}


def market_uninstall(name):
    """Uninstall a market-installed skill by name."""
    with _market_lock:
        meta = _load_metadata()
        if name not in meta:
            raise ValueError(f"Skill '{name}' is not installed from Market")

        skill_dir = _SKILLS_ROOT / name
        if skill_dir.exists():
            shutil.rmtree(str(skill_dir))

        del meta[name]
        _save_metadata(meta)

        return {"ok": True, "name": name}


def market_check_updates():
    """Check installed skills for available updates. Returns dict with 'updates' list."""
    meta = _load_metadata()
    if not meta:
        logger.debug("market_check_updates: no metadata, returning empty")
        return {"updates": []}

    asset_ids = [info["assetId"] for info in meta.values() if info.get("assetId")]
    if not asset_ids:
        logger.debug("market_check_updates: no asset_ids in metadata, returning empty")
        return {"updates": []}

    payload = {
        "categoryCode": "coclaw_skill",
        "page": 1,
        "rows": 100,
        "bo": {
            "assetIds": asset_ids,
        },
    }
    logger.info("market_check_updates: payload=%s", payload)
    try:
        resp = _market_request(_MARKET_LIST_URL, data=payload, method="POST")
    except Exception as exc:
        logger.warning("market_check_updates request failed: %s", exc)
        return {"updates": []}

    logger.info("market_check_updates: raw resp=%s", resp)

    remote_map = {}
    rows_list = []
    if isinstance(resp, dict):
        bo = resp.get("bo") or {}
        rows_list = bo.get("rows") or bo.get("list") or []
    logger.info("market_check_updates: rows_list len=%d", len(rows_list))
    for item in rows_list:
        aid = str(item.get("assetId", ""))
        ver = item.get("version", "")
        logger.info("market_check_updates: row assetId=%s raw_item_keys=%s", aid, list(item.keys()))
        if aid:
            remote_map[aid] = ver

    logger.info("market_check_updates: remote_map=%s", remote_map)

    updates = []
    for skill_name, info in meta.items():
        aid = info.get("assetId", "")
        local_ver = info.get("version", "")
        remote_ver = remote_map.get(aid, "")
        needs_update = remote_ver and _version_lt(local_ver, remote_ver)
        logger.info(
            "market_check_updates: skill=%s aid=%s local=%s remote=%s needs_update=%s",
            skill_name, aid, local_ver, remote_ver, needs_update,
        )
        # Only report update if remote version is strictly newer than local
        if needs_update:
            updates.append({
                "name": skill_name,
                "localVersion": local_ver,
                "remoteVersion": remote_ver,
                "assetId": aid,
                "skillId": info.get("skillId", ""),
            })

    logger.info("market_check_updates: final updates=%s", updates)
    return {"updates": updates}


def market_upgrade(name, skill_id):
    """Upgrade a market-installed skill: download new version first, then replace."""
    with _market_lock:
        meta = _load_metadata()
        if name not in meta:
            raise ValueError(f"Skill '{name}' is not installed from Market")

        old_info = meta[name]
        old_asset_id = old_info.get("assetId", "")
        old_version = old_info.get("version", "")

        try:
            zip_bytes = _download_skill(skill_id)
            parsed_name, parsed_version = _extract_and_install(zip_bytes, name)
        except Exception as exc:
            logger.error("Failed to download skill '%s' (skill_id=%s): %s", name, skill_id, exc)
            raise RuntimeError(f"Download failed: {exc}") from exc

        # Validate: new version should be strictly newer than old version
        if old_version and parsed_version and not _version_lt(old_version, parsed_version):
            logger.warning(
                "Upgrade rejected for '%s': new version %s is not newer than current %s",
                name, parsed_version, old_version
            )
            # Rollback: remove the newly extracted skill and restore old one if needed
            dest_dir = _SKILLS_ROOT / parsed_name
            if dest_dir.exists():
                shutil.rmtree(str(dest_dir))
            raise ValueError(
                f"Upgrade rejected: new version {parsed_version} is not newer than current {old_version}"
            )

        meta = _load_metadata()
        meta[parsed_name] = {
            "assetId": old_asset_id or skill_id,
            "skillId": skill_id,
            "version": parsed_version,
            "installedAt": time.time(),
        }
        _save_metadata(meta)

        return {"ok": True, "name": parsed_name, "version": parsed_version}


def market_get_by_asset_ids(asset_ids):
    """Fetch market data for specific asset IDs. Returns dict keyed by assetId."""
    if not asset_ids:
        return {}

    payload = {
        "categoryCode": "coclaw_skill",
        "page": 1,
        "rows": 100,
        "bo": {
            "assetIds": asset_ids,
        },
    }
    try:
        resp = _market_request(_MARKET_LIST_URL, data=payload, method="POST")
    except Exception as exc:
        logger.warning("market_get_by_asset_ids request failed: %s", exc)
        return {}

    result = {}
    rows_list = []
    if isinstance(resp, dict):
        bo = resp.get("bo") or {}
        rows_list = bo.get("rows") or bo.get("list") or []
    for item in rows_list:
        aid = str(item.get("assetId", ""))
        if aid:
            result[aid] = item

    return result
