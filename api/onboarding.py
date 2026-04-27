"""Hermes Web UI -- first-run onboarding helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from api.auth import is_auth_enabled
from api.config import (
    DEFAULT_MODEL,
    DEFAULT_WORKSPACE,
    SESSION_DIR,
    STATE_DIR,
    _FALLBACK_MODELS,
    _HERMES_FOUND,
    _PROVIDER_DISPLAY,
    _PROVIDER_MODELS,
    _get_config_path,
    get_available_models,
    get_config,
    load_settings,
    reload_config,
    save_settings,
    verify_hermes_imports,
)
from api.workspace import get_last_workspace, load_workspaces, save_workspaces

logger = logging.getLogger(__name__)


_SUPPORTED_PROVIDER_SETUPS = {
    "openrouter": {
        "label": "OpenRouter",
        "env_var": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-sonnet-4.6",
        "requires_base_url": False,
        "models": [
            {"id": model["id"], "label": model["label"]} for model in _FALLBACK_MODELS
        ],
    },
    "anthropic": {
        "label": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4.6",
        "requires_base_url": False,
        "models": list(_PROVIDER_MODELS.get("anthropic", [])),
    },
    "openai": {
        "label": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "default_base_url": "https://api.openai.com/v1",
        "requires_base_url": False,
        "models": list(_PROVIDER_MODELS.get("openai", [])),
    },
    "custom": {
        "label": "Custom OpenAI-compatible",
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "requires_base_url": True,
        "models": [],
    },
    "zte": {
        "label": "ZTE MaaS (Qwen3-235B)",
        "env_var": "OPENAI_API_KEY",
        "default_model": "Qwen3-235B-A22B",
        "default_base_url": "https://maas-apigateway.dt.zte.com.cn/model-cop/qwen3-235b-a22b-instrust-2507-coclaw/v1",
        "requires_base_url": True,
        "models": [],
    },
}

_UNSUPPORTED_PROVIDER_NOTE = (
    "OAuth and advanced provider flows such as Nous Portal, OpenAI Codex, and GitHub "
    "Copilot are still terminal-first. Use `hermes model` for those flows."
)


def _get_active_hermes_home() -> Path:
    try:
        from api.profiles import get_active_hermes_home

        return get_active_hermes_home()
    except ImportError:
        return Path.home() / ".hermes"


def _load_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:
        return {}
    return values


def _write_env_file(env_path: Path, updates: dict[str, str]) -> None:
    current = _load_env_file(env_path)
    for key, value in updates.items():
        if value is None:
            current.pop(key, None)
            os.environ.pop(key, None)
            continue
        clean = str(value).strip()
        if not clean:
            continue
        # Reject embedded newlines/carriage returns to prevent .env injection
        if "\n" in clean or "\r" in clean:
            raise ValueError("API key must not contain newline characters.")
        current[key] = clean
        os.environ[key] = clean

    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={current[key]}" for key in sorted(current)]
    env_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _load_yaml_config(config_path: Path) -> dict:
    try:
        import yaml as _yaml
    except ImportError:
        return {}

    if not config_path.exists():
        return {}
    try:
        loaded = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _get_skeleton_config_path() -> Path | None:
    """Find skeleton config.yaml shipped with the installation."""
    candidates = [
        Path("/usr/share/hermes-webui/.hermes-skel/.hermes/config.yaml"),
        Path(__file__).resolve().parent.parent.parent / "packaging" / "skel" / ".hermes" / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values take precedence."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_config_with_defaults(config_path: Path) -> dict:
    """Load user config, merging skeleton defaults for any missing keys."""
    cfg = _load_yaml_config(config_path)
    skel_path = _get_skeleton_config_path()
    if skel_path:
        skel = _load_yaml_config(skel_path)
        if skel:
            cfg = _deep_merge(skel, cfg)
    return cfg


def _save_yaml_config(config_path: Path, config: dict) -> None:
    try:
        import yaml as _yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write Hermes config.yaml") from exc

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        _yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _normalize_model_for_provider(provider: str, model: str) -> str:
    clean = (model or "").strip()
    if not clean:
        return ""
    if provider in {"anthropic", "openai"} and clean.startswith(provider + "/"):
        return clean.split("/", 1)[1]
    return clean


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _extract_current_provider(cfg: dict) -> str:
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        provider = str(model_cfg.get("provider") or "").strip().lower()
        if provider:
            return provider
    return ""


def _extract_current_model(cfg: dict) -> str:
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg.strip()
    if isinstance(model_cfg, dict):
        return str(model_cfg.get("default") or "").strip()
    return ""


def _extract_current_base_url(cfg: dict) -> str:
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        return _normalize_base_url(str(model_cfg.get("base_url") or ""))
    return ""


def _provider_api_key_present(
    provider: str, cfg: dict, env_values: dict[str, str]
) -> bool:
    provider = (provider or "").strip().lower()
    if not provider:
        return False

    env_var = _SUPPORTED_PROVIDER_SETUPS.get(provider, {}).get("env_var")
    if env_var and env_values.get(env_var):
        return True

    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict) and str(model_cfg.get("api_key") or "").strip():
        return True

    providers_cfg = cfg.get("providers", {})
    if isinstance(providers_cfg, dict):
        provider_cfg = providers_cfg.get(provider, {})
        if (
            isinstance(provider_cfg, dict)
            and str(provider_cfg.get("api_key") or "").strip()
        ):
            return True
        if provider == "custom":
            custom_cfg = providers_cfg.get("custom", {})
            if (
                isinstance(custom_cfg, dict)
                and str(custom_cfg.get("api_key") or "").strip()
            ):
                return True

    # For providers not in _SUPPORTED_PROVIDER_SETUPS (e.g. minimax-cn, deepseek,
    # xai, etc.), ask the hermes_cli auth registry — it knows every provider's env
    # var names and can check os.environ for a valid key.
    # Exclude known OAuth/token-flow providers — those are handled separately by
    # _provider_oauth_authenticated() and should not be short-circuited here.
    _known_oauth = {"openai-codex", "copilot", "copilot-acp", "qwen-oauth", "nous"}
    if provider not in _SUPPORTED_PROVIDER_SETUPS and provider not in _known_oauth:
        try:
            from hermes_cli.auth import get_auth_status as _gas
            status = _gas(provider)
            if isinstance(status, dict) and status.get("logged_in"):
                return True
        except Exception:
            pass

    return False



def _oauth_payload_has_token(payload: dict) -> bool:
    """Return True if an auth payload contains usable token material."""
    if not isinstance(payload, dict):
        return False

    token_fields = (
        payload,
        payload.get("tokens") if isinstance(payload.get("tokens"), dict) else {},
    )
    for candidate in token_fields:
        if not isinstance(candidate, dict):
            continue
        if any(
            str(candidate.get(key) or "").strip()
            for key in ("access_token", "refresh_token", "api_key")
        ):
            return True
    return False



def _provider_oauth_authenticated(provider: str, hermes_home: "Path") -> bool:
    """Return True if the provider has valid OAuth credentials.

    Reads the profile-scoped auth.json directly so onboarding respects the
    requested Hermes home. Known OAuth providers may store auth either in the
    legacy providers[provider_id] singleton state or in credential_pool entries
    used by current Hermes runtime auth resolution.
    """
    provider = (provider or "").strip().lower()
    if not provider:
        return False

    _known_oauth_providers = {"openai-codex", "copilot", "copilot-acp", "qwen-oauth", "nous"}
    if provider not in _known_oauth_providers:
        return False

    try:
        import json as _j

        auth_path = hermes_home / "auth.json"
        if not auth_path.exists():
            return False
        store = _j.loads(auth_path.read_text(encoding="utf-8"))

        providers_store = store.get("providers")
        if isinstance(providers_store, dict):
            state = providers_store.get(provider)
            if _oauth_payload_has_token(state):
                return True

        pool_store = store.get("credential_pool")
        if isinstance(pool_store, dict):
            entries = pool_store.get(provider)
            if isinstance(entries, list):
                return any(_oauth_payload_has_token(entry) for entry in entries)

        return False
    except Exception:
        return False


def _get_openclaw_config() -> dict:
    """Load .openclaw configuration if it exists."""
    openclaw_path = Path.home() / ".openclaw" / "openclaw.json"
    return _load_yaml_config(openclaw_path)


def _openclaw_has_skills() -> bool:
    """Check if .openclaw has any skills configured."""
    cfg = _get_openclaw_config()
    skills = cfg.get("skills", {})
    entries = skills.get("entries", {})
    return bool(entries)


def _openclaw_models_info() -> dict:
    """Extract model info from .openclaw configuration."""
    cfg = _get_openclaw_config()
    models_cfg = cfg.get("models", {})
    providers = models_cfg.get("providers", {})
    return {
        "has_openclaw": True,
        "providers": providers,
        "agents_default_model": cfg.get("agents", {}).get("defaults", {}).get("model", {}).get("primary", ""),
    }


def _get_openclaw_api_keys() -> dict[str, str]:
    """Load API keys from .openclaw.

    Checks two locations (in priority order):
    1. ~/.openclaw-dev/openclaw.json (dev config with embedded credentials)
    2. ~/.openclaw/agents/main/agent/auth-profiles.json (agent auth store)

    Returns a dict mapping provider name to API key.
    """
    keys: dict[str, str] = {}

    # Try ~/.openclaw-dev/openclaw.json first
    dev_config = _load_yaml_config(Path.home() / ".openclaw-dev" / "openclaw.json")
    if dev_config:
        providers = dev_config.get("providers", {})
        if isinstance(providers, dict):
            for provider_id, provider_data in providers.items():
                if isinstance(provider_data, dict):
                    api_key = provider_data.get("apiKey") or provider_data.get("api_key")
                    if api_key:
                        keys[provider_id] = api_key

    # Try ~/.openclaw/openclaw.json providers section
    if not keys:
        openclaw_cfg = _get_openclaw_config()
        if openclaw_cfg:
            providers = openclaw_cfg.get("models", {}).get("providers", {})
            if isinstance(providers, dict):
                for provider_id, provider_data in providers.items():
                    if isinstance(provider_data, dict):
                        api_key = provider_data.get("apiKey") or provider_data.get("api_key")
                        if api_key and provider_id not in keys:
                            keys[provider_id] = api_key

    # Fall back to ~/.openclaw/agents/main/agent/auth-profiles.json
    if not keys:
        import json as _j
        auth_path = Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
        if auth_path.exists():
            try:
                auth_data = _j.loads(auth_path.read_text(encoding="utf-8"))
                profiles = auth_data.get("profiles", {})
                if isinstance(profiles, dict):
                    for profile_key, profile_data in profiles.items():
                        if isinstance(profile_data, dict) and profile_data.get("type") == "api_key":
                            provider = profile_data.get("provider", "")
                            key = profile_data.get("key", "")
                            if provider and key:
                                keys[provider] = key
            except Exception:
                pass

    return keys


def _status_from_runtime(cfg: dict, imports_ok: bool) -> dict:
    provider = _extract_current_provider(cfg)
    model = _extract_current_model(cfg)
    base_url = _extract_current_base_url(cfg)
    env_values = _load_env_file(_get_active_hermes_home() / ".env")

    provider_configured = bool(provider and model)
    provider_ready = False

    if provider_configured:
        if provider == "custom":
            provider_ready = bool(
                base_url and _provider_api_key_present(provider, cfg, env_values)
            )
        elif provider in _SUPPORTED_PROVIDER_SETUPS:
            provider_ready = _provider_api_key_present(provider, cfg, env_values)
        else:
            # Unknown provider — may be an OAuth flow (openai-codex, copilot, etc.)
            # OR an API-key provider not in the quick-setup list (minimax-cn, deepseek,
            # xai, etc.).  Check both: api key presence first (covers the majority of
            # third-party providers), then OAuth auth.json.
            provider_ready = (
                _provider_api_key_present(provider, cfg, env_values)
                or _provider_oauth_authenticated(provider, _get_active_hermes_home())
            )

    chat_ready = bool(_HERMES_FOUND and imports_ok and provider_ready)

    if not _HERMES_FOUND or not imports_ok:
        state = "agent_unavailable"
        note = (
            "Hermes is not fully importable from the Web UI yet. Finish bootstrap or fix the "
            "agent install before provider setup will work."
        )
    elif chat_ready:
        state = "ready"
        provider_name = _PROVIDER_DISPLAY.get(
            provider, provider.title() if provider else "Hermes"
        )
        note = f"Hermes is minimally configured and ready to chat via {provider_name}."
    elif provider_configured:
        state = "provider_incomplete"
        if provider == "custom" and not base_url:
            note = (
                "Hermes has a saved provider/model selection but still needs the "
                "base URL and API key required to chat."
            )
        elif provider not in _SUPPORTED_PROVIDER_SETUPS:
            # OAuth / unsupported provider: avoid misleading "API key" wording.
            note = (
                f"Provider '{provider}' is configured but not yet authenticated. "
                "Run 'hermes auth' or 'hermes model' in a terminal to complete "
                "setup, then reload the Web UI."
            )
        else:
            note = (
                "Hermes has a saved provider/model selection but still needs the "
                "API key required to chat."
            )
    else:
        state = "needs_provider"
        note = "Hermes is installed, but you still need to choose a provider and save working credentials."

    return {
        "provider_configured": provider_configured,
        "provider_ready": provider_ready,
        "chat_ready": chat_ready,
        "setup_state": state,
        "provider_note": note,
        "current_provider": provider or None,
        "current_model": model or None,
        "current_base_url": base_url or None,
        "env_path": str(_get_active_hermes_home() / ".env"),
    }


def _build_setup_catalog(cfg: dict) -> dict:
    current_provider = _extract_current_provider(cfg) or "zte"
    current_model = _extract_current_model(cfg)
    current_base_url = _extract_current_base_url(cfg)

    providers = []
    for provider_id, meta in _SUPPORTED_PROVIDER_SETUPS.items():
        providers.append(
            {
                "id": provider_id,
                "label": meta["label"],
                "env_var": meta["env_var"],
                "default_model": meta["default_model"],
                "default_base_url": meta.get("default_base_url") or "",
                "requires_base_url": bool(meta.get("requires_base_url")),
                "models": list(meta.get("models", [])),
                "quick": provider_id == "openrouter",
            }
        )

    # Flag whether the currently-configured provider is OAuth-based (not in the
    # API-key flow).  The frontend uses this to show a confirmation card instead
    # of a key input when the user has already authenticated via 'hermes auth'.
    current_is_oauth = current_provider not in _SUPPORTED_PROVIDER_SETUPS and bool(
        current_provider
    )

    return {
        "providers": providers,
        "unsupported_note": _UNSUPPORTED_PROVIDER_NOTE,
        "current_is_oauth": current_is_oauth,
        "current": {
            "provider": current_provider,
            "model": current_model
            or _SUPPORTED_PROVIDER_SETUPS.get(current_provider, {}).get(
                "default_model", ""
            ),
            "base_url": current_base_url,
        },
    }


def get_onboarding_status() -> dict:
    settings = load_settings()
    cfg = get_config()
    imports_ok, missing, errors = verify_hermes_imports()
    runtime = _status_from_runtime(cfg, imports_ok)
    workspaces = load_workspaces()
    last_workspace = get_last_workspace()
    available_models = get_available_models()

    # HERMES_WEBUI_SKIP_ONBOARDING=1 lets hosting providers (e.g. Agent37) ship
    # a pre-configured instance without the wizard blocking the first load.
    # This is an operator-level override and is honoured unconditionally —
    # the operator knows their deployment is configured; we must not second-guess
    # it by requiring chat_ready to also be true.
    skip_env = os.environ.get("HERMES_WEBUI_SKIP_ONBOARDING", "").strip()
    skip_requested = skip_env in {"1", "true", "yes"}
    auto_completed = skip_requested  # unconditional: operator says skip, we skip

    # Auto-complete for existing Hermes users: if config.yaml already exists
    # AND the system is chat_ready, treat onboarding as done.  These users
    # configured Hermes via the CLI before the Web UI existed; they must never
    # be shown the first-run wizard — it would silently overwrite their config.
    config_exists = Path(_get_config_path()).exists()
    config_auto_completed = config_exists and bool(runtime.get("chat_ready"))

    # Persist the flag so it survives future transient import failures (e.g. after
    # a git branch switch in the hermes-agent repo).  Without this, a CLI-configured
    # user who never ran the wizard has no onboarding_completed flag — any momentary
    # imports_ok=False during restart makes chat_ready=False, config_auto_completed=False,
    # and the wizard reappears with a broken dropdown that clobbers their config.
    #
    # Best-effort: if save_settings raises (read-only FS, disk full, permission error),
    # log and continue.  The `config_auto_completed` branch of `completed=` below still
    # returns True for this request, so the user sees the correct state — only the
    # persistence-across-restart guarantee is degraded.  Raising here would turn every
    # /api/onboarding/status call into a 500 until disk was writable, which is worse UX
    # than losing the next-restart protection.
    if config_auto_completed and not settings.get("onboarding_completed"):
        try:
            save_settings({"onboarding_completed": True})
            settings["onboarding_completed"] = True
        except Exception:
            logger.debug("Failed to persist onboarding_completed", exc_info=True)

    # Check .openclaw for existing configuration
    openclaw_cfg = _get_openclaw_config()
    has_openclaw = bool(openclaw_cfg)
    openclaw_has_skills = _openclaw_has_skills()
    openclaw_models = _openclaw_models_info() if has_openclaw else {}

    return {
        "completed": bool(settings.get("onboarding_completed")) or auto_completed or config_auto_completed,
        "settings": {
            "default_model": settings.get("default_model") or DEFAULT_MODEL,
            "default_workspace": settings.get("default_workspace")
            or str(DEFAULT_WORKSPACE),
            "password_enabled": is_auth_enabled(),
            "bot_name": settings.get("bot_name") or "Hermes",
        },
        "system": {
            "hermes_found": bool(_HERMES_FOUND),
            "imports_ok": bool(imports_ok),
            "missing_modules": missing,
            "import_errors": errors,
            "config_path": str(_get_config_path()),
            "config_exists": Path(_get_config_path()).exists(),
            **runtime,
        },
        "setup": _build_setup_catalog(cfg),
        "workspaces": {
            "items": workspaces,
            "last": last_workspace,
        },
        "models": available_models,
        "openclaw": {
            "exists": has_openclaw,
            "has_skills": openclaw_has_skills,
            "models_info": openclaw_models,
        },
    }


def apply_onboarding_setup(body: dict) -> dict:
    # Hard guard: if the operator set SKIP_ONBOARDING, the wizard should never
    # have appeared.  Even if the frontend somehow calls this endpoint anyway
    # (e.g. a stale JS bundle or a curious user), we must not overwrite the
    # operator's config.yaml or .env files.  Just mark onboarding complete and
    # return the current status — no file writes.
    skip_env = os.environ.get("HERMES_WEBUI_SKIP_ONBOARDING", "").strip()
    if skip_env in {"1", "true", "yes"}:
        save_settings({"onboarding_completed": True})
        return get_onboarding_status()

    provider = str(body.get("provider") or "").strip().lower()
    model = str(body.get("model") or "").strip()
    api_key = str(body.get("api_key") or "").strip()
    base_url = _normalize_base_url(str(body.get("base_url") or ""))

    if provider not in _SUPPORTED_PROVIDER_SETUPS:
        # Unsupported providers (openai-codex, copilot, nous, etc.) are already
        # configured via the CLI. Just mark onboarding as complete and let the
        # user through — the agent is already set up, no further setup needed.
        save_settings({"onboarding_completed": True})
        return get_onboarding_status()
    if not model:
        raise ValueError("model is required")

    provider_meta = _SUPPORTED_PROVIDER_SETUPS[provider]
    if provider_meta.get("requires_base_url"):
        if not base_url:
            raise ValueError("base_url is required for custom endpoints")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("base_url must start with http:// or https://")

    config_path = _get_config_path()
    # Guard: if config.yaml already exists and the caller did not explicitly
    # acknowledge the overwrite, refuse to proceed.  The frontend must pass
    # confirm_overwrite=True after showing the user a confirmation step.
    if Path(config_path).exists() and not body.get("confirm_overwrite"):
        return {
            "error": "config_exists",
            "message": (
                "Hermes is already configured (config.yaml exists). "
                "Pass confirm_overwrite=true to overwrite it."
            ),
            "requires_confirm": True,
        }

    cfg = _load_config_with_defaults(config_path)
    env_path = _get_active_hermes_home() / ".env"
    env_values = _load_env_file(env_path)

    if not api_key and not _provider_api_key_present(provider, cfg, env_values):
        raise ValueError(f"{provider_meta['env_var']} is required")

    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    model_cfg["provider"] = provider
    model_cfg["default"] = _normalize_model_for_provider(provider, model)

    if provider == "zte":
        model_cfg.pop("base_url", None)
        model_cfg["context_length"] = 128000
    elif provider == "custom":
        model_cfg["base_url"] = base_url
    elif provider == "openai":
        model_cfg["base_url"] = (
            provider_meta.get("default_base_url") or "https://api.openai.com/v1"
        )
    else:
        model_cfg.pop("base_url", None)

    cfg["model"] = model_cfg

    providers_cfg = cfg.setdefault("providers", {})
    if not isinstance(providers_cfg, dict):
        providers_cfg = {}
        cfg["providers"] = providers_cfg
    provider_cfg = providers_cfg.setdefault(provider, {})
    if not isinstance(provider_cfg, dict):
        provider_cfg = {}
        providers_cfg[provider] = provider_cfg

    if provider == "zte":
        provider_cfg["base_url"] = base_url
        provider_cfg["models"] = [
            {"id": model_cfg["default"], "label": model_cfg["default"]}
        ]

    if api_key:
        provider_cfg["api_key"] = api_key

    _save_yaml_config(config_path, cfg)

    if api_key:
        _write_env_file(env_path, {provider_meta["env_var"]: api_key})

    # Reload the hermes_cli provider/config cache so the next streaming call
    # picks up the new key without requiring a server restart.
    try:
        from api.profiles import _reload_dotenv
        _reload_dotenv(_get_active_hermes_home())
    except Exception:
        logger.debug("Failed to reload dotenv")

    # Belt-and-braces: set directly on os.environ AFTER _reload_dotenv so the
    # value survives even if _reload_dotenv cleared it (e.g. when _write_env_file
    # wrote to disk but the profile isolation tracking hasn't seen it yet).
    if api_key:
        os.environ[provider_meta["env_var"]] = api_key

    try:
        # hermes_cli may cache config at import time; ask it to reload if possible.
        from hermes_cli.config import reload as _cli_reload
        _cli_reload()
    except Exception:
        logger.debug("Failed to reload hermes_cli config")

    reload_config()
    return get_onboarding_status()


def complete_onboarding() -> dict:
    save_settings({"onboarding_completed": True})
    return get_onboarding_status()


def sync_from_openclaw() -> dict:
    """Sync model config and skills from .openclaw to hermes config.

    If .openclaw exists, copy the model configuration from it.
    If .openclaw has skills configured, those will be synced too.
    """
    openclaw_cfg = _get_openclaw_config()
    if not openclaw_cfg:
        raise RuntimeError(".openclaw not found, cannot sync")

    hermes_cfg = _load_config_with_defaults(_get_config_path())
    hermes_home = _get_active_hermes_home()

    # Sync model config from .openclaw/models to hermes config
    openclaw_models = openclaw_cfg.get("models", {})
    openclaw_providers = openclaw_models.get("providers", {})

    if openclaw_providers:
        # Convert .openclaw model format to hermes model format
        hermes_model_providers = {}
        for provider_id, provider_data in openclaw_providers.items():
            # Unify zte-maas / zte to a single "zte" provider ID
            hermes_provider_id = "zte" if provider_id in ("zte", "zte-maas") else provider_id
            base_url = provider_data.get("baseUrl", "")
            models_list = provider_data.get("models", [])
            hermes_models = []
            for m in models_list:
                hermes_models.append({
                    "id": m.get("id", ""),
                    "label": m.get("name", m.get("id", "")),
                })

            hermes_model_providers[hermes_provider_id] = {
                "base_url": base_url,
                "models": hermes_models,
            }

            # Set the first provider/model as default (always override during sync)
            default_model = models_list[0].get("id", "") if models_list else ""
            if default_model:
                hermes_cfg["model"] = {
                    "provider": hermes_provider_id,
                    "default": default_model,
                    "context_length": 128000,
                }

        hermes_cfg["providers"] = hermes_model_providers

    # Sync skills from .openclaw if they exist
    openclaw_skills = openclaw_cfg.get("skills", {})
    if openclaw_skills:
        hermes_skills = hermes_cfg.get("skills", {})
        if not hermes_skills:
            hermes_skills = {"external_dirs": [], "creation_nudge_interval": 15}
        entries = openclaw_skills.get("entries", {})
        if entries:
            hermes_skills["entries"] = entries
        hermes_cfg["skills"] = hermes_skills

    # Copy skill files from .openclaw/workspace/skills/ to ~/.hermes/skills/co-claw/
    openclaw_skills_dir = hermes_home.parent / ".openclaw" / "workspace" / "skills"
    hermes_skills_dir = hermes_home / "skills" / "co-claw"
    hermes_skills_dir.mkdir(parents=True, exist_ok=True)
    if openclaw_skills_dir.is_dir():
        import shutil
        for skill_entry in openclaw_skills_dir.iterdir():
            if skill_entry.is_dir():
                target = hermes_skills_dir / skill_entry.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(skill_entry, target)
        logger.info("Synced skills from %s to %s", openclaw_skills_dir, hermes_skills_dir)

    _save_yaml_config(_get_config_path(), hermes_cfg)

    # Sync API keys from .openclaw into providers.<id>.api_key
    openclaw_api_keys = _get_openclaw_api_keys()
    if openclaw_api_keys:
        model_cfg = hermes_cfg.get("model", {})
        if isinstance(model_cfg, dict):
            active_provider = model_cfg.get("provider", "")
            providers_cfg = hermes_cfg.get("providers", {})
            if isinstance(providers_cfg, dict):
                for provider, api_key in openclaw_api_keys.items():
                    # Unify zte-maas / zte for matching
                    _key = "zte" if provider in ("zte", "zte-maas") else provider
                    if _key == active_provider and _key in providers_cfg:
                        providers_cfg[_key]["api_key"] = api_key
                        break
        _save_yaml_config(_get_config_path(), hermes_cfg)

    # Create hermes-native workspace under ~/.hermes/workspace
    workspace_dir = hermes_home / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    workspace_path = str(workspace_dir.resolve())

    # Add to workspace list and set as default
    saved = load_workspaces()
    saved_paths = {w["path"] for w in saved}
    if workspace_path not in saved_paths:
        saved.insert(0, {"path": workspace_path, "name": "Home"})
        save_workspaces(saved)
    from api.workspace import set_last_workspace
    set_last_workspace(workspace_path)

    reload_config()
    # Invalidate the models cache so the next /api/models call returns fresh data
    # with the newly synced provider and models. Without this, the 60s TTL cache
    # would return stale data.
    from api.config import invalidate_models_cache
    invalidate_models_cache()
    # Ensure essential directories exist (normally created at server startup,
    # but needed here when ~/.hermes was freshly cleared)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    save_settings({"onboarding_completed": True, "default_workspace": workspace_path})
    return get_onboarding_status()


def _parse_skill_description(skill_md_path: Path) -> str:
    """Extract description from SKILL.md YAML frontmatter."""
    try:
        content = skill_md_path.read_text(encoding="utf-8", errors="replace")
        if content.startswith("---"):
            end = content.find("---", 3)
            if end > 0:
                for line in content[3:end].splitlines():
                    stripped = line.strip()
                    if stripped.startswith("description:"):
                        return stripped.split(":", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def list_openclaw_skills() -> dict:
    """List available skills in ~/.openclaw/workspace/skills/."""
    hermes_home = _get_active_hermes_home()
    openclaw_skills_dir = hermes_home.parent / ".openclaw" / "workspace" / "skills"

    if not openclaw_skills_dir.is_dir():
        return {"available": False, "skills": [], "reason": "no_openclaw_dir"}

    hermes_skills_dir = hermes_home / "skills" / "co-claw"
    entries = []
    for skill_entry in sorted(openclaw_skills_dir.iterdir()):
        if not skill_entry.is_dir() or skill_entry.name.startswith("."):
            continue

        description = _parse_skill_description(skill_entry / "SKILL.md")
        has_conflict = (hermes_skills_dir / skill_entry.name).exists()
        entries.append({
            "name": skill_entry.name,
            "description": description,
            "has_conflict": has_conflict,
        })

    if not entries:
        return {"available": False, "skills": [], "reason": "empty_dir"}

    return {"available": True, "skills": entries}


def migrate_skills_from_openclaw(selected_names: list[str]) -> dict:
    """Copy selected skill folders from ~/.openclaw/workspace/skills/ to ~/.hermes/skills/co-claw/."""
    import shutil

    hermes_home = _get_active_hermes_home()
    openclaw_skills_dir = hermes_home.parent / ".openclaw" / "workspace" / "skills"
    hermes_skills_dir = hermes_home / "skills" / "co-claw"

    if not openclaw_skills_dir.is_dir():
        raise RuntimeError("OpenClaw skills directory not found")

    hermes_skills_dir.mkdir(parents=True, exist_ok=True)

    migrated: list[str] = []
    overwritten: list[str] = []

    for name in selected_names:
        if "/" in name or ".." in name or not name:
            continue
        source = openclaw_skills_dir / name
        if not source.is_dir():
            continue

        target = hermes_skills_dir / name
        if target.exists():
            shutil.rmtree(target)
            overwritten.append(name)

        shutil.copytree(source, target)
        migrated.append(name)

    logger.info("Migrated %d skills from OpenClaw (overwritten: %s)", len(migrated), overwritten)

    return {
        "ok": True,
        "migrated": migrated,
        "overwritten": overwritten,
        "count": len(migrated),
    }
