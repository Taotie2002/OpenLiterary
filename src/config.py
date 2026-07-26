import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Config:
    _instance: Optional['Config'] = None
    _data: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._data:
            self._load()

    def _load(self):
        root_dir = Path(__file__).resolve().parent.parent
        config_path = root_dir / "config.yaml"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        else:
            self._data = {}

        self._data = self._expand_env_vars(self._data)

        if not self._data:
            self._data = {}

        self._apply_env_overrides()

    def _apply_env_overrides(self):
        prefix = "OPENLITERARY_"
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            raw = env_key[len(prefix):].strip("_").lower()
            path = [p for p in raw.split("__") if p]
            if not path:
                continue
            self._set_nested(self._data, path, self._parse_env_value(env_val))

    @staticmethod
    def _parse_env_value(val: str) -> Any:
        if val.lower() in ("true", "yes", "1"):
            return True
        if val.lower() in ("false", "no", "0"):
            return False
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            pass
        try:
            return yaml.safe_load(val)
        except Exception:
            pass
        return val

    @staticmethod
    def _expand_env_vars(node: Any) -> Any:
        import re

        if isinstance(node, dict):
            return {k: Config._expand_env_vars(v) for k, v in node.items()}
        if isinstance(node, list):
            return [Config._expand_env_vars(v) for v in node]
        if isinstance(node, str):
            def _sub(m: "re.Match") -> str:
                return os.environ.get(m.group(1), m.group(0))

            return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", _sub, node)
        return node

    @staticmethod
    def _set_nested(d: dict, path: list[str], value: Any):
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    def get_section(self, section: str) -> Dict[str, Any]:
        return self._data.get(section, {})

    @property
    def task_routing(self) -> Dict[str, Any]:
        return self.get("task_routing", {})

    @property
    def pipeline(self) -> Dict[str, Any]:
        return self.get("pipeline", {})

    @property
    def chunker(self) -> Dict[str, Any]:
        return self.get("chunker", {})

    @property
    def decision_engine(self) -> Dict[str, Any]:
        return self.get("decision_engine", {})

    @property
    def critic_thresholds(self) -> Dict[str, float]:
        return self.get("critic_thresholds", {})

    @property
    def style_guide(self) -> Dict[str, Any]:
        return self.get("style_guide", {})

    @property
    def paths(self) -> Dict[str, str]:
        return self.get("paths", {})

    @property
    def logging(self) -> Dict[str, Any]:
        return self.get("logging", {})

    @property
    def role_models(self) -> Dict[str, Any]:
        return self.get("main.models") or self.get("models", {})

    def has_role_models(self) -> bool:
        return bool(self.role_models)

    def get_role_config(self, role: str) -> Dict[str, Any]:
        cfg = self.role_models.get(role)
        if cfg:
            return cfg
        raise ValueError(f"未定义的角色: {role}，请在 config.yaml 的 main.models 中配置")

    def get_task_role(self, task_name: str) -> str:
        routing = self.task_routing.get(task_name, {})
        return routing.get("model", "primary")

    def resolve_task_model(self, task_name: str) -> tuple[str, Dict[str, Any]]:
        model_key = self.get_task_role(task_name)
        routing = self.task_routing.get(task_name, {})
        params = routing.get("params", {})
        return model_key, params

    def reload(self):
        self._data = {}
        self._load()


config = Config()


def get_config() -> Config:
    return config
