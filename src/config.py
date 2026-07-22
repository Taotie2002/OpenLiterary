"""
OpenLiterary 配置加载器
支持：config.yaml + 环境变量覆盖
环境变量前缀：OPENLITERARY_
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """配置单例，惰性加载"""
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
        # 1. 找到项目根目录 (translator_agent.py 的上两级)
        root_dir = Path(__file__).resolve().parent.parent
        config_path = root_dir / "config.yaml"

        # 2. 读取 YAML
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        else:
            self._data = {}

        # 2.5 展开 ${ENV} 占位符（config.yaml 中 api_key: ${DEEPSEEK_API_KEY} 等）
        self._data = self._expand_env_vars(self._data)

        if not self._data:
            self._data = {}

        # 3. 环境变量覆盖
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """环境变量 OPENLITERARY_* 覆盖配置，支持嵌套键用 __ 分隔
        例：OPENLITERARY__MODELS__REASONING_PRIMARY__MODEL_NAME=xxx
        """
        prefix = "OPENLITERARY_"
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            # 去掉前缀 + 首尾下划线，按 __ 分割为嵌套路径
            raw = env_key[len(prefix):].strip("_").lower()
            path = [p for p in raw.split("__") if p]
            if not path:
                continue
            self._set_nested(self._data, path, self._parse_env_value(env_val))

    @staticmethod
    def _parse_env_value(val: str) -> Any:
        """尝试解析环境变量值为 Python 类型"""
        # 布尔值
        if val.lower() in ("true", "yes", "1"):
            return True
        if val.lower() in ("false", "no", "0"):
            return False
        # 数字
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            pass
        # JSON 尝试
        try:
            return yaml.safe_load(val)
        except Exception:
            pass
        return val

    @staticmethod
    def _expand_env_vars(node: Any) -> Any:
        """递归展开配置值中的 ${ENV} 占位符为环境变量值。

        仅当整个字符串就是一个占位符（如 "${DEEPSEEK_API_KEY}"）时整体替换；
        内嵌形式（如 "prefix-${VAR}-suffix"）也支持。未设置的环境变量
        保留原样字符串，便于后续暴露配置错误而非静默变成空值。
        """
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
        """按路径设置嵌套字典值"""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """点号分隔的键路径获取，如 'mlx.models.primary.model_id'"""
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
        """获取整个配置节"""
        return self._data.get(section, {})

    @property
    def llm_backend(self) -> str:
        return self.get("llm_backend", "mock")

    @property
    def task_routing(self) -> Dict[str, Any]:
        return self.get("task_routing", {})

    @property
    def mlx_models(self) -> Dict[str, Any]:
        return self.get("mlx.models", {})

    @property
    def openai_models(self) -> Dict[str, Any]:
        return self.get("openai_api.models", {})

    @property
    def mlx_memory(self) -> Dict[str, Any]:
        return self.get("mlx.memory", {})

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
    def openai_api(self) -> Dict[str, Any]:
        return self.get("openai_api", {})

    @property
    def role_models(self) -> Dict[str, Any]:
        return self.get("main.models") or self.get("models", {})

    def has_role_models(self) -> bool:
        return bool(self.role_models)

    def get_role_config(self, role: str) -> Dict[str, Any]:
        # 全局 mock 覆盖一切 per-role backend（dry-run / 回归测试）
        if self.llm_backend == "mock":
            return {"backend": "mock", "default_params": {}, "extra_body": {}}
        if self.has_role_models():
            cfg = self.role_models.get(role)
            if cfg:
                return cfg
        return self._legacy_role_config(role)

    def get_task_role(self, task_name: str) -> str:
        routing = self.task_routing.get(task_name, {})
        return routing.get("model", "primary")

    def _legacy_role_config(self, role: str) -> Dict[str, Any]:
        legacy = self._get_model_config(role)
        if self.llm_backend == "mock":
            return {"backend": "mock", "default_params": {}}
        if self.llm_backend == "mlx":
            return {
                "backend": "mlx",
                "model_id": legacy.get("model_id", "qwen/Qwen2.5-7B-Instruct-MLX-4bit"),
                "default_params": legacy.get("default_params", {}),
            }
        if self.llm_backend == "openai_api":
            section = self.openai_api
            return {
                "backend": "openai_api",
                "api_base": section.get("api_base", "http://127.0.0.1:1234/v1"),
                "api_key": section.get("api_key", "lm-studio"),
                "model_name": legacy.get("model_name", "deepseek-v4-flash"),
                "default_params": legacy.get("default_params", {}),
            }
        return {"backend": "mock", "default_params": {}}

    def resolve_task_model(self, task_name: str) -> tuple[str, Dict[str, Any]]:
        model_key = self.get_task_role(task_name)
        routing = self.task_routing.get(task_name, {})
        override_params = routing.get("params_override", {})
        if self.has_role_models():
            default_params = self.role_models.get(model_key, {}).get("default_params", {})
        else:
            default_params = self._get_model_config(model_key).get("default_params", {})
        merged = {**default_params, **override_params}
        return model_key, merged

    def _get_model_config(self, model_key: str) -> Dict[str, Any]:
        """从当前后端的模型池获取模型配置"""
        if self.llm_backend == "mlx":
            return self.mlx_models.get(model_key, {})
        elif self.llm_backend == "openai_api":
            return self.openai_models.get(model_key, {})
        return {}

    def reload(self):
        """强制重新加载（测试用）"""
        self._data = {}
        self._load()


# 全局单例
config = Config()


def get_config() -> Config:
    return config