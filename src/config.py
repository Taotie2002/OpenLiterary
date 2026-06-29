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

        # 3. 环境变量覆盖
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """环境变量 OPENLITERARY_* 覆盖配置，支持嵌套键用 __ 分隔
        例：OPENLITERARY_MLX__MODELS__REASONING_PRIMARY__MODEL_ID=xxx
        """
        prefix = "OPENLITERARY_"
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            # 去掉前缀，按 __ 分割为嵌套路径
            path = env_key[len(prefix):].lower().split("__")
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
    def _set_nested(d: dict, path: list[str], value: Any):
        """按路径设置嵌套字典值"""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """点号分隔的键路径获取，如 'mlx.models.reasoning_primary.model_id'"""
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

    # 便捷方法：解析任务路由得到 (model_key, params)
    def resolve_task_model(self, task_name: str) -> tuple[str, Dict[str, Any]]:
        """返回 (model_key, merged_params)"""
        routing = self.task_routing.get(task_name, {})
        model_key = routing.get("model", "reasoning_primary")
        # 合并默认参数 + 任务覆盖参数
        model_cfg = self._get_model_config(model_key)
        default_params = model_cfg.get("default_params", {})
        override_params = routing.get("params_override", {})
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