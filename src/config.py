"""
SagaTrans 配置加载器
支持：config.yaml + 环境变量覆盖
环境变量前缀：SAGATRANS_
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

def _load_dotenv(env_path: 'Path', *, override: bool = False) -> int:
    """极简 .env 加载器（无外部依赖）。

    仅当环境变量未设置时注入（override=False），与 python-dotenv 默认行为一致。
    支持 KEY=value 格式、# 注释、空行、双引号/单引号包裹的值。
    返回成功加载的键数。
    """
    if not env_path.exists():
        return 0
    loaded = 0
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # 去引号
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("\"", "'"):
                val = val[1:-1]
            if not override and key in os.environ:
                continue
            os.environ[key] = val
            loaded += 1
    return loaded


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
        # 0. 项目根 .env 自动加载（无外部依赖）。已被 .gitignore 屏蔽，安全。
        #    override=False：避免覆盖 shell 已显式 export 的同名变量。
        root_dir = Path(__file__).resolve().parent.parent
        _loaded_keys = _load_dotenv(root_dir / ".env", override=False)
        if _loaded_keys:
            import logging
            logging.getLogger(__name__).debug(
                "SagaTrans .env auto-loaded %d key(s) from %s",
                _loaded_keys, root_dir / ".env",
            )

        # 1. 找到项目根目录 (translator_agent.py 的上两级)
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
        """环境变量 SAGATRANS_* 覆盖配置，支持嵌套键用 __ 分隔
        例：SAGATRANS__MODELS__REASONING_PRIMARY__MODEL_NAME=xxx
        """
        prefix = "SAGATRANS_"
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
    def verse(self) -> Dict[str, Any]:
        """Round 119 L1 韵文通道配置（enabled/阈值/lint 参数）。"""
        return self.get("verse", {})

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
    def typesafe_thresholds(self) -> Dict[str, float]:
        """Round 8: typesafe-ai (Jev) verdict 阈值（与 critic_thresholds 并列，专供 System One 路径使用）"""
        return self.get("typesafe_thresholds", {})

    @property
    def style_guide(self) -> Dict[str, Any]:
        # Round 117: 运行时风格覆盖层（Style Profile 生成后注入）。
        # 覆盖层只影响内存中的本次运行，不写 config.yaml；未设置时退回文件配置。
        overlay = getattr(self, "_runtime_style_guide", None)
        if overlay is not None:
            return overlay
        return self.get("style_guide", {})

    @property
    def base_style_guide(self) -> Dict[str, Any]:
        """config.yaml 的 style_guide 原始值（忽略运行时覆盖层）。"""
        return self.get("style_guide", {})

    def set_runtime_style_guide(self, guide: Optional[Dict[str, Any]]) -> None:
        """注入/清除运行时风格覆盖层（None 或空 dict = 清除）。"""
        self._runtime_style_guide = dict(guide) if guide else None

    @property
    def entity_variants(self) -> Dict[str, Any]:
        """P1-4: per-work 实体异体种子（entity_registry 初始化时合并用；缺失/为空 = 不注入）。"""
        return self.get("entity_variants", {}) or {}

    # Round 86: 移除 Config.paths 和 Config.logging 死 property.
    # 历史: config.yaml 仍含 paths: 和 logging: 节, 但代码全部用硬编码路径
    # (Path(__file__).resolve().parent.parent) 或默认 logging.INFO, 从未通过
    # `config.paths` / `config.logging` 访问. 整个 codebase 0 引用.
    # 注: yaml 节保留 (向后兼容运维 .yaml 配置, 不破坏现有部署); 仅删除无用的 property.

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
