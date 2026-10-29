"""
SagaTrans 配置加载器
支持：config.yaml + 环境变量覆盖
环境变量前缀：SAGATRANS_
"""
import json
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


# Round 129: 任务级 effort 档位表（各家文档口径 + 2026-10-26/27 实测）。
# 「none」= 显式关闭思考（DeepSeek/Qwen 合法）；MiniMax 服务端强制 adaptive，
# 无 none 档（写 none 会 400）。未识别的 provider 不校验、原样透传。
EFFORT_LEVELS_BY_PROVIDER: Dict[str, tuple] = {
    "minimax": ("low", "medium", "high", "xhigh", "max"),
    "deepseek": ("none", "low", "high", "xhigh", "max"),
    "qwen": ("none", "low", "medium", "xhigh"),
}


def effort_provider_of(model_name: str, api_base: str = "") -> Optional[str]:
    """按 model_name/api_base 推断 effort 档位的 provider 归属；未知返回 None。"""
    target = f"{model_name or ''} {api_base or ''}".lower()
    for provider in EFFORT_LEVELS_BY_PROVIDER:
        if provider in target:
            return provider
    return None


class Config:
    """配置单例，惰性加载"""
    _instance: Optional['Config'] = None
    _data: Dict[str, Any] = {}
    _book_thresh_cache: Dict[str, Any] = {}

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
        """点号分隔的键路径获取，如 'models.primary.model_name'"""
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
    def input_prep(self) -> Dict[str, Any]:
        """切分后预处理：前页/索引排除表 + 文本修复表（见 config.yaml input_prep 段）。"""
        return self.get("input_prep", {})

    @property
    def decision_engine(self) -> Dict[str, Any]:
        return self.get("decision_engine", {})

    @property
    def critic_thresholds(self) -> Dict[str, float]:
        return self.get("critic_thresholds", {})

    def critic_thresholds_for_book(self, project_root=None) -> Dict[str, float]:
        """有效 critic 阈值 = config 通用默认 ⊕ book_profile.json 可选覆盖。

        作品无关原则（Round 130 外部审计④）：任何作品的专属阈值不硬编码进 config；
        按书校准写入 input/book_profile.json 的 "critic_thresholds" 块（仅列出需要
        覆盖的维度，取 0-10 数值）。剖面缺失/损坏/字段非法时静默回退通用默认。
        """
        merged: Dict[str, float] = dict(self.critic_thresholds or {})
        try:
            root = Path(project_root) if project_root else Path.cwd()
            profile_path = root / "input" / "book_profile.json"
            st = profile_path.stat()
            key = f"{profile_path}:{st.st_mtime_ns}"
            cache = Config._book_thresh_cache
            if cache.get("key") != key:
                override: Dict[str, float] = {}
                data = json.loads(profile_path.read_text(encoding="utf-8"))
                blk = data.get("critic_thresholds") if isinstance(data, dict) else None
                if isinstance(blk, dict):
                    for k, v in blk.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 10.0:
                            override[str(k)] = float(v)
                cache["key"] = key
                cache["value"] = override
            merged.update(cache.get("value") or {})
        except Exception:
            pass
        return merged

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
        """模型定义节（唯一事实源）。优先顶层 models；兼容旧版 main.models 嵌套。"""
        return self.get("models") or self.get("main.models", {})

    @property
    def models(self) -> Dict[str, Any]:
        return self.role_models

    def has_role_models(self) -> bool:
        return bool(self.role_models)

    def get_role_config(self, role: str) -> Dict[str, Any]:
        # 全局 mock 覆盖一切 per-role 配置（dry-run / 回归测试）
        if self.llm_backend == "mock":
            return {"backend": "mock", "default_params": {}}
        cfg = None
        if self.has_role_models():
            cfg = self.role_models.get(role)
        if not cfg:
            cfg = self._legacy_role_config(role)
        # Round 126: 归一 —— backend 省略即 OpenAI 兼容 HTTP（本地 mlx 支持已移除）
        merged = dict(cfg)
        merged.setdefault("backend", "openai_api")
        return merged

    def get_task_role(self, task_name: str) -> str:
        routing = self.task_routing.get(task_name, {})
        return routing.get("model", "primary")

    def _legacy_role_config(self, role: str) -> Dict[str, Any]:
        legacy = self._get_model_config(role)
        if self.llm_backend == "mock":
            return {"backend": "mock", "default_params": {}}
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
        """解析任务实际使用的 (模型 key, 参数)。

        Round 128: effort 归一为任务级 —— 规范位置 `task_routing.<task>.effort`，
        `params_override.effort` 为兼容位置（两者都写时规范位置优先）。
        旧的 `thinking` 开关已废弃：models 段或 params_override 出现该键即报错
        （避免"改了旧字段却静默不生效"）。档位按 provider 校验：MiniMax
        low/medium/high/xhigh/max（服务端强制 adaptive，无 none 档）；DeepSeek
        none/low/high/xhigh/max；Qwen none/low/medium/xhigh；未识别 provider 不校验。
        「none」= 显式关闭思考。未配置 effort → 返回值不带该键（=不下发参数）。
        YAML 裸 off（解析为布尔 False）归一为 "none"（Round 129）。

        Round 127: extra_body 归一为任务级 —— 规范位置
        `task_routing.<task>.extra_body`，`params_override.extra_body` 为兼容位置
        （两者都写时规范位置优先）；模型级 extra_body 不再读取。返回值统一携带
        `extra_body` dict，调用方直接 **params 透传给适配层（适配层最后合并）。
        """
        model_key = self.get_task_role(task_name)
        routing = self.task_routing.get(task_name, {})
        override_params = dict(routing.get("params_override", {}) or {})
        role_cfg: Dict[str, Any] = {}
        if self.has_role_models():
            role_cfg = self.role_models.get(model_key, {}) or {}
            default_params = dict(role_cfg.get("default_params", {}) or {})
        else:
            default_params = dict(self._get_model_config(model_key).get("default_params", {}) or {})
        # Round 128: thinking/enable_thinking 双开关废弃 → fail fast
        legal = "；".join(f"{p}: {'/'.join(v)}" for p, v in EFFORT_LEVELS_BY_PROVIDER.items())
        if "thinking" in override_params or "thinking" in role_cfg or "thinking" in default_params:
            raise ValueError(
                f"{task_name}: `thinking` 开关已废弃（Round 128）——请改用任务级 "
                f"`effort: <level>`（合法档位 {legal}）"
            )
        if "effort" in role_cfg:
            raise ValueError(
                f"models.{model_key}.effort 不受支持（Round 128）：档位是任务级配置，"
                f"请写 task_routing.<task>.effort"
            )
        nested_effort = override_params.pop("effort", None)
        merged = {**default_params, **override_params}
        merged.pop("effort", None)
        effort = routing.get("effort")
        if effort is None:
            effort = nested_effort
        if effort is not None:
            if isinstance(effort, bool):  # YAML 1.1: 裸 off/on/no/yes 会被解析为布尔
                if effort is True:
                    raise ValueError(
                        f"{task_name}: effort 为布尔 true（应出自裸 on/yes）——"
                        f"需要思考请写具体档位（{legal}）；关闭思考写 none（引号可选）"
                    )
                effort = "none"
            effort = str(effort).strip().lower()
            provider = effort_provider_of(
                role_cfg.get("model_name", ""), role_cfg.get("api_base", "")
            )
            levels = EFFORT_LEVELS_BY_PROVIDER.get(provider) if provider else None
            if levels and effort not in levels:
                raise ValueError(
                    f"{task_name}: effort '{effort}' 对 {provider} 非法；合法档位: {'/'.join(levels)}"
                )
            merged["effort"] = effort
        nested_extra = merged.pop("extra_body", None)
        task_extra = routing.get("extra_body")
        if task_extra is None:
            task_extra = nested_extra
        merged["extra_body"] = dict(task_extra) if isinstance(task_extra, dict) else {}
        return model_key, merged

    def _get_model_config(self, model_key: str) -> Dict[str, Any]:
        """旧版模型池（兼容）：openai_api.models.<role>。

        新结构请直接用顶层 models 节；该路径仅为老配置文件保留。
        """
        return (self.openai_api or {}).get("models", {}).get(model_key, {})

    def reload(self):
        """强制重新加载（测试用）"""
        self._data = {}
        self._load()


# 全局单例
config = Config()


def get_config() -> Config:
    return config
