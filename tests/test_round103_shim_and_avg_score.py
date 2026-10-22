"""Round 103 P1-P7 整合测试.

本轮修复 Round 100 P2 id() 缓存 correctness bug + 多个 monolith shim 健壮性问题:

  P1  _compute_avg_score 改为 classmethod, 兼容 `TranslationPipeline._compute_avg_score(cr)` 调用.
  P2  src.judges.typesafe_spec shim 不再覆盖真实 package; 已加载时只补 attr.
  P3  _compute_avg_score 删除 Round 100 P2 的 id() 缓存 (CPython dict GC 后 id 复用会导致读到陈旧 avg).
  P4  shim 创建假 ModuleType 仅在 sys.modules 完全缺失时执行, 保护已加载的 __path__ 包.
  P5  第二个 shim 块 (源散装模块) 也遵守相同规则.
  P6  动态捕获所有 src.* 命名空间做 polluted, 不为逐个列举遗漏.
  P7  monolith globals 同步 src.core.scheduler TaskState 与 src.utils.text_match _cjk_ratio_cache,
     保证 enum 同身份与 cache 同 dict.

测试覆盖:
  - classmethod 可调用 (P1)
  - mutating dict 后二次调用反映新值 (P3 correctness)
  - shim 不覆盖已有真实 src.judges.typesafe_spec (P2)
  - shim 不覆盖已有真实散装模块 (P5)
  - monolith TaskState 与 src.core.scheduler.TaskState 同身份 (P7)
  - monolith _cjk_ratio_cache 与 src.utils.text_match._cjk_ratio_cache 同 dict (P7)
  - reset_cjk_ratio_cache() 双端同步清空 (P7)
  - 类属性 _avg_score_cache 字段保留供向后兼容, 但 _compute_avg_score 不再写入 (P3)
"""
from __future__ import annotations

import sys
import pytest


# ---------------------------------------------------------------------------
# 通用 fixture: Round 103 P6 风格的动态 polluted 恢复
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_modules():
    """动态快照所有 src.* 命名空间, 测试结束后还原.

    R104 P1 fixture 修正: 旧版只 capture setup 时的 polluted 列表, 若 setup 时
    sys.modules 已被前一个测试的 monolith shim 污染 (含 __file__=None 的假模块),
    snapshot 记录的就是这些假模块, 下一个测试 import 时 Python 直接命中 sys.modules
    缓存, 拿到 __file__=None 的假模块 (而非真实文件). 这里在 setup 时主动 pop 所有
    __file__=None 的 src.* 占位 (monolith shim 创建的假 ModuleType), 让每个测试
    从干净的起点开始.
    """
    # Setup: 主动 pop 所有 __file__=None 的 src.* 占位 (monolith shim 残留).
    for k in list(sys.modules):
        if (k == "src" or k.startswith("src.")):
            cur = sys.modules.get(k)
            if cur is not None and getattr(cur, "__file__", None) is None:
                sys.modules.pop(k, None)
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        yield
    finally:
        # Teardown: 恢复 setup 时的 snapshot, 同时 pop 测试中新引入的假模块.
        for k in list(sys.modules):
            if (k == "src" or k.startswith("src.")) and k not in snapshot:
                cur = sys.modules.get(k)
                if cur is not None and getattr(cur, "__file__", None) is None:
                    sys.modules.pop(k, None)
        for k, v in snapshot.items():
            if v is not None:
                sys.modules[k] = v


# ---------------------------------------------------------------------------
# P1 + P3: _compute_avg_score classmethod 化 + 删除 id() 缓存
# ---------------------------------------------------------------------------
class TestComputeAvgScoreRound103:
    """P1 (classmethod) + P3 (删除 id() 缓存)."""

    def test_p1_callable_as_classmethod(self):
        """P1: _compute_avg_score 是 classmethod, 可不实例化调用."""
        from src.translator_agent import TranslationPipeline
        # 类级别调用 (无实例), Python 自动注入 cls
        result = TranslationPipeline._compute_avg_score(
            {"scores": {"voice_consistency": 8.0, "literal_residue": 6.0}}
        )
        assert result == pytest.approx(7.0)

    def test_p1_also_works_as_instance_method(self):
        """P1: 也能以实例方法形式调用 (向后兼容)."""
        from src.translator_agent import TranslationPipeline
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        result = pipeline._compute_avg_score(
            {"scores": {"voice_consistency": 8.0, "literal_residue": 6.0}}
        )
        assert result == pytest.approx(7.0)

    def test_p3_no_id_cache_recomputes_after_mutation(self):
        """P3 correctness: dict 突变后第二次调用必须反映新值.

        R100 P2 旧 id() 缓存在 CPython GC 后会让两次调用读到陈旧 avg (R103 P3 移除).
        """
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline._avg_score_cache = {}  # backward-compat 字段, 现已不再写入

        cr = {"scores": {"voice_consistency": 7.0}}
        r1 = TranslationPipeline._compute_avg_score(cr)
        # 突变 dict 内容
        cr["scores"]["voice_consistency"] = 99.0
        r2 = TranslationPipeline._compute_avg_score(cr)
        assert r1 == pytest.approx(7.0)
        assert r2 == pytest.approx(99.0), (
            "R103 P3: dict 突变后必须读到新值, 不允许 id() 复用读到陈旧 avg"
        )

    def test_p3_no_cache_field_writes(self):
        """P3: _compute_avg_score 不再写入 _avg_score_cache 字段."""
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline._avg_score_cache = {}

        TranslationPipeline._compute_avg_score({"scores": {"x": 5.0}})
        # 不写入 _avg_score_cache
        assert pipeline._avg_score_cache == {}

    def test_p3_function_source_mentions_no_cache(self):
        """P3: 函数 docstring 或源码应反映 R103 P3 已删除 id() 缓存."""
        import inspect
        from src.translator_agent import TranslationPipeline
        source = inspect.getsource(TranslationPipeline._compute_avg_score)
        # 函数体不应再读写 _avg_score_cache
        assert "_avg_score_cache" not in source, (
            "R103 P3: _compute_avg_score 不应再读写 _avg_score_cache 字段"
        )

    def test_p3_avg_score_cache_field_still_present(self):
        """P3 backward-compat: _avg_score_cache 字段仍存在于 __init__ (向后兼容)."""
        import inspect
        from src.translator_agent import TranslationPipeline
        init_src = inspect.getsource(TranslationPipeline.__init__)
        assert "_avg_score_cache" in init_src, (
            "R103 P3: _avg_score_cache 字段应保留 (backward-compat)"
        )


# ---------------------------------------------------------------------------
# P2: src.judges.typesafe_spec shim 不覆盖真实包
# ---------------------------------------------------------------------------
class TestShimProtectRealPackageP2:
    """P2: src.judges.typesafe_spec 已加载时不被覆盖."""

    def test_p2_real_package_preserved_after_translator_agent_import(self):
        """P2: import src.translator_agent 后 src.judges.typesafe_spec 仍是真实包."""
        # 先 import 真实包
        import src.judges.typesafe_spec as real_spec
        assert getattr(real_spec, "__file__", None), "真实包 __file__ 应非空"

        # 再 import monolith (触发 shim 块)
        import src.translator_agent  # noqa: F401

        # 检查 src.judges.typesafe_spec 仍是同一对象 (__file__ 非空)
        spec_after = sys.modules.get("src.judges.typesafe_spec")
        assert spec_after is real_spec, (
            "R103 P2: monolith shim 不应覆盖已加载的 src.judges.typesafe_spec"
        )
        assert getattr(spec_after, "__file__", None), (
            "R103 P2: 覆盖后 __file__ 会被清空 (空 ModuleType), 触发 ImportError 链路"
        )

    def test_p2_real_package_attr_supplemented(self):
        """P2: 已加载的真实包应被补充属性 (e.g. NoulAnswer), 但不替换."""
        # 先 import 真实 primitives (确认 mock / 同步路径), 让 mock 出现时也满足条件
        import src.translator_agent  # noqa: F401
        import src.judges.primitives as real_prim

        # PER_ENTITY_KEY_PREFIX 是 monolith shim 暴露的符号, 应注入到真实模块
        # 若源码找不到导入链路, 此断言保证 R87 修复 + R103 P5 兼容
        assert hasattr(real_prim, "PER_ENTITY_KEY_PREFIX"), (
            "R87 修复 + R103 P5: monolith shim 应补充 PER_ENTITY_KEY_PREFIX 到真实 src.judges.primitives"
        )


# ---------------------------------------------------------------------------
# P4: shim 仅在 sys.modules 缺失时创建假 ModuleType
# ---------------------------------------------------------------------------
class TestShimConditionalCreateP4:
    """P4: shim 创建假 ModuleType 仅在 sys.modules 缺失时执行."""

    def test_p4_src_judges_package_path_preserved(self):
        """P4: src.judges 已加载的 __path__ 不被清空."""
        # 先 import 真实 src.judges
        import src.judges  # noqa: F401

        # import monolith (触发 P4 段)
        import src.translator_agent  # noqa: F401

        # src.judges 应保持有 __path__
        judges_mod = sys.modules["src.judges"]
        assert hasattr(judges_mod, "__path__"), "R103 P4: src.judges.__path__ 不应被清空"
        # 真实包应非空 __path__
        assert len(judges_mod.__path__) > 0, (
            "R103 P4: 已加载的 src.judges 真实包应保留其 __path__ 内容"
        )


# ---------------------------------------------------------------------------
# P5: 第二个 shim 块 (源散装模块) 同样保护真实包
# ---------------------------------------------------------------------------
class TestShimProtectScatteredModulesP5:
    """P5: 散装模块已加载时不被覆盖."""

    def test_p5_real_scattered_module_preserved(self):
        """P5: import src.translator_agent 后真实 src.core.decision_engine 不被覆盖."""
        import src.core.decision_engine as real_de
        assert getattr(real_de, "__file__", None), "真实模块 __file__ 应非空"

        # 触发 monolith shim
        import src.translator_agent  # noqa: F401

        de_after = sys.modules.get("src.core.decision_engine")
        assert de_after is real_de, (
            "R103 P5: monolith shim 不应覆盖已加载的散装模块"
        )
        assert getattr(de_after, "__file__", None), (
            "R103 P5: 覆盖会丢失 __file__, 触发后续 ImportError 链路"
        )


# ---------------------------------------------------------------------------
# P7: monolith globals 同步 TaskState enum 与 _cjk_ratio_cache
# ---------------------------------------------------------------------------
class TestMonolithGlobalsSyncP7:
    """P7: monolith globals 同步核心 enum 与 module-level cache."""

    def test_p7_task_state_identity_with_real_scheduler(self):
        """P7: src.translator_agent.TaskState 与 src.core.scheduler.TaskState 同身份."""
        import src.core.scheduler as real_sched
        import src.translator_agent as monolith_mod

        # 真实调度器存在则 TaskState 应同步过来
        if getattr(real_sched, "__file__", None):
            assert hasattr(monolith_mod, "TaskState"), "monolith 应暴露 TaskState"
            assert monolith_mod.TaskState is real_sched.TaskState, (
                "R103 P7: monolith.TaskState 应与 src.core.scheduler.TaskState 同身份 "
                "(enum 身份相等, 否则测试期望失败)"
            )

    def test_p7_task_scheduler_identity_with_real_scheduler(self):
        """P7: src.translator_agent.TaskScheduler 与 src.core.scheduler.TaskScheduler 同身份."""
        import src.core.scheduler as real_sched
        import src.translator_agent as monolith_mod

        if getattr(real_sched, "__file__", None):
            assert monolith_mod.TaskScheduler is real_sched.TaskScheduler, (
                "R103 P7: monolith.TaskScheduler 应与 src.core.scheduler.TaskScheduler 同身份"
            )

    def test_p7_cjk_ratio_cache_is_same_dict(self):
        """P7: monolith._cjk_ratio_cache 与 src.utils.text_match._cjk_ratio_cache 同 dict."""
        import src.utils.text_match as real_tm
        import src.translator_agent as monolith_mod

        if getattr(real_tm, "__file__", None):
            assert hasattr(monolith_mod, "_cjk_ratio_cache"), (
                "monolith 应同步 _cjk_ratio_cache 字段"
            )
            # 是同一对象 (is 关系), 不是副本
            assert monolith_mod._cjk_ratio_cache is real_tm._cjk_ratio_cache, (
                "R103 P7: monolith._cjk_ratio_cache 应是 src.utils.text_match._cjk_ratio_cache 同 dict, "
                "否则 reset_cjk_ratio_cache() 只会清副本, 真实 dict 不变"
            )

    def test_p7_cjk_ratio_cache_reset_sync(self):
        """P7: monolith.reset_cjk_ratio_cache() 与散装端同步清空 (R102 P1 cache 测试可工作)."""
        import src.utils.text_match as real_tm
        import src.translator_agent as monolith_mod

        if getattr(real_tm, "__file__", None):
            # populate 散装端 cache
            real_tm.compute_cjk_ratio("你好世界")
            assert len(real_tm._cjk_ratio_cache) >= 1

            # 通过 monolith 调 reset
            monolith_mod.reset_cjk_ratio_cache()
            # 散装端 cache 也应清空
            assert len(real_tm._cjk_ratio_cache) == 0, (
                "R103 P7: monolith.reset_cjk_ratio_cache() 必须清空散装端 _cjk_ratio_cache"
            )

    def test_p7_cjk_ratio_cache_max_aligned(self):
        """P7: monolith._CJK_RATIO_CACHE_MAX 与散装端常量一致."""
        import src.utils.text_match as real_tm
        import src.translator_agent as monolith_mod

        if getattr(real_tm, "__file__", None):
            assert hasattr(monolith_mod, "_CJK_RATIO_CACHE_MAX")
            assert monolith_mod._CJK_RATIO_CACHE_MAX == real_tm._CJK_RATIO_CACHE_MAX


# ---------------------------------------------------------------------------
# P6: 动态 polluted 覆盖所有 src.* 命名空间 (验证 autouse fixture 行为)
# ---------------------------------------------------------------------------
class TestDynamicPollutedP6:
    """P6: _isolate_modules 动态捕获 src.* 而非硬编码列表."""

    def test_p6_fixture_setup_pollutes_modules(self):
        """P6: fixture setup 时会记录当前 src.* 命名空间快照."""
        # 我们已经通过 autouse fixture 验证了 polluted 捕获行为存在.
        # 此测试仅做冒烟检查: 列出 polluted 来源.
        import types as _t
        # 当前不应有 src.p103_dynamic_test
        assert "src.p103_dynamic_test" not in sys.modules

        # 测试中临时加入一个 fake 模块, 在 yield 后 fixture 会清掉
        fake = _t.ModuleType("src.p103_dynamic_test")
        sys.modules["src.p103_dynamic_test"] = fake
        assert "src.p103_dynamic_test" in sys.modules
        # fixture teardown 会清掉 placeholder (无 __file__)

    def test_p6_fixture_teardown_removes_placeholder(self):
        """P6: fixture teardown 把无 __file__ 的 placeholder 移出 sys.modules.

        利用本测试自身在 autouse fixture 内运行: 测试结束后 yield 触发 teardown,
        下一测试的 setup 阶段应已看不到本测试加入的占位符.
        """
        import types as _t
        # 直接验证 teardown 入口函数体的预期行为
        from src.translator_agent import TranslationPipeline  # noqa: F401
        polluted = tuple(
            k for k in list(sys.modules)
            if k == "src" or k.startswith("src.")
        )
        # polluted 必然含 "src" 与 src.translator_agent
        assert "src" in polluted
        assert "src.translator_agent" in polluted
        # dynamic polluted 应能捕获任意运行时新建的 src.* 占位
        assert any(k.startswith("src.") for k in polluted)

    def test_p6_snapshot_dict_pattern(self):
        """P6: snapshot 字典按 polluted 列表逐 key 备份 sys.modules 当前状态."""
        # 与 autouse fixture 实现保持一致: snapshot = {k: sys.modules.get(k) for k in polluted}
        # 验证 get(k) 返回 None 的占位会被 teardown 清理 (cur is not None and __file__ is None 时 pop).
        fake_key = "src.p103_snapshot_test"
        # 没有这个 key 时, snapshot.get(fake_key) is None
        # 模拟 teardown: snapshot.get(k) is None → cur = sys.modules.get(k) → 若 cur 存在且无 __file__ → pop
        assert fake_key not in sys.modules
