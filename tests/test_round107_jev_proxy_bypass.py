"""Round 107 P2: JevBackend 代理方案归一化 (socks:// → socks5://).

背景 (三次误诊的教训):
  1. 最初诊断 "Jev 451 区域封锁" → 加 mock 护栏 (没解决根因)
  2. 再诊断 "SOCKS 代理 scheme 错误" → 用 trust_env=False 绕过代理
     → 反而绕过了用户绕过 451 封锁所依赖的 SOCKS 代理, 直连仍 451.
  3. 装 socksio 后仍失败 → httpx2 只支持 socks5/socks5h 方案, 不支持 socks.
     用户环境变量用 socks://, httpx2 报 "Unknown scheme for proxy URL".

最终方案:
  - 代码读代理环境变量, 若方案是 socks:// 则归一化为 socks5://
  - 创建自定义 httpx2.Client(proxy=归一化后的 URL) 传给 TypeSafeClient
  - 前置条件: pip install socksio
  - 若代理不可用, Jev 调用失败 → 自动降级 openai_compat (已有链路)

测试覆盖:
  - socks:// 环境变量被归一化为 socks5://
  - socks5:// 环境变量保持不变
  - 无代理环境变量时不传自定义 http_client
  - client 缓存
  - 无 API key 抛 RuntimeError
  - 两版行为一致
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Round 130: 本文件直接构造 JevBackend 并 patch httpx2/typesafe_sdk（可选依赖，
# 缺失时生产自动降级）；无依赖环境应跳过而非失败。
pytest.importorskip("httpx2", reason="可选依赖 httpx2/socksio 未安装（Jev 代理）")
pytest.importorskip("typesafe_sdk", reason="可选依赖 typesafe_sdk 未安装（Jev 后端）")


class TestJevProxyNormalization:
    def test_socks_scheme_normalized_to_socks5(self):
        """socks:// 环境变量被归一化为 socks5://."""
        from src.judges.typesafe_backend import JevBackend

        backend = JevBackend(api_key="test-key")

        with patch.dict(os.environ, {"ALL_PROXY": "socks://127.0.0.1:10808/"}), \
             patch("httpx2.Client") as mock_httpx_client, \
             patch("typesafe_sdk.TypeSafeClient") as mock_typesafe_client:
            mock_httpx_client.return_value = MagicMock()
            mock_typesafe_client.return_value = MagicMock()

            backend._ensure_client()

            # 验证 httpx2.Client 被调用时 proxy 是 socks5:// (归一化后)
            mock_httpx_client.assert_called_once()
            call_kwargs = mock_httpx_client.call_args.kwargs
            proxy = call_kwargs.get("proxy", "")
            assert proxy.startswith("socks5://"), \
                f"proxy 应归一化为 socks5://, 实际: {proxy}"
            assert "127.0.0.1:10808" in proxy

    def test_socks5_scheme_unchanged(self):
        """socks5:// 环境变量保持不变."""
        from src.judges.typesafe_backend import JevBackend

        backend = JevBackend(api_key="test-key")

        with patch.dict(os.environ, {"ALL_PROXY": "socks5://127.0.0.1:10808/"}), \
             patch("httpx2.Client") as mock_httpx_client, \
             patch("typesafe_sdk.TypeSafeClient") as mock_typesafe_client:
            mock_httpx_client.return_value = MagicMock()
            mock_typesafe_client.return_value = MagicMock()

            backend._ensure_client()

            call_kwargs = mock_httpx_client.call_args.kwargs
            proxy = call_kwargs.get("proxy", "")
            assert proxy == "socks5://127.0.0.1:10808/", \
                f"socks5:// 应保持不变, 实际: {proxy}"

    def test_no_proxy_env_uses_sdk_default(self):
        """无代理环境变量时不传自定义 http_client (SDK 默认)."""
        from src.judges.typesafe_backend import JevBackend

        backend = JevBackend(api_key="test-key")

        # 清除所有代理环境变量
        env_clean = {k: v for k, v in os.environ.items()
                     if k.upper() not in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY")}
        with patch.dict(os.environ, env_clean, clear=True), \
             patch("typesafe_sdk.TypeSafeClient") as mock_typesafe_client:
            mock_typesafe_client.return_value = MagicMock()

            backend._ensure_client()

            ts_kwargs = mock_typesafe_client.call_args.kwargs
            assert "http_client" not in ts_kwargs, \
                f"无代理时不应传 http_client, 实际 kwargs: {list(ts_kwargs.keys())}"

    def test_ensure_client_caches_client(self):
        """_ensure_client 缓存 client, 多次调用只创建一次."""
        from src.judges.typesafe_backend import JevBackend

        backend = JevBackend(api_key="test-key")

        with patch.dict(os.environ, {"ALL_PROXY": "socks5://127.0.0.1:10808/"}), \
             patch("httpx2.Client") as mock_httpx_client, \
             patch("typesafe_sdk.TypeSafeClient") as mock_typesafe_client:
            mock_httpx_client.return_value = MagicMock()
            mock_typesafe_client.return_value = MagicMock()

            client1 = backend._ensure_client()
            client2 = backend._ensure_client()

            assert client1 is client2
            assert mock_httpx_client.call_count == 1
            assert mock_typesafe_client.call_count == 1

    def test_ensure_client_requires_api_key(self):
        """无 API key 时 _ensure_client 抛 RuntimeError."""
        from src.judges.typesafe_backend import JevBackend

        with patch.dict("os.environ", {}, clear=True):
            backend = JevBackend(api_key=None)
            with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
                backend._ensure_client()

    def test_monolith_and_scattered_match(self):
        """单体版与散装版代理归一化行为一致."""
        from src.judges.typesafe_backend import JevBackend as ScatteredJev
        from src.translator_agent import JevBackend as MonolithJev

        for JevClass, name in [(ScatteredJev, "散装版"), (MonolithJev, "单体版")]:
            backend = JevClass(api_key="test-key")
            with patch.dict(os.environ, {"ALL_PROXY": "socks://127.0.0.1:10808/"}), \
                 patch("httpx2.Client") as mock_httpx_client, \
                 patch("typesafe_sdk.TypeSafeClient") as mock_typesafe_client:
                mock_httpx_client.return_value = MagicMock()
                mock_typesafe_client.return_value = MagicMock()

                backend._ensure_client()

                call_kwargs = mock_httpx_client.call_args.kwargs
                proxy = call_kwargs.get("proxy", "")
                assert proxy.startswith("socks5://"), \
                    f"{name} proxy 应归一化为 socks5://, 实际: {proxy}"
