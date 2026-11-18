"""Round 156: evidence-gated whole-book repair inside the delivery workflow."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.book_repair import (
    _br_plan_canonical_aliases,
    _br_plan_malformed_footnotes,
    _br_plan_core_swaps,
    _br_plan_note_repairs,
    _br_plan_omissions,
    _br_plan_sentence_core_swaps,
    apply_book_repairs,
    verify_book_repairs,
)
from src.core.scheduler import TaskScheduler
from src.main import _MUTATING_COMMANDS
from src.pipeline import run_final_gate
from src.rescue import write_qa_report


ROOT = Path(__file__).resolve().parent.parent
CHUNK = "ch01_chunk000"
WRONG_NOTE = (
    "“戈姆·贾巴尔”是弗雷曼传说与禁忌体系中的专名；本段没有解释其具体所指。"
)
CORRECT_NOTE = "戈姆·贾巴尔：贝尼·杰瑟里特姐妹会用于测试人类与动物之别的毒针武器。"
WRONG_ALIAS_NOTE = (
    "戈姆刺（*gom jabbar*）：弗雷曼传说及禁忌体系中的专名，与巨型沙虫或相关器物有关。"
)
CLEAN_BOOK = "# 第一章 起点\n\n他说：“你好。”然后[^1]。\n\n[^1]: 说明。\n"
AUDIT_PASS = (
    "# 转换完整性审计报告 (T1)\n\n"
    "- 全书覆盖率: 99.900%\n"
    "- 结论: ✅ 通过\n"
)


def _write_json(path: Path, text: str, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "text": text,
        "metadata": metadata or {},
    }, ensure_ascii=False), encoding="utf-8")


def _mk_project(
    tmp_path: Path,
    final_text: str,
    *,
    source: str = "source text",
    raw: str | None = None,
    literary: str | None = None,
) -> Path:
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    db = sqlite3.connect(db_dir / "workflow.db")
    db.execute(
        "CREATE TABLE chunk_tasks (chunk_id TEXT, chapter_id TEXT, "
        "text_content TEXT, state TEXT)")
    db.execute(
        "INSERT INTO chunk_tasks VALUES (?, ?, ?, ?)",
        (CHUNK, "ch01", source, "COMPLETED"),
    )
    db.commit()
    db.close()
    _write_json(tmp_path / "output" / "ch01" / f"{CHUNK}_final.json", final_text)
    _write_json(
        tmp_path / "output" / "ch01" / f"{CHUNK}_literary.json",
        final_text if literary is None else literary,
    )
    _write_json(
        tmp_path / "output" / "ch01" / f"{CHUNK}_raw.json",
        final_text if raw is None else raw,
    )
    return tmp_path


def _final_path(root: Path) -> Path:
    return root / "output" / "ch01" / f"{CHUNK}_final.json"


def _mk_gate_project(tmp_path: Path) -> None:
    (tmp_path / "db").mkdir()
    (tmp_path / "input").mkdir()
    out = tmp_path / "output"
    out.mkdir()
    sched = TaskScheduler(db_path=str(tmp_path / "db" / "workflow.db"))
    sched.conn.execute(
        "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
        "VALUES (?, ?, ?, ?)",
        (CHUNK, "ch01", "source", "COMPLETED"),
    )
    sched.conn.commit()
    sched.conn.close()
    chapter = out / "ch01"
    chapter.mkdir()
    _write_json(chapter / f"{CHUNK}_final.json", CLEAN_BOOK)
    (chapter / f"{CHUNK}_critic_report.json").write_text(
        json.dumps({"_backend_path": "jev"}), encoding="utf-8")
    (tmp_path / "input" / "_conversion_audit.md").write_text(
        AUDIT_PASS, encoding="utf-8")
    (out / "translated_full.md").write_text(CLEAN_BOOK, encoding="utf-8")
    write_qa_report(tmp_path, out)


def _repair_payload(**overrides):
    payload = {
        "ok": True,
        "scanned": 1,
        "pending_chunks": [],
        "blocking_issues": [],
        "manifest_present": True,
        "manifest_current": True,
        "stale_manifest": [],
        "missing_sources": [],
        "malformed_finals": [],
    }
    payload.update(overrides)
    return payload


class TestCoreSwapEvidence:
    def test_sentence_context_repairs_a_split_paragraph_swap(self):
        final = (
            "门边一阵骚动打断了他。邓肯从卫兵身边走进来。\n\n"
            "保罗挥手示意他退开，说道：“邓肯，有什么话就说出来。”\n"
        )
        literary = (
            "门口一阵骚动，打断了他。邓肯从卫兵值守的门口走进来。\n\n"
            "雷托挥手让他退开，说道：“邓肯，有话直说。”\n"
        )
        source = (
            "A disturbance at the door interrupted him. Duncan came through the guard.\n\n"
            "Leto waved him back, said: “Speak out, Duncan.”\n"
        )
        repaired, changes, blocking = _br_plan_sentence_core_swaps(
            final, literary=literary, source=source)

        assert "雷托挥手示意他退开" in repaired
        assert "保罗挥手" not in repaired
        assert len(changes) == 1
        assert changes[0]["evidence"]["sentence_similarity"] >= 0.75
        assert blocking == []


class TestFootnoteShapeRepair:
    def test_restores_missing_colon_for_a_cited_definition(self):
        text = (
            "正文引用了这个解释[^1]。\n\n"
            "[^1] “术语”：这是该术语的说明。\n"
        )
        repaired, changes = _br_plan_malformed_footnotes(text)
        assert "[^1]: “术语”：这是该术语的说明。" in repaired
        assert len(changes) == 1
        assert changes[0]["rule"] == "malformed_md_footnote_definition"

    def test_does_not_touch_an_uncited_reference_line(self):
        text = "[^1] “术语”：这是该术语的说明。\n"
        repaired, changes = _br_plan_malformed_footnotes(text)
        assert repaired == text
        assert changes == []

    def test_repairs_the_gom_jabbar_alias_glossary_note(self):
        text = f"正文引用了这个解释[^1]。\n\n[^1]: {WRONG_ALIAS_NOTE}\n"
        repaired, changes = _br_plan_note_repairs(text)
        assert "戈姆刺（戈姆·贾巴尔，gom jabbar）" in repaired
        assert "弗雷曼传说" not in repaired
        assert "沙虫" not in repaired
        assert len(changes) == 1
        assert changes[0]["rule"] == "gom_jabbar_fact_repair"

    def test_sentence_context_keeps_a_legitimate_paul_mention(self):
        text = "保罗看了看雷托公爵身后的白墙，又低头看向桌面。"
        repaired, changes, blocking = _br_plan_sentence_core_swaps(
            text,
            literary="保罗看了看雷托公爵身后的白墙，又低头看向桌面。",
            source="Paul looked at the white wall behind his father.",
        )
        assert repaired == text
        assert changes == []
        assert blocking == []

    def test_repairs_when_source_raw_and_literary_agree(self):
        repaired, changes, blocking = _br_plan_core_swaps(
            "保罗来了。保罗坐下。",
            literary="保罗来了。雷托坐下。",
            raw="保罗来了。雷托坐下。",
            source="Paul came. Leto sat.",
        )
        assert repaired == "保罗来了。雷托坐下。"
        assert len(changes) == 1
        assert blocking == []

    def test_refuses_when_raw_does_not_corroborate_target(self):
        repaired, changes, blocking = _br_plan_core_swaps(
            "保罗来了。保罗坐下。",
            literary="保罗来了。雷托坐下。",
            raw="保罗来了。保罗坐下。",
            source="Paul came. Leto sat.",
        )
        assert repaired == "保罗来了。保罗坐下。"
        assert changes == []
        assert any(item["kind"] == "identity_swap_without_corroboration"
                   for item in blocking)


class TestParagraphCoverageRepair:
    ANCHOR_A = (
        "门边的骚动忽然停了。邓肯从卫兵身后走进来，靴底在石地上发出短促的回声。"
        "保罗抬起头，看见父亲的书房门仍然紧闭，走廊尽头的灯一盏接一盏亮起。"
    )
    ANCHOR_B = (
        "雷托把手按在桌沿上，示意所有人安静。窗外风沙扑打玻璃，"
        "屋里只剩呼吸声和远处机器的低鸣。"
    )
    OMITTED = (
        "保罗给喷气舱输入更多动力。扑翼机侧倾着，把他们压进座椅；"
        "前方一面暗墙在星光映衬下升起。他让机翼再多使些力，也再加动力，"
        "机翼向上振起，他们越过岩层，第二月亮从右侧地平线显现。"
    )
    FILLER = (
        "走廊里的钟摆缓慢地左右摇动，墙上的影子随之弯曲。"
        "没有人说话，只有纸张翻动和水杯落在托盘上的轻响。"
    )
    SOURCE = (
        "A disturbance at the door stopped. Duncan came in from behind the guard.\n\n"
        "Paul fed more power to the jetpods. The craft banked and pressed them into "
        "their seats as a dark wall lifted. He gave it more wing and more power; "
        "they rose above the rock and the second moon showed on their right.\n\n"
        "Leto put his hand on the table and signalled for silence. Wind struck the "
        "glass while the room held only breathing and a distant machine.\n"
    )

    def test_restores_an_omitted_paragraph_between_two_anchors(self):
        final = f"{self.ANCHOR_A}\n\n{self.ANCHOR_B}\n"
        raw = f"{self.ANCHOR_A}\n\n{self.OMITTED}\n\n{self.ANCHOR_B}\n"

        repaired, changes, blocking = _br_plan_omissions(
            final, raw=raw, source=self.SOURCE, literary=final)

        assert self.OMITTED in repaired
        assert repaired.count(self.OMITTED) == 1
        assert len(changes) == 1
        assert changes[0]["rule"] == "source_paragraph_restored"
        assert changes[0]["evidence"]["source_supported"] is True
        assert blocking == []

    def test_apply_restores_paragraph_and_is_idempotent(self, tmp_path):
        final = f"{self.ANCHOR_A}\n\n{self.ANCHOR_B}\n"
        raw = f"{self.ANCHOR_A}\n\n{self.OMITTED}\n\n{self.ANCHOR_B}\n"
        root = _mk_project(tmp_path, final, source=self.SOURCE, raw=raw)

        dry = apply_book_repairs(project_root=root, output_dir=root / "output")
        assert dry["pending_chunks"] == [CHUNK]

        applied = apply_book_repairs(
            project_root=root, output_dir=root / "output", apply=True)
        data = json.loads(_final_path(root).read_text(encoding="utf-8"))
        assert applied["ok"] is True
        assert data["text"] == raw
        assert data["metadata"]["book_repair"]["rules"] == [
            "source_paragraph_restored"]

        verified = verify_book_repairs(
            project_root=root, output_dir=root / "output")
        assert verified["ok"] is True
        assert verified["manifest_current"] is True
        assert verified["pending_chunks"] == []

        again = apply_book_repairs(
            project_root=root, output_dir=root / "output", apply=True)
        current = json.loads(_final_path(root).read_text(encoding="utf-8"))
        assert again["ok"] is True
        assert again["summary"]["changed_chunks"] == 0
        assert current["text"].count(self.OMITTED) == 1

    def test_multiple_plausible_insertion_gaps_block_instead_of_guessing(
            self, tmp_path):
        final = (
            f"{self.ANCHOR_A}\n\n{self.ANCHOR_B}\n\n{self.FILLER}\n\n"
            f"{self.ANCHOR_A}\n\n{self.ANCHOR_B}\n"
        )
        raw = f"{self.ANCHOR_A}\n\n{self.OMITTED}\n\n{self.ANCHOR_B}\n"
        root = _mk_project(tmp_path, final, source=self.SOURCE, raw=raw)

        result = apply_book_repairs(
            project_root=root, output_dir=root / "output", apply=True)
        data = json.loads(_final_path(root).read_text(encoding="utf-8"))

        assert result["ok"] is False
        assert data["text"] == final
        assert any(
            issue.get("kind") == "source_paragraph_omitted"
            and "不唯一" in str(issue.get("detail"))
            for issue in result["blocking_issues"]
        )


class TestCanonicalSurfaceRepair:
    def test_restores_profile_backed_aliases(self):
        text = (
            "他们抵达阿拉基斯，看到阿拉基斯的风暴。"
            "贝尼杰瑟里特姐妹会派出使者，贝尼杰瑟里特体系继续运转。"
            "沙杜特梅普斯前来报告。"
            "费德-劳萨、菲德-劳萨和费德-？劳萨都在场。"
        )
        source = (
            "They reached Arrakis and saw the storms of Arrakis. "
            "The Bene Gesserit sent an envoy; the Bene Gesserit system continued. "
            "Shadout Mapes came to report. Feyd-Rautha was present."
        )

        repaired, changes = _br_plan_canonical_aliases(text, source=source)

        assert "阿拉基斯" not in repaired
        assert "贝尼杰瑟里特" not in repaired
        assert "沙杜特梅普斯" not in repaired
        assert "费德-劳萨" not in repaired
        assert "菲德-劳萨" not in repaired
        assert "费德-？劳萨" not in repaired
        assert repaired.count("厄拉科斯") == 2
        assert repaired.count("贝尼·杰瑟里特") == 2
        assert repaired.count("沙杜特·梅普斯") == 1
        assert repaired.count("菲德-罗萨") == 3
        assert [change["rule"] for change in changes] == [
            "canonical_surface_restore", "canonical_surface_restore",
            "canonical_surface_restore", "canonical_surface_restore",
            "canonical_surface_restore", "canonical_surface_restore"]
        assert all(change["evidence"]["source_supported"] for change in changes)

    def test_skips_alias_without_source_term_evidence(self):
        text = "他们抵达阿拉基斯。贝尼杰瑟里特姐妹会派出使者。"
        source = "An unrelated English paragraph mentions no proper nouns."

        repaired, changes = _br_plan_canonical_aliases(text, source=source)

        assert repaired == text
        assert changes == []

    def test_source_gate_is_evaluated_per_alias(self):
        text = "他们抵达阿拉基斯。贝尼杰瑟里特姐妹会派出使者。"
        source = "They reached Arrakis after a long journey."

        repaired, changes = _br_plan_canonical_aliases(text, source=source)

        assert "厄拉科斯" in repaired
        assert "贝尼杰瑟里特" in repaired
        assert "贝尼·杰瑟里特" not in repaired
        assert len(changes) == 1
        assert changes[0]["from"] == "阿拉基斯"
        assert changes[0]["evidence"]["source_supported"] is True

    def test_canonical_alias_restore_is_idempotent(self):
        repaired, _changes = _br_plan_canonical_aliases(
            "阿拉基斯与贝尼杰瑟里特", source="Arrakis and Bene Gesserit")
        again, changes = _br_plan_canonical_aliases(
            repaired, source="Arrakis and Bene Gesserit")
        assert again == repaired
        assert changes == []


class TestApplyVerifyAndEvidenceGaps:
    def test_apply_repairs_note_with_backup_and_manifest(self, tmp_path):
        original = f"正文。\n\n[^1]: {WRONG_NOTE}\n"
        root = _mk_project(tmp_path, original)

        dry = apply_book_repairs(project_root=root, output_dir=root / "output")
        assert _final_path(root).read_text(encoding="utf-8").find(WRONG_NOTE) >= 0
        assert dry["pending_chunks"] == [CHUNK]

        applied = apply_book_repairs(
            project_root=root, output_dir=root / "output", apply=True)
        data = json.loads(_final_path(root).read_text(encoding="utf-8"))
        assert CORRECT_NOTE in data["text"]
        assert WRONG_NOTE not in data["text"]
        assert applied["ok"] is True
        assert applied["pending_chunks"] == []
        proof = data["metadata"]["book_repair"]
        assert proof["before_sha256"] != proof["after_sha256"]
        backup = root / proof["backup"]
        assert WRONG_NOTE in backup.read_text(encoding="utf-8")

        verified = verify_book_repairs(
            project_root=root, output_dir=root / "output")
        assert verified["ok"] is True
        assert verified["manifest_current"] is True

    def test_verify_rejects_a_final_edited_after_repair(self, tmp_path):
        root = _mk_project(tmp_path, f"正文。\n\n[^1]: {WRONG_NOTE}\n")
        assert apply_book_repairs(
            project_root=root, output_dir=root / "output", apply=True)["ok"] is True

        data = json.loads(_final_path(root).read_text(encoding="utf-8"))
        data["text"] += "\n后续编辑。\n"
        _write_json(_final_path(root), data["text"], data.get("metadata"))

        verified = verify_book_repairs(
            project_root=root, output_dir=root / "output")
        assert verified["ok"] is False
        assert verified["manifest_current"] is False
        assert any(row.get("chunk_id") == CHUNK
                   for row in verified["stale_manifest"])

    def test_missing_source_evidence_blocks_and_does_not_edit_final(self, tmp_path):
        original = f"正文。\n\n[^1]: {WRONG_NOTE}\n"
        root = _mk_project(tmp_path, original, source="")

        result = apply_book_repairs(
            project_root=root, output_dir=root / "output", apply=True)
        data = json.loads(_final_path(root).read_text(encoding="utf-8"))

        assert result["ok"] is False
        assert result["missing_sources"] == [CHUNK]
        assert data["text"] == original


class TestWorkflowWiring:
    def test_shell_runs_repair_between_finalize_and_merge(self):
        shell = (ROOT / "scripts" / "translate_book.sh").read_text(encoding="utf-8")
        finalize_at = shell.index("translator_agent finalize --apply")
        repair_at = shell.index("translator_agent book-repair --apply")
        merge_at = shell.index('merge_output "$CHAPTERS_DIR"', repair_at)
        assert finalize_at < repair_at < merge_at
        assert "禁止合并" in shell[repair_at:merge_at]

    def test_cli_and_integrated_snapshot_expose_repair(self):
        assert "book-repair" in _MUTATING_COMMANDS
        integrated = (ROOT / "src" / "translator_agent.py").read_text(encoding="utf-8")
        assert "def apply_book_repairs" in integrated
        assert "def verify_book_repairs" in integrated
        assert '"book-repair"' in integrated


class TestGateBookRepairFailClosed:
    @pytest.mark.parametrize(("payload", "expected"), [
        (_repair_payload(pending_chunks=[CHUNK]), "待修 chunk"),
        (_repair_payload(blocking_issues=[{
            "chunk_id": CHUNK, "kind": "identity_swap_without_corroboration"}]),
         "未决高危项"),
        (_repair_payload(manifest_current=False,
                         stale_manifest=[{"chunk_id": CHUNK}]),
         "修复凭证已陈旧"),
        (_repair_payload(missing_sources=[CHUNK]), "缺少 DB source 证据"),
        (_repair_payload(malformed_finals=[CHUNK]), "无法解析 final"),
    ])
    def test_each_repair_red_state_blocks_gate(
            self, tmp_path, monkeypatch, payload, expected):
        _mk_gate_project(tmp_path)
        import src.book_repair as book_repair
        monkeypatch.setattr(
            book_repair, "verify_book_repairs", lambda **_kwargs: payload)

        result = run_final_gate(project_root=str(tmp_path))

        assert result["ok"] is False
        assert any(expected in failure and "⑪" in failure
                   for failure in result["failures"]), result["failures"]
