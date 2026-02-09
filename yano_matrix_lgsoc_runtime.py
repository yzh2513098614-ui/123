from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass
class RunDiagnostic:
    stage: str
    status: str
    duration_ms: float
    key_metrics: Dict[str, Any]
    notes_zh: str


@dataclass
class RunDiagnostics:
    items: List[RunDiagnostic] = field(default_factory=list)

    def add(self, stage: str, status: str, duration_ms: float, key_metrics: Dict[str, Any], notes_zh: str) -> None:
        self.items.append(
            RunDiagnostic(
                stage=stage,
                status=status,
                duration_ms=round(duration_ms, 2),
                key_metrics=key_metrics,
                notes_zh=notes_zh,
            )
        )

    def to_list(self) -> List[Dict[str, Any]]:
        return [asdict(i) for i in self.items]


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.title = ""

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data


@dataclass
class RemoteFileInspector:
    endpoint: str
    timeout_s: int = 15

    def inspect(self, path_or_url: str, role: str) -> Dict[str, Any]:
        parsed = urlparse(path_or_url)
        is_remote = parsed.scheme in {"http", "https"}
        filename = Path(parsed.path).name if is_remote else Path(path_or_url).name
        suffix = Path(filename).suffix.lower()
        inferred_type = "text"
        if suffix == ".json":
            inferred_type = "json"
        if suffix == ".txt":
            inferred_type = "txt"

        payload = {
            "role": role,
            "path_or_url": path_or_url,
            "filename": filename,
            "suffix": suffix,
            "is_remote": is_remote,
            "inferred_type": inferred_type,
        }
        req = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        try:
            remote_echo = json.loads(body)
        except json.JSONDecodeError:
            remote_echo = {"raw": body[:256]}

        return {
            "role": role,
            "filename": filename,
            "suffix": suffix,
            "is_remote": is_remote,
            "inferred_type": inferred_type,
            "inspect_endpoint": self.endpoint,
            "inspect_ok": True,
            "inspect_response_preview": str(remote_echo)[:200],
        }

    def read_after_inspection(self, path_or_url: str, role: str, expected_type: str) -> str:
        meta = self.inspect(path_or_url=path_or_url, role=role)
        if meta["inferred_type"] != expected_type:
            raise ValueError(
                f"文件类型不匹配: role={role}, expected={expected_type}, got={meta['inferred_type']}, file={path_or_url}"
            )

        parsed = urlparse(path_or_url)
        if parsed.scheme in {"http", "https"}:
            req = Request(path_or_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        return Path(path_or_url).read_text(encoding="utf-8")


@dataclass
class SemanticParserAgent:
    def parse(self, raw_note: str) -> Dict[str, Any]:
        mutation = re.findall(r"([A-Z0-9]{2,}_[A-Z0-9]{2,})", raw_note.upper())
        dose = re.findall(r"(\d+)\s*mg", raw_note.lower())
        return {
            "mutation": mutation[0] if mutation else None,
            "dose_mg": int(dose[0]) if dose else None,
            "ovary_resected": "切除" in raw_note,
            "symptom": "排卵痛" if "排卵痛" in raw_note else None,
        }


@dataclass
class LogicAuditAgent:
    def audit(self, record: Dict[str, Any]) -> List[str]:
        warnings: List[str] = []
        if record.get("ovary_resected") and record.get("symptom") == "排卵痛":
            warnings.append("红色警告：已切除卵巢却记录排卵痛，建议人工复核/剔除。")
        if record.get("mutation") is None:
            warnings.append("黄色警告：病历未抽取到突变位点，建议补充原始文本。")
        return warnings


@dataclass
class ReasoningAgent:
    def reason(self, top_gene: str, top_score: float, evidence_count: int) -> str:
        return f"候选靶点 {top_gene} 评分 {top_score:.3f}，外部证据条数 {evidence_count}。"


@dataclass
class ReflectionAgent:
    memory: List[Dict[str, Any]] = field(default_factory=list)

    def apply_feedback_update(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        self.memory.append(feedback)
        return {
            "priority_delta": float(feedback.get("priority_delta", 0.0)),
            "validated": bool(feedback.get("validated", False)),
            "memory_size": len(self.memory),
        }


@dataclass
class ReportAgent:
    def generate(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)


@dataclass
class KnowledgeGraphConstraint:
    graph: Dict[str, List[str]]
    disease_node: str = "LGSOC"
    max_hop: int = 3

    def shortest_hop(self, src: str, dst: str) -> int:
        if src == dst:
            return 0
        q: List[Tuple[str, int]] = [(src, 0)]
        visited = {src}
        while q:
            node, d = q.pop(0)
            for nxt in self.graph.get(node, []):
                if nxt == dst:
                    return d + 1
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, d + 1))
        return math.inf


@dataclass
class WebEvidenceCrawler:
    timeout_s: int = 15

    def fetch(self, urls: List[str], keywords: List[str]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for url in urls:
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=self.timeout_s) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
                parser = _TitleParser()
                parser.feed(html)
                lowered = html.lower()
                hit = sum(1 for k in keywords if k and k.lower() in lowered)
                results.append({"url": url, "title": parser.title.strip()[:120], "keyword_hits": hit, "status": "OK"})
            except (URLError, TimeoutError, ValueError) as exc:
                results.append({"url": url, "title": "", "keyword_hits": 0, "status": "ERROR", "error": str(exc)})
        return {"results": results, "evidence_count": sum(r["keyword_hits"] for r in results)}


@dataclass
class YanoMatrixPipeline:
    kg: KnowledgeGraphConstraint
    parser_agent: SemanticParserAgent
    audit_agent: LogicAuditAgent
    reasoning_agent: ReasoningAgent
    reflection_agent: ReflectionAgent
    report_agent: ReportAgent
    crawler: WebEvidenceCrawler

    def _format_diag_md(self, diagnostics: RunDiagnostics) -> str:
        lines = ["## 运行诊断", "", "| stage | status | duration_ms | key_metrics | notes_zh |", "|---|---|---:|---|---|"]
        for d in diagnostics.items:
            lines.append(f"| {d.stage} | {d.status} | {d.duration_ms:.2f} | `{json.dumps(d.key_metrics, ensure_ascii=False)}` | {d.notes_zh} |")
        return "\n".join(lines) + "\n"

    def _print_summary(self, diagnostics: RunDiagnostics) -> None:
        print("\n=== 运行诊断摘要 ===")
        for d in diagnostics.items:
            print(f"- [{d.stage}] {d.status} / {d.duration_ms:.2f}ms / {d.notes_zh}")

    def run_agents(
        self,
        raw_note: str,
        candidates: List[Dict[str, Any]],
        report_out: str,
        evidence_urls: Optional[List[str]] = None,
        wetlab_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        diagnostics = RunDiagnostics()
        evidence_urls = evidence_urls or []

        t0 = time.perf_counter()
        parsed = self.parser_agent.parse(raw_note)
        diagnostics.add("parser", "OK", (time.perf_counter() - t0) * 1000, {"mutation_count": int(parsed.get("mutation") is not None)}, "完成病历解析。")

        t0 = time.perf_counter()
        warnings = self.audit_agent.audit(parsed)
        diagnostics.add(
            "audit",
            "WARN" if any("红色警告" in w for w in warnings) else "OK",
            (time.perf_counter() - t0) * 1000,
            {"red_warning_count": sum("红色警告" in w for w in warnings)},
            "完成病历逻辑审查。",
        )

        t0 = time.perf_counter()
        crawl = self.crawler.fetch(evidence_urls, [parsed.get("mutation") or "", "LGSOC", "KRAS"])
        top = max(candidates, key=lambda x: float(x.get("score", 0.0))) if candidates else {"gene": "UNKNOWN", "score": 0.0}
        hop = self.kg.shortest_hop(str(top.get("gene", "UNKNOWN")), self.kg.disease_node)
        high_priority_count = sum(1 for c in candidates if str(c.get("priority", "LOW")).upper() == "HIGH")
        reason = self.reasoning_agent.reason(str(top.get("gene", "UNKNOWN")), float(top.get("score", 0.0)), int(crawl["evidence_count"]))
        status = "OK"
        notes = "推理完成。"
        if hop is math.inf:
            status = "WARN"
            notes = "KG hop 不可达，建议补全图谱边或降低 hop 限制。"
        elif high_priority_count == 0:
            status = "WARN"
            notes = "所有候选均低优先级，建议扩充候选并调整阈值。"
        diagnostics.add(
            "reasoning",
            status,
            (time.perf_counter() - t0) * 1000,
            {"evidence_count": crawl["evidence_count"], "high_priority_count": high_priority_count, "kg_hop": None if hop is math.inf else hop},
            notes,
        )

        reflection = None
        if wetlab_feedback is not None:
            t0 = time.perf_counter()
            reflection = self.reflection_agent.apply_feedback_update(wetlab_feedback)
            delta = abs(float(reflection.get("priority_delta", 0.0)))
            diagnostics.add(
                "reflection",
                "WARN" if delta >= 0.20 else "OK",
                (time.perf_counter() - t0) * 1000,
                {"priority_delta": delta},
                "反馈后优先级大幅波动，建议人工复核。" if delta >= 0.20 else "反馈更新已写入经验库。",
            )

        t0 = time.perf_counter()
        report_payload = {
            "parsed_record": parsed,
            "warnings": warnings,
            "candidates": candidates,
            "reasoning": reason,
            "web_evidence": crawl,
            "reflection": reflection,
        }
        report_json = self.report_agent.generate(report_payload)
        diagnostics.add("report", "OK", (time.perf_counter() - t0) * 1000, {"report_char_count": len(report_json)}, "报告生成完成。")

        report_path = Path(report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        diag_path = report_path.with_suffix(".diagnostics.json")

        report_md = "# LGSOC 运行报告\n\n## 结果摘要\n\n```json\n" + report_json + "\n```\n\n" + self._format_diag_md(diagnostics)
        report_path.write_text(report_md, encoding="utf-8")
        diag_path.write_text(json.dumps({"diagnostics": diagnostics.to_list()}, ensure_ascii=False, indent=2), encoding="utf-8")

        self._print_summary(diagnostics)
        print(f"报告输出: {report_path}")
        print(f"诊断输出: {diag_path}")
        return {"report_out": str(report_path), "diagnostics_out": str(diag_path)}


def _load_text_json_or_urls(inspector: RemoteFileInspector, path_or_url: str, role: str, expected_type: str) -> str:
    return inspector.read_after_inspection(path_or_url=path_or_url, role=role, expected_type=expected_type)


def main() -> None:
    parser = argparse.ArgumentParser(description="LGSOC 文件驱动诊断流水线（先联网识别文件，再读取运行）")
    parser.add_argument("--raw-note-file", required=True)
    parser.add_argument("--candidates-file", required=True, help="JSON 数组，元素含 gene/score/priority")
    parser.add_argument("--kg-file", required=True, help="JSON 邻接表")
    parser.add_argument("--report-out", default="report_out/report.md")
    parser.add_argument("--evidence-urls-file", default=None, help="TXT 文件，每行一个 URL")
    parser.add_argument("--wetlab-feedback-file", default=None, help="JSON 文件，可选")
    parser.add_argument("--file-inspect-endpoint", default="https://httpbin.org/anything/file-inspect", help="联网识别文件接口")
    args = parser.parse_args()

    inspector = RemoteFileInspector(endpoint=args.file_inspect_endpoint, timeout_s=15)

    raw_note = _load_text_json_or_urls(inspector, args.raw_note_file, "raw_note", "txt")
    candidates = json.loads(_load_text_json_or_urls(inspector, args.candidates_file, "candidates", "json"))
    kg_graph = json.loads(_load_text_json_or_urls(inspector, args.kg_file, "kg", "json"))

    evidence_urls: List[str] = []
    if args.evidence_urls_file:
        evidence_text = _load_text_json_or_urls(inspector, args.evidence_urls_file, "evidence_urls", "txt")
        evidence_urls = [line.strip() for line in evidence_text.splitlines() if line.strip() and not line.strip().startswith("#")]

    wetlab_feedback = None
    if args.wetlab_feedback_file:
        wetlab_feedback = json.loads(_load_text_json_or_urls(inspector, args.wetlab_feedback_file, "wetlab_feedback", "json"))

    pipeline = YanoMatrixPipeline(
        kg=KnowledgeGraphConstraint(graph=kg_graph, disease_node="LGSOC", max_hop=3),
        parser_agent=SemanticParserAgent(),
        audit_agent=LogicAuditAgent(),
        reasoning_agent=ReasoningAgent(),
        reflection_agent=ReflectionAgent(),
        report_agent=ReportAgent(),
        crawler=WebEvidenceCrawler(timeout_s=15),
    )
    pipeline.run_agents(
        raw_note=raw_note,
        candidates=candidates,
        report_out=args.report_out,
        evidence_urls=evidence_urls,
        wetlab_feedback=wetlab_feedback,
    )


if __name__ == "__main__":
    main()
