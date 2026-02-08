"""
YANO Matrix (LGSOC) 神经符号 AI 实验设计示例代码。

说明：
- 这是一个“可运行的研究原型（research prototype）”，用于把你给出的方案落地成 Python 架构。
- 代码重点展示：
  1) 迁移学习（HGSOC -> LGSOC）
  2) 原型网络小样本学习 + 简化 SMOTE
  3) 知识图谱约束（3-hop 门控惩罚）
  4) 多智能体流水线（清洗/质控/推理/报告/反思）

依赖：
- numpy
- torch
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# 数据层（模拟多组学数据）
# -----------------------------


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_omics(n: int, d: int, n_classes: int, shift: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """构造简化多组学特征：不同类别高斯簇。"""
    x_list, y_list = [], []
    per_cls = n // n_classes
    for c in range(n_classes):
        mean = np.zeros(d)
        mean[c % d] = 2.5 + shift
        cov = np.eye(d) * (0.7 + c * 0.03)
        cls_x = np.random.multivariate_normal(mean, cov, size=per_cls)
        cls_y = np.full(per_cls, c)
        x_list.append(cls_x)
        y_list.append(cls_y)
    x = np.vstack(x_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)
    idx = np.random.permutation(len(y))
    return x[idx], y[idx]


# -----------------------------
# Layer 1: 迁移学习特征提取
# -----------------------------


class FeatureNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, emb_dim: int, n_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(emb_dim, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))


@dataclass
class TransferLearningLayer:
    in_dim: int
    hidden: int
    emb_dim: int
    src_classes: int
    tgt_classes: int
    device: str = "cpu"
    model: FeatureNet = field(init=False)

    def __post_init__(self) -> None:
        self.model = FeatureNet(self.in_dim, self.hidden, self.emb_dim, self.src_classes).to(self.device)

    def pretrain_source(self, x_src: np.ndarray, y_src: np.ndarray, epochs: int = 20, lr: float = 1e-3) -> None:
        x = torch.tensor(x_src, device=self.device)
        y = torch.tensor(y_src, device=self.device)

        opt = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(epochs):
            logits = self.model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def finetune_target_last_layer(self, x_tgt: np.ndarray, y_tgt: np.ndarray, epochs: int = 15, lr: float = 5e-3) -> None:
        # 冻结前 N-1 层（backbone），只训练最后分类头
        for p in self.model.backbone.parameters():
            p.requires_grad = False

        # 重新定义分类头以适配 LGSOC 任务
        self.model.classifier = nn.Linear(self.emb_dim, self.tgt_classes).to(self.device)

        x = torch.tensor(x_tgt, device=self.device)
        y = torch.tensor(y_tgt, device=self.device)

        opt = optim.Adam(self.model.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(epochs):
            logits = self.model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def extract_embeddings(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            emb = self.model.embed(torch.tensor(x, device=self.device)).cpu().numpy()
        return emb


# -----------------------------
# Layer 2: 原型网络 + 简化 SMOTE
# -----------------------------


def simple_smote(x: np.ndarray, y: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """不依赖第三方 imblearn 的轻量 SMOTE：类内随机线性插值。"""
    if len(y) >= target_size:
        return x, y

    x_new = [x]
    y_new = [y]
    classes, counts = np.unique(y, return_counts=True)
    class_to_idx = {c: np.where(y == c)[0] for c in classes}

    needed = target_size - len(y)
    for _ in range(needed):
        c = classes[np.argmin(counts)]  # 优先补最小类
        idxs = class_to_idx[c]
        i1, i2 = np.random.choice(idxs, 2, replace=True)
        lam = np.random.uniform(0.15, 0.85)
        syn = x[i1] * lam + x[i2] * (1 - lam)

        x_new.append(syn[None, :])
        y_new.append(np.array([c], dtype=y.dtype))

        counts[np.where(classes == c)[0][0]] += 1

    return np.vstack(x_new), np.concatenate(y_new)


@dataclass
class PrototypicalLayer:
    def fit_prototypes(self, emb: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
        protos: Dict[int, np.ndarray] = {}
        for c in np.unique(y):
            protos[int(c)] = emb[y == c].mean(axis=0)
        return protos

    def predict(self, emb_q: np.ndarray, prototypes: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        classes = sorted(prototypes.keys())
        proto_mat = np.stack([prototypes[c] for c in classes], axis=0)

        # 距离越小，分数越高
        dists = np.sqrt(((emb_q[:, None, :] - proto_mat[None, :, :]) ** 2).sum(axis=2))
        scores = -dists
        pred_idx = np.argmax(scores, axis=1)
        preds = np.array([classes[i] for i in pred_idx])
        conf = torch.softmax(torch.tensor(scores), dim=1).numpy()
        return preds, conf


# -----------------------------
# Layer 3: 知识图谱约束（图门控）
# -----------------------------


@dataclass
class KnowledgeGraphConstraint:
    graph: Dict[str, List[str]]
    disease_node: str = "LGSOC"
    max_hop: int = 3
    penalty: float = 0.4

    def shortest_hop(self, src: str, dst: str) -> int:
        if src == dst:
            return 0
        q = [(src, 0)]
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

    def gate_score(self, gene: str, raw_score: float) -> float:
        hop = self.shortest_hop(gene, self.disease_node)
        if hop is math.inf or hop > self.max_hop:
            return raw_score * (1.0 - self.penalty)
        return raw_score


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

    def add(
        self,
        stage: str,
        status: str,
        duration_ms: float,
        key_metrics: Dict[str, Any],
        notes_zh: str,
    ) -> None:
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


# -----------------------------
# 生成式多智能体层（框架化）
# -----------------------------


@dataclass
class SemanticParserAgent:
    """语义解析智能体：将非结构化文本转 JSON（示例化规则 + 占位LLM接口）。"""

    def parse(self, raw_note: str) -> Dict[str, Any]:
        record = {
            "mutation": "KRAS_G12D" if "KRAS" in raw_note.upper() else None,
            "dose_mg": 100 if "100mg" in raw_note.lower() else None,
            "ovary_resected": "切除" in raw_note,
            "symptom": "排卵痛" if "排卵痛" in raw_note else None,
        }
        return record


@dataclass
class LogicAuditAgent:
    """逻辑审查智能体：识别临床常识/时序矛盾。"""

    def audit(self, record: Dict[str, Any]) -> List[str]:
        warnings = []
        if record.get("ovary_resected") and record.get("symptom") == "排卵痛":
            warnings.append("红色警告：已切除卵巢却记录排卵痛，建议人工复核/剔除。")
        return warnings


@dataclass
class ReasoningAgent:
    """推理决策智能体：输出可解释因果链。"""

    def reason(self, gene: str, score: float) -> str:
        return (
            f"候选靶点 {gene} 的综合得分为 {score:.3f}。"
            "推理链：上游调控 -> 靶点功能 -> 下游信号 -> 表型关联。"
            "建议进一步湿实验验证。"
        )


@dataclass
class ReportAgent:
    """报告生成智能体：整合结构化结果为文本报告（简版）。"""

    def generate(self, analysis: Dict[str, Any]) -> str:
        return json.dumps(analysis, ensure_ascii=False, indent=2)


@dataclass
class ReflectionAgent:
    """反思进化智能体：接收反馈并写入经验库（示例）。"""

    memory: List[Dict[str, Any]] = field(default_factory=list)

    def apply_feedback_update(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        self.memory.append(feedback)
        return {
            "memory_size": len(self.memory),
            "priority_delta": float(feedback.get("delta", 0.0)),
            "validated": bool(feedback.get("validated", False)),
        }


# -----------------------------
# 总控 Pipeline
# -----------------------------


@dataclass
class YanoMatrixPipeline:
    transfer: TransferLearningLayer
    proto: PrototypicalLayer
    kg: KnowledgeGraphConstraint
    parser_agent: SemanticParserAgent
    audit_agent: LogicAuditAgent
    reason_agent: ReasoningAgent
    report_agent: ReportAgent
    reflection_agent: ReflectionAgent

    def train(self, x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray, y_tgt: np.ndarray) -> Dict[int, np.ndarray]:
        self.transfer.pretrain_source(x_src, y_src)
        self.transfer.finetune_target_last_layer(x_tgt, y_tgt)

        emb_tgt = self.transfer.extract_embeddings(x_tgt)
        emb_aug, y_aug = simple_smote(emb_tgt, y_tgt, target_size=1800)
        prototypes = self.proto.fit_prototypes(emb_aug, y_aug)
        return prototypes

    def infer_target_gene(self, prototypes: Dict[int, np.ndarray], x_query: np.ndarray, gene_name: str) -> Dict[str, Any]:
        emb_q = self.transfer.extract_embeddings(x_query)
        preds, conf = self.proto.predict(emb_q, prototypes)

        # 用最大类别置信度作为原始分数（示例）
        raw_score = float(conf.max(axis=1).mean())
        kg_score = self.kg.gate_score(gene_name, raw_score)
        hop = self.kg.shortest_hop(gene_name, self.kg.disease_node)
        high_priority_count = int(kg_score >= 0.60)

        explain = self.reason_agent.reason(gene_name, kg_score)
        return {
            "gene": gene_name,
            "pred_class": preds.tolist(),
            "raw_score": raw_score,
            "kg_score": kg_score,
            "kg_hop": None if hop is math.inf else int(hop),
            "evidence_count": len(preds),
            "high_priority_count": high_priority_count,
            "explanation": explain,
        }

    def _format_diag_markdown(self, diagnostics: RunDiagnostics) -> str:
        lines = [
            "## 运行诊断",
            "",
            "| stage | status | duration_ms | key_metrics | notes_zh |",
            "|---|---|---:|---|---|",
        ]
        for d in diagnostics.items:
            metrics_text = json.dumps(d.key_metrics, ensure_ascii=False)
            lines.append(f"| {d.stage} | {d.status} | {d.duration_ms:.2f} | `{metrics_text}` | {d.notes_zh} |")
        lines.append("")
        return "\n".join(lines)

    def _print_diagnostics_summary(self, diagnostics: RunDiagnostics) -> None:
        print("\n=== 运行诊断摘要 ===")
        for d in diagnostics.items:
            print(f"- [{d.stage}] {d.status} / {d.duration_ms:.2f} ms / {d.notes_zh}")

    def run_agents(
        self,
        raw_text: str,
        analysis: Dict[str, Any],
        report_out: str = "report_out/report.md",
        wetlab_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        diagnostics = RunDiagnostics()

        parser_t0 = time.perf_counter()
        parser_status = "OK"
        parser_note = "语义解析完成，已得到结构化记录。"
        parsed_obj = self.parser_agent.parse(raw_text)
        parsed: Dict[str, Any]
        if isinstance(parsed_obj, dict):
            parsed = parsed_obj
        else:
            try:
                parsed = json.loads(parsed_obj)
                if not isinstance(parsed, dict):
                    raise ValueError("parsed result is not object")
                parser_status = "WARN"
                parser_note = "Parser 返回字符串并已转为 JSON 对象。"
            except Exception:
                parsed = {}
                parser_status = "ERROR"
                parser_note = "检测到 LLM 返回非 JSON，建议启用 JSON Schema 约束与重试策略。"
        parser_ms = (time.perf_counter() - parser_t0) * 1000
        diagnostics.add(
            stage="parser",
            status=parser_status,
            duration_ms=parser_ms,
            key_metrics={
                "mutation_count": int(parsed.get("mutation") is not None),
                "evidence_count": len([v for v in parsed.values() if v is not None]),
            },
            notes_zh=parser_note,
        )

        audit_t0 = time.perf_counter()
        warnings = self.audit_agent.audit(parsed)
        red_warning_count = sum("红色警告" in w for w in warnings)
        audit_status = "WARN" if red_warning_count > 0 else "OK"
        audit_note = (
            f"审计完成，发现 {red_warning_count} 条红色警告。"
            if red_warning_count > 0
            else "审计完成，未发现红色警告。"
        )
        diagnostics.add(
            stage="audit",
            status=audit_status,
            duration_ms=(time.perf_counter() - audit_t0) * 1000,
            key_metrics={"red_warning_count": int(red_warning_count)},
            notes_zh=audit_note,
        )

        reasoning_t0 = time.perf_counter()
        gene = str(analysis.get("gene", "UNKNOWN"))
        kg_score = float(analysis.get("kg_score", 0.0))
        kg_hop = analysis.get("kg_hop")
        reasoning_text = self.reason_agent.reason(gene, kg_score)
        high_priority_count = int(analysis.get("high_priority_count", int(kg_score >= 0.6)))
        reasoning_status = "OK"
        reasoning_note = "推理完成，可解释链路已生成。"
        if kg_hop is None:
            reasoning_status = "WARN"
            reasoning_note = "KG hop 不可达，建议补全基因到疾病节点的图谱边或放宽 hop 限制。"
        elif high_priority_count == 0:
            reasoning_status = "WARN"
            reasoning_note = "所有候选均低优先级，建议扩大候选池并复核评分阈值。"
        diagnostics.add(
            stage="reasoning",
            status=reasoning_status,
            duration_ms=(time.perf_counter() - reasoning_t0) * 1000,
            key_metrics={
                "evidence_count": int(analysis.get("evidence_count", 0)),
                "high_priority_count": high_priority_count,
                "kg_hop": kg_hop,
            },
            notes_zh=reasoning_note,
        )

        reflection_result = None
        if wetlab_feedback is not None:
            reflection_t0 = time.perf_counter()
            reflection_result = self.reflection_agent.apply_feedback_update(wetlab_feedback)
            delta = abs(float(reflection_result.get("priority_delta", 0.0)))
            reflection_status = "WARN" if delta >= 0.20 else "OK"
            reflection_note = (
                "反馈后优先级大幅波动，建议人工复核并冻结自动升降级。"
                if reflection_status == "WARN"
                else "反馈已写入经验库，优先级变动稳定。"
            )
            diagnostics.add(
                stage="reflection",
                status=reflection_status,
                duration_ms=(time.perf_counter() - reflection_t0) * 1000,
                key_metrics={"priority_delta": float(reflection_result.get("priority_delta", 0.0))},
                notes_zh=reflection_note,
            )

        report_t0 = time.perf_counter()
        payload = {
            "parsed": parsed,
            "analysis": {**analysis, "explanation": reasoning_text},
            "warnings": warnings,
            "reflection": reflection_result,
        }
        report_json = self.report_agent.generate(payload)
        report_status = "OK"
        report_note = "报告生成完成。"
        diagnostics.add(
            stage="report",
            status=report_status,
            duration_ms=(time.perf_counter() - report_t0) * 1000,
            key_metrics={"report_char_count": len(report_json)},
            notes_zh=report_note,
        )

        report_path = Path(report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path = report_path.with_suffix(".diagnostics.json")

        diagnostics_json = {
            "summary": {
                "stage_count": len(diagnostics.items),
                "warn_or_error_count": sum(i.status in {"WARN", "ERROR"} for i in diagnostics.items),
            },
            "diagnostics": diagnostics.to_list(),
        }
        diagnostics_path.write_text(json.dumps(diagnostics_json, ensure_ascii=False, indent=2), encoding="utf-8")

        report_markdown = "# LGSOC 运行报告\n\n## 结果摘要\n\n```json\n"
        report_markdown += report_json
        report_markdown += "\n```\n\n"
        report_markdown += self._format_diag_markdown(diagnostics)
        report_path.write_text(report_markdown, encoding="utf-8")

        self._print_diagnostics_summary(diagnostics)
        print(f"诊断文件已写入：{diagnostics_path}")
        print(f"报告文件已写入：{report_path}")

        return {
            "parsed_record": parsed,
            "warnings": warnings,
            "analysis": {**analysis, "explanation": reasoning_text},
            "report": report_json,
            "report_out": str(report_path),
            "diagnostics_out": str(diagnostics_path),
            "diagnostics": diagnostics.to_list(),
            "memory_size": len(self.reflection_agent.memory),
        }


def build_demo_kg() -> Dict[str, List[str]]:
    return {
        "KRAS": ["MAPK_pathway"],
        "BRAF": ["MAPK_pathway"],
        "MAPK_pathway": ["LGSOC"],
        "MEK_inhibitor": ["MAPK_pathway"],
        "TP53": ["DNA_repair"],
        "DNA_repair": ["OtherCancer"],
    }


def main() -> None:
    seed_all(42)

    # Source(HGSOC) > 500；Target(LGSOC) < 200（按你的设定）
    x_src, y_src = make_synthetic_omics(n=600, d=64, n_classes=4, shift=0.2)
    x_tgt, y_tgt = make_synthetic_omics(n=160, d=64, n_classes=3, shift=0.0)
    x_query, _ = make_synthetic_omics(n=20, d=64, n_classes=3, shift=0.05)

    pipeline = YanoMatrixPipeline(
        transfer=TransferLearningLayer(in_dim=64, hidden=128, emb_dim=32, src_classes=4, tgt_classes=3),
        proto=PrototypicalLayer(),
        kg=KnowledgeGraphConstraint(graph=build_demo_kg(), disease_node="LGSOC", max_hop=3, penalty=0.6),
        parser_agent=SemanticParserAgent(),
        audit_agent=LogicAuditAgent(),
        reason_agent=ReasoningAgent(),
        report_agent=ReportAgent(),
        reflection_agent=ReflectionAgent(),
    )

    prototypes = pipeline.train(x_src, y_src, x_tgt, y_tgt)
    analysis = pipeline.infer_target_gene(prototypes, x_query, gene_name="KRAS")

    raw_note = "患者术后卵巢已切除，但主诉仍有排卵痛；KRAS突变；方案100mg。"
    result = pipeline.run_agents(
        raw_text=raw_note,
        analysis=analysis,
        report_out="report_out/report.md",
        wetlab_feedback={"gene": "KRAS", "validated": True, "delta": +0.07},
    )

    print("=== YANO Matrix Demo Result ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
