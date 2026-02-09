"""
YANO Matrix (LGSOC) 神经符号 AI 实验设计示例代码（带统一进度显示与 CLI）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import importlib
import importlib.util
import itertools
import json
import math
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_omics(n: int, d: int, n_classes: int, shift: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
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


class TextProgressBar:
    def __init__(self, total: int, desc: str):
        self.total = max(total, 1)
        self.desc = desc
        self.current = 0

    def update(self, step: int = 1) -> None:
        self.current += step
        pct = min(100, int(self.current / self.total * 100))
        print(f"[{self.desc}] {pct:3d}% ({self.current}/{self.total})")

    def set_postfix(self, **kwargs: Any) -> None:
        extras = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        if extras:
            print(f"[{self.desc}] {extras}")

    def close(self) -> None:
        return


class ProgressManager:
    def __init__(self, enabled: bool = True, verbose: bool = False):
        self.enabled = enabled
        self.verbose = verbose
        self._tqdm_mod = None
        self._is_tty = bool(getattr(__import__("sys").stdout, "isatty", lambda: False)())
        if importlib.util.find_spec("tqdm") is not None:
            self._tqdm_mod = importlib.import_module("tqdm")

    @property
    def mode(self) -> str:
        if not self.enabled:
            return "off"
        if self._tqdm_mod is not None and self._is_tty:
            return "tqdm"
        return "text"

    def bar(self, total: int, desc: str):
        if not self.enabled:
            return None
        if self._tqdm_mod is not None and self._is_tty:
            return self._tqdm_mod.tqdm(total=total, desc=desc, leave=False)
        return TextProgressBar(total=total, desc=desc)

    def wrap(self, iterable: Iterable[Any], total: Optional[int], desc: str) -> Iterator[Any]:
        if not self.enabled:
            yield from iterable
            return
        if self._tqdm_mod is not None and self._is_tty:
            yield from self._tqdm_mod.tqdm(iterable, total=total, desc=desc, leave=False)
            return
        total_v = total if total is not None else len(list(iterable))
        bar = TextProgressBar(total_v, desc)
        for item in iterable:
            yield item
            bar.update(1)


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


def multi_class_auc(y_true: np.ndarray, probs: np.ndarray) -> Optional[float]:
    classes = np.unique(y_true)
    aucs: List[float] = []
    for c in classes:
        y_bin = (y_true == c).astype(np.int32)
        pos = int(y_bin.sum())
        neg = int(len(y_bin) - pos)
        if pos == 0 or neg == 0:
            continue
        scores = probs[:, int(c)]
        ranks = scores.argsort().argsort() + 1
        rank_pos = ranks[y_bin == 1].sum()
        auc = (rank_pos - (pos * (pos + 1) / 2)) / (pos * neg)
        aucs.append(float(auc))
    if not aucs:
        return None
    return float(np.mean(aucs))


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

    def pretrain_source(
        self,
        x_src: np.ndarray,
        y_src: np.ndarray,
        epochs: int,
        lr: float,
        progress: ProgressManager,
    ) -> None:
        x = torch.tensor(x_src, device=self.device)
        y = torch.tensor(y_src, device=self.device)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        bar = progress.bar(epochs, "pretrain_source")
        best_loss = float("inf")

        self.model.train()
        for ep in range(1, epochs + 1):
            logits = self.model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            vloss = float(loss.detach().cpu().item())
            best_loss = min(best_loss, vloss)
            if bar is not None:
                bar.update(1)
                if hasattr(bar, "set_postfix"):
                    bar.set_postfix(loss=f"{vloss:.4f}", best=f"{best_loss:.4f}")
            if progress.verbose:
                print(f"[pretrain_source][epoch {ep}] loss={vloss:.4f} best={best_loss:.4f}")
        if bar is not None:
            bar.close()

    def finetune_target_classifier(
        self,
        x_tgt: np.ndarray,
        y_tgt: np.ndarray,
        epochs: int,
        lr: float,
        progress: ProgressManager,
    ) -> None:
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        self.model.classifier = nn.Linear(self.emb_dim, self.tgt_classes).to(self.device)

        x = torch.tensor(x_tgt, device=self.device)
        y = torch.tensor(y_tgt, device=self.device)
        n = len(y_tgt)
        split = max(1, int(n * 0.8))
        x_train, y_train = x[:split], y[:split]
        x_val, y_val = x[split:], y_tgt[split:]

        opt = optim.Adam(self.model.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        bar = progress.bar(epochs, "finetune_target_classifier")
        best_auc = -1.0

        self.model.train()
        for ep in range(1, epochs + 1):
            logits = self.model(x_train)
            loss = criterion(logits, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_v = float(loss.detach().cpu().item())

            val_auc = None
            if len(y_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(x_val)
                    probs = torch.softmax(val_logits, dim=1).cpu().numpy()
                val_auc = multi_class_auc(y_val, probs)
                self.model.train()
                if val_auc is not None:
                    best_auc = max(best_auc, val_auc)

            if bar is not None:
                bar.update(1)
                postfix = {"loss": f"{loss_v:.4f}"}
                if val_auc is not None:
                    postfix["val_auc"] = f"{val_auc:.4f}"
                    postfix["best"] = f"{best_auc:.4f}"
                if hasattr(bar, "set_postfix"):
                    bar.set_postfix(**postfix)
            if progress.verbose:
                print(
                    f"[finetune_target_classifier][epoch {ep}] loss={loss_v:.4f} "
                    f"val_auc={('NA' if val_auc is None else f'{val_auc:.4f}')} best={best_auc:.4f}"
                )
        if bar is not None:
            bar.close()

    def extract_embeddings(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            emb = self.model.embed(torch.tensor(x, device=self.device)).cpu().numpy()
        return emb


def simple_smote(x: np.ndarray, y: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) >= target_size:
        return x, y
    x_new = [x]
    y_new = [y]
    classes, counts = np.unique(y, return_counts=True)
    class_to_idx = {c: np.where(y == c)[0] for c in classes}
    needed = target_size - len(y)
    for _ in range(needed):
        c = classes[np.argmin(counts)]
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
        return {int(c): emb[y == c].mean(axis=0) for c in np.unique(y)}

    def predict(self, emb_q: np.ndarray, prototypes: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        classes = sorted(prototypes.keys())
        proto_mat = np.stack([prototypes[c] for c in classes], axis=0)
        dists = np.sqrt(((emb_q[:, None, :] - proto_mat[None, :, :]) ** 2).sum(axis=2))
        scores = -dists
        pred_idx = np.argmax(scores, axis=1)
        preds = np.array([classes[i] for i in pred_idx])
        conf = torch.softmax(torch.tensor(scores), dim=1).numpy()
        return preds, conf


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
class SemanticParserAgent:
    def run(self, raw_note: str) -> Dict[str, Any]:
        return {
            "mutation": "KRAS_G12D" if "KRAS" in raw_note.upper() else None,
            "dose_mg": 100 if "100mg" in raw_note.lower() else None,
            "ovary_resected": "切除" in raw_note,
            "symptom": "排卵痛" if "排卵痛" in raw_note else None,
        }


@dataclass
class LogicAuditAgent:
    def run(self, record: Dict[str, Any]) -> List[str]:
        warnings = []
        if record.get("ovary_resected") and record.get("symptom") == "排卵痛":
            warnings.append("红色警告：已切除卵巢却记录排卵痛，建议人工复核/剔除。")
        return warnings


@dataclass
class ReasoningAgent:
    def run(self, gene: str, score: float) -> str:
        return f"候选靶点 {gene} 的综合得分为 {score:.3f}。推理链：上游调控 -> 靶点功能 -> 下游信号 -> 表型关联。"


@dataclass
class ReportAgent:
    def run(self, analysis: Dict[str, Any]) -> str:
        return json.dumps(analysis, ensure_ascii=False, indent=2)


@dataclass
class ReflectionAgent:
    memory: List[Dict[str, Any]] = field(default_factory=list)

    def run(self, feedback: Dict[str, Any]) -> None:
        self.memory.append(feedback)


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

    def train(self, x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray, y_tgt: np.ndarray, progress: ProgressManager) -> Dict[int, np.ndarray]:
        self.transfer.pretrain_source(x_src, y_src, epochs=20, lr=1e-3, progress=progress)
        self.transfer.finetune_target_classifier(x_tgt, y_tgt, epochs=15, lr=5e-3, progress=progress)
        emb_tgt = self.transfer.extract_embeddings(x_tgt)
        emb_aug, y_aug = simple_smote(emb_tgt, y_tgt, target_size=1800)
        return self.proto.fit_prototypes(emb_aug, y_aug)

    def infer_target_gene(self, prototypes: Dict[int, np.ndarray], x_query: np.ndarray, gene_name: str) -> Dict[str, Any]:
        emb_q = self.transfer.extract_embeddings(x_query)
        preds, conf = self.proto.predict(emb_q, prototypes)
        raw_score = float(conf.max(axis=1).mean())
        kg_score = self.kg.gate_score(gene_name, raw_score)
        return {
            "pred_class": preds.tolist(),
            "raw_score": raw_score,
            "kg_score": kg_score,
            "explanation": self.reason_agent.run(gene_name, kg_score),
        }

    def run_agents(self, raw_text: str, analysis: Dict[str, Any], progress: ProgressManager) -> Dict[str, Any]:
        stages = ["解析", "审查", "推理", "反馈", "报告", "记忆更新"]
        bar = progress.bar(len(stages), "run-agents")

        parsed: Dict[str, Any] = {}
        warnings: List[str] = []
        explain = ""
        report = ""
        stage_metrics: List[Dict[str, Any]] = []

        def finish_stage(stage: str, fn):
            t0 = perf_counter()
            status = "成功"
            try:
                data = fn()
            except Exception:
                status = "失败"
                data = None
            elapsed = perf_counter() - t0
            if status == "成功" and stage == "审查" and warnings:
                status = "告警"
            print(f"[{stage}] {status} | 耗时 {elapsed:.3f}s")
            stage_metrics.append({"stage": stage, "status": status, "elapsed_s": round(elapsed, 3)})
            if bar is not None:
                bar.update(1)
            return data

        parsed = finish_stage("解析", lambda: self.parser_agent.run(raw_text)) or {}
        warnings = finish_stage("审查", lambda: self.audit_agent.run(parsed)) or []
        explain = finish_stage("推理", lambda: self.reason_agent.run("KRAS", float(analysis.get("kg_score", 0.0)))) or ""
        finish_stage("反馈", lambda: {"user_feedback": "候选可信"})
        report = finish_stage("报告", lambda: self.report_agent.run({"parsed": parsed, "analysis": analysis, "warnings": warnings, "explain": explain})) or ""
        finish_stage("记忆更新", lambda: self.reflection_agent.run({"gene": "KRAS", "validated": True, "ts": datetime.utcnow().isoformat()}))

        if bar is not None:
            bar.close()

        return {
            "parsed_record": parsed,
            "warnings": warnings,
            "analysis": analysis,
            "report": report,
            "memory_size": len(self.reflection_agent.memory),
            "stage_metrics": stage_metrics,
        }


TOKEN_RE = re.compile(r"[A-Z][A-Z0-9_]{1,}")


def build_kg_from_literature(literature_dir: Path, report_every: int, progress: ProgressManager) -> Dict[str, Any]:
    files = sorted([p for p in literature_dir.glob("**/*") if p.is_file()])
    entities: set[str] = set()
    relations: set[Tuple[str, str]] = set()

    last_report_e = 0
    last_report_r = 0
    for idx, fp in enumerate(progress.wrap(files, total=len(files), desc="build-kg 文件"), start=1):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        ents = TOKEN_RE.findall(text.upper())
        entities.update(ents)
        for a, b in itertools.pairwise(ents):
            if a != b:
                relations.add((a, b))
        if idx % report_every == 0:
            de = len(entities) - last_report_e
            dr = len(relations) - last_report_r
            print(f"[build-kg] 已处理 {idx} 篇 | 实体累计={len(entities)} 关系累计={len(relations)} | 本批去重净增长 实体={de} 关系={dr}")
            last_report_e = len(entities)
            last_report_r = len(relations)

    return {
        "files": len(files),
        "entities": len(entities),
        "relations": len(relations),
        "sample_entities": sorted(list(entities))[:10],
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


def make_pipeline() -> YanoMatrixPipeline:
    return YanoMatrixPipeline(
        transfer=TransferLearningLayer(in_dim=64, hidden=128, emb_dim=32, src_classes=4, tgt_classes=3),
        proto=PrototypicalLayer(),
        kg=KnowledgeGraphConstraint(graph=build_demo_kg(), disease_node="LGSOC", max_hop=3, penalty=0.6),
        parser_agent=SemanticParserAgent(),
        audit_agent=LogicAuditAgent(),
        reason_agent=ReasoningAgent(),
        report_agent=ReportAgent(),
        reflection_agent=ReflectionAgent(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YANO Matrix CLI")
    parser.add_argument("--progress", choices=["on", "off"], default="on", help="是否启用进度显示")
    parser.add_argument("--verbose", action="store_true", help="输出详细诊断")

    sub = parser.add_subparsers(dest="command", required=True)
    p_kg = sub.add_parser("build-kg", help="从文献目录构建简化 KG")
    p_kg.add_argument("--literature-dir", type=Path, required=True)
    p_kg.add_argument("--report-every", type=int, default=10)

    sub.add_parser("train", help="训练迁移学习 + 原型网络")

    p_agents = sub.add_parser("run-agents", help="执行 6 阶段智能体流程")
    p_agents.add_argument("--note", type=str, default="患者术后卵巢已切除，但主诉仍有排卵痛；KRAS突变；方案100mg。")
    return parser.parse_args()


def main() -> None:
    seed_all(42)
    args = parse_args()
    progress = ProgressManager(enabled=(args.progress == "on"), verbose=args.verbose)
    if args.verbose:
        print(f"[diag] progress_mode={progress.mode} tty={progress._is_tty}")

    if args.command == "build-kg":
        summary = build_kg_from_literature(args.literature_dir, args.report_every, progress)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    pipeline = make_pipeline()

    if args.command == "train":
        x_src, y_src = make_synthetic_omics(n=600, d=64, n_classes=4, shift=0.2)
        x_tgt, y_tgt = make_synthetic_omics(n=160, d=64, n_classes=3, shift=0.0)
        prototypes = pipeline.train(x_src, y_src, x_tgt, y_tgt, progress=progress)
        print(json.dumps({"prototype_classes": sorted(list(prototypes.keys())), "count": len(prototypes)}, ensure_ascii=False, indent=2))
        return

    if args.command == "run-agents":
        x_src, y_src = make_synthetic_omics(n=600, d=64, n_classes=4, shift=0.2)
        x_tgt, y_tgt = make_synthetic_omics(n=160, d=64, n_classes=3, shift=0.0)
        x_query, _ = make_synthetic_omics(n=20, d=64, n_classes=3, shift=0.05)
        prototypes = pipeline.train(x_src, y_src, x_tgt, y_tgt, progress=ProgressManager(enabled=False))
        analysis = pipeline.infer_target_gene(prototypes, x_query, gene_name="KRAS")
        result = pipeline.run_agents(args.note, analysis, progress=progress)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
