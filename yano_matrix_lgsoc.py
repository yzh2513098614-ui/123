"""
YANO Matrix (LGSOC) 神经符号 AI 能力化实现。

本版本不再停留在“仅演示函数调用”的层面，而是提供：
1) 可训练的迁移学习数值核心（纯 numpy 两层 MLP + 分类头冻结微调）；
2) 可解释的知识约束评分（3-hop 图约束 + 连续惩罚）；
3) 可执行的多智能体编排（解析 -> 审计 -> 决策 -> 报告 -> 反思记忆闭环）；
4) 可落盘可复用的 CLI 工程流程（build-kg/train/infer/run-agents）。
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import random
import re
import time

import numpy as np


# -----------------------------
# 通用工具
# -----------------------------

def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def softmax(x: np.ndarray) -> np.ndarray:
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + eps)))


def make_synthetic_omics(n: int, d: int, n_classes: int, shift: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """构造模拟多组学特征（分类高斯簇）。"""
    x_list, y_list = [], []
    per_cls = n // n_classes
    for c in range(n_classes):
        mean = np.zeros(d, dtype=np.float32)
        mean[c % d] = 2.2 + shift
        cov = np.eye(d, dtype=np.float32) * (0.7 + 0.03 * c)
        cls_x = np.random.multivariate_normal(mean, cov, size=per_cls).astype(np.float32)
        cls_y = np.full(per_cls, c, dtype=np.int64)
        x_list.append(cls_x)
        y_list.append(cls_y)
    x = np.vstack(x_list)
    y = np.concatenate(y_list)
    idx = np.random.permutation(len(y))
    return x[idx], y[idx]


# -----------------------------
# Layer 1: 迁移学习特征提取
# -----------------------------

@dataclass
class FeatureNet:
    """纯 numpy 两层 MLP（backbone）+ 线性分类头。"""

    in_dim: int
    hidden: int
    emb_dim: int
    n_classes: int

    def __post_init__(self) -> None:
        scale = 0.05
        self.w1 = np.random.randn(self.in_dim, self.hidden).astype(np.float32) * scale
        self.b1 = np.zeros((1, self.hidden), dtype=np.float32)
        self.w2 = np.random.randn(self.hidden, self.emb_dim).astype(np.float32) * scale
        self.b2 = np.zeros((1, self.emb_dim), dtype=np.float32)
        self.wc = np.random.randn(self.emb_dim, self.n_classes).astype(np.float32) * scale
        self.bc = np.zeros((1, self.n_classes), dtype=np.float32)

    def embed(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.w2 + self.b2
        emb = np.maximum(z2, 0.0)
        cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "emb": emb}
        return emb, cache

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        emb, cache = self.embed(x)
        logits = emb @ self.wc + self.bc
        cache["logits"] = logits
        return logits, cache

    def reset_classifier(self, n_classes: int) -> None:
        self.n_classes = n_classes
        scale = 0.05
        self.wc = np.random.randn(self.emb_dim, self.n_classes).astype(np.float32) * scale
        self.bc = np.zeros((1, self.n_classes), dtype=np.float32)


@dataclass
class TransferLearningLayer:
    in_dim: int
    hidden: int
    emb_dim: int
    src_classes: int
    tgt_classes: int
    model: FeatureNet = field(init=False)

    def __post_init__(self) -> None:
        self.model = FeatureNet(self.in_dim, self.hidden, self.emb_dim, self.src_classes)

    def pretrain_source(self, x_src: np.ndarray, y_src: np.ndarray, epochs: int = 100, lr: float = 1e-2) -> List[float]:
        """
        Source 预训练（TransferMLP.pretrain_source）：
        输入 x_src=(N_src,D), y_src=(N_src,)；输出为训练 loss 曲线。
        """
        losses: List[float] = []
        y_one = one_hot(y_src, self.src_classes)

        for _ in range(epochs):
            logits, cache = self.model.forward(x_src)
            probs = softmax(logits)
            loss = cross_entropy(probs, y_src)
            losses.append(loss)

            grad_logits = (probs - y_one) / len(y_src)
            grad_wc = cache["emb"].T @ grad_logits
            grad_bc = np.sum(grad_logits, axis=0, keepdims=True)

            grad_emb = grad_logits @ self.model.wc.T
            grad_z2 = grad_emb * (cache["z2"] > 0)
            grad_w2 = cache["h1"].T @ grad_z2
            grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)

            grad_h1 = grad_z2 @ self.model.w2.T
            grad_z1 = grad_h1 * (cache["z1"] > 0)
            grad_w1 = cache["x"].T @ grad_z1
            grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

            self.model.w1 -= lr * grad_w1
            self.model.b1 -= lr * grad_b1
            self.model.w2 -= lr * grad_w2
            self.model.b2 -= lr * grad_b2
            self.model.wc -= lr * grad_wc
            self.model.bc -= lr * grad_bc
        return losses

    def finetune_target_last_layer(
        self, x_tgt: np.ndarray, y_tgt: np.ndarray, epochs: int = 120, lr: float = 2e-2
    ) -> List[float]:
        """
        Target 头部微调（finetune_target_classifier）：冻结 backbone，仅训练 classifier。
        这样可减少小样本中 backbone 过拟合噪声。
        """
        self.model.reset_classifier(self.tgt_classes)
        losses: List[float] = []
        y_one = one_hot(y_tgt, self.tgt_classes)

        emb, _ = self.model.embed(x_tgt)
        for _ in range(epochs):
            logits = emb @ self.model.wc + self.model.bc
            probs = softmax(logits)
            losses.append(cross_entropy(probs, y_tgt))

            grad_logits = (probs - y_one) / len(y_tgt)
            grad_wc = emb.T @ grad_logits
            grad_bc = np.sum(grad_logits, axis=0, keepdims=True)

            self.model.wc -= lr * grad_wc
            self.model.bc -= lr * grad_bc
        return losses

    def extract_embeddings(self, x: np.ndarray) -> np.ndarray:
        emb, _ = self.model.embed(x)
        return emb

    def save(self, model_path: Path) -> None:
        np.savez(
            model_path,
            w1=self.model.w1,
            b1=self.model.b1,
            w2=self.model.w2,
            b2=self.model.b2,
            wc=self.model.wc,
            bc=self.model.bc,
            src_classes=np.array([self.src_classes], dtype=np.int64),
            tgt_classes=np.array([self.tgt_classes], dtype=np.int64),
        )

    def load(self, model_path: Path) -> None:
        ckpt = np.load(model_path)
        self.model.w1 = ckpt["w1"]
        self.model.b1 = ckpt["b1"]
        self.model.w2 = ckpt["w2"]
        self.model.b2 = ckpt["b2"]
        self.model.wc = ckpt["wc"]
        self.model.bc = ckpt["bc"]


# -----------------------------
# Layer 2: 原型网络 + 简化 SMOTE
# -----------------------------

def simple_smote(x: np.ndarray, y: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) >= target_size:
        return x, y
    x_new = [x]
    y_new = [y]
    classes, counts = np.unique(y, return_counts=True)
    class_to_idx = {c: np.where(y == c)[0] for c in classes}

    for _ in range(target_size - len(y)):
        c = classes[np.argmin(counts)]
        idxs = class_to_idx[c]
        i1, i2 = np.random.choice(idxs, 2, replace=True)
        lam = np.random.uniform(0.1, 0.9)
        syn = x[i1] * lam + x[i2] * (1.0 - lam)
        x_new.append(syn[None, :])
        y_new.append(np.array([c], dtype=np.int64))
        counts[np.where(classes == c)[0][0]] += 1
    return np.vstack(x_new), np.concatenate(y_new)


@dataclass
class PrototypicalLayer:
    def fit_prototypes(self, emb: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
        return {int(c): emb[y == c].mean(axis=0) for c in np.unique(y)}

    def predict(self, emb_q: np.ndarray, prototypes: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        classes = sorted(prototypes)
        proto_mat = np.stack([prototypes[c] for c in classes], axis=0)
        dists = np.sqrt(((emb_q[:, None, :] - proto_mat[None, :, :]) ** 2).sum(axis=2))
        scores = -dists
        probs = softmax(scores)
        pred_idx = np.argmax(scores, axis=1)
        preds = np.array([classes[i] for i in pred_idx], dtype=np.int64)
        return preds, probs


# -----------------------------
# Layer 3: 知识图谱约束
# -----------------------------

@dataclass
class KnowledgeGraphConstraint:
    graph: Dict[str, List[str]]
    disease_node: str = "LGSOC"
    max_hop: int = 3
    penalty: float = 0.55

    def shortest_hop(self, src: str, dst: str) -> int:
        if src == dst:
            return 0
        q: deque[Tuple[str, int]] = deque([(src, 0)])
        visited = {src}
        while q:
            node, depth = q.popleft()
            for nxt in self.graph.get(node, []):
                if nxt == dst:
                    return depth + 1
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, depth + 1))
        return 10**9

    def gate_score(self, gene: str, raw_score: float) -> Tuple[float, Dict[str, Any]]:
        """KnowledgeConstrainedScorer：raw_score -> kg_score。"""
        hop = self.shortest_hop(gene, self.disease_node)
        if hop <= self.max_hop:
            kg_score = raw_score
            reason = "within_3_hop"
        else:
            overflow = min(3, hop - self.max_hop)
            factor = 1.0 - self.penalty * (overflow / 3.0)
            kg_score = raw_score * max(0.2, factor)
            reason = "beyond_3_hop_or_unreachable"
        return float(np.clip(kg_score, 0.0, 1.0)), {"hop": hop, "kg_reason": reason}


# -----------------------------
# 多智能体能力层
# -----------------------------

@dataclass
class SemanticParserAgent:
    """语义解析：规则优先，规则置信度不足时走补全分支。"""

    mutation_pattern: re.Pattern[str] = field(default_factory=lambda: re.compile(r"\b([A-Z]{2,8}[_-]?[A-Z0-9]{0,8})\b"))

    def parse(self, raw_note: str) -> Dict[str, Any]:
        note_upper = raw_note.upper()
        mutation = None
        for m in self.mutation_pattern.findall(note_upper):
            if "KRAS" in m or "BRAF" in m or "TP53" in m:
                mutation = m
                break

        dose_match = re.search(r"(\d+)\s*mg", raw_note, flags=re.IGNORECASE)
        dose = int(dose_match.group(1)) if dose_match else None
        symptom = "排卵痛" if "排卵痛" in raw_note else ("腹痛" if "腹痛" in raw_note else None)

        parsed = {
            "mutation": mutation,
            "dose_mg": dose,
            "ovary_resected": ("切除" in raw_note) or ("术后" in raw_note and "卵巢" in raw_note),
            "symptom": symptom,
            "menopause": "绝经" in raw_note,
            "tumor_stage": "III" if "III" in note_upper else ("II" if "II" in note_upper else None),
            "parser_branch": "rule",
            "parser_confidence": 0.0,
        }

        filled = sum(v is not None and v is not False for k, v in parsed.items() if k not in {"parser_branch", "parser_confidence"})
        parsed["parser_confidence"] = round(filled / 6.0, 3)

        if parsed["parser_confidence"] < 0.35:
            parsed = self._fallback_completion(parsed)
        return parsed

    def _fallback_completion(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        parsed = dict(parsed)
        parsed["parser_branch"] = "fallback_completion"
        if parsed["mutation"] is None:
            parsed["mutation"] = "UNKNOWN_DRIVER"
        parsed["parser_confidence"] = max(parsed["parser_confidence"], 0.35)
        return parsed


@dataclass
class LogicAuditAgent:
    """审计：输出警告、阻断项和风险分。"""

    def audit(self, record: Dict[str, Any]) -> Dict[str, Any]:
        warnings: List[str] = []
        blockers: List[str] = []
        risk_score = 0.0

        if record.get("ovary_resected") and record.get("symptom") == "排卵痛":
            warnings.append("病历矛盾：已切除卵巢却记录排卵痛。")
            risk_score += 0.45

        dose = record.get("dose_mg")
        if dose is not None and (dose < 10 or dose > 600):
            warnings.append("剂量异常：超出常见口服区间。")
            risk_score += 0.25

        if record.get("mutation") == "UNKNOWN_DRIVER":
            blockers.append("驱动突变未知，禁止直接给出高优先级结论。")
            risk_score += 0.25

        if record.get("parser_confidence", 0.0) < 0.4:
            warnings.append("解析置信度偏低，建议人工复核原始病历。")
            risk_score += 0.15

        return {
            "warnings": warnings,
            "blockers": blockers,
            "risk_score": float(np.clip(risk_score, 0.0, 1.0)),
            "audit_passed": len(blockers) == 0,
        }


@dataclass
class ReflectionEvolutionAgent:
    """反思进化：支持写入、更新、检索和持久化。"""

    memory: List[Dict[str, Any]] = field(default_factory=list)

    def apply_feedback_update(self, feedback: Dict[str, Any]) -> None:
        feedback = dict(feedback)
        feedback.setdefault("ts", int(time.time()))
        self.memory.append(feedback)

    def update(self, gene: str, delta: float, validated: bool, source: str = "lab") -> None:
        self.apply_feedback_update({"gene": gene, "delta": delta, "validated": validated, "source": source})

    def retrieve(self, gene: str | None = None, topk: int = 5) -> List[Dict[str, Any]]:
        items = self.memory if gene is None else [m for m in self.memory if m.get("gene") == gene]
        items = sorted(items, key=lambda x: x.get("ts", 0), reverse=True)
        return items[:topk]

    def memory_prior(self, gene: str) -> float:
        recs = self.retrieve(gene=gene, topk=10)
        if not recs:
            return 0.5
        score = 0.5
        for i, r in enumerate(recs):
            weight = 1.0 / (1 + i)
            delta = float(r.get("delta", 0.0))
            if not r.get("validated", False):
                delta *= -0.5
            score += weight * delta
        return float(np.clip(score, 0.0, 1.0))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.memory, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if path.exists():
            self.memory = json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ReasoningDecisionAgent:
    """核心决策智能体：融合模型、KG、记忆和审计风险。"""

    def reason(self, gene: str, raw_score: float, kg_score: float, memory_prior: float, audit: Dict[str, Any]) -> Dict[str, Any]:
        risk_penalty = 1.0 - 0.45 * audit["risk_score"]
        fused = (0.30 * raw_score + 0.50 * kg_score + 0.20 * memory_prior) * risk_penalty
        fused = float(np.clip(fused, 0.0, 1.0))

        if not audit["audit_passed"]:
            priority = "BLOCKED"
        elif fused >= 0.72:
            priority = "HIGH"
        elif fused >= 0.55:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        explanation = (
            f"gene={gene}; raw={raw_score:.3f}; kg={kg_score:.3f}; memory={memory_prior:.3f}; "
            f"risk={audit['risk_score']:.3f}; fused={fused:.3f}; priority={priority}"
        )
        return {"fused_score": fused, "priority": priority, "explanation": explanation}


@dataclass
class ReportGeneratorAgent:
    def generate(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)


# 兼容旧命名
KnowledgeConstrainedScorer = KnowledgeGraphConstraint
ReportAgent = ReportGeneratorAgent
ReasoningAgent = ReasoningDecisionAgent
ReflectionAgent = ReflectionEvolutionAgent
TransferMLP = TransferLearningLayer


# -----------------------------
# 总控编排
# -----------------------------

@dataclass
class YanoMatrixPipeline:
    transfer: TransferLearningLayer
    proto: PrototypicalLayer
    kg: KnowledgeGraphConstraint
    parser_agent: SemanticParserAgent
    audit_agent: LogicAuditAgent
    reason_agent: ReasoningDecisionAgent
    report_agent: ReportGeneratorAgent
    reflection_agent: ReflectionEvolutionAgent

    def train(self, x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray, y_tgt: np.ndarray) -> Dict[int, np.ndarray]:
        self.transfer.pretrain_source(x_src, y_src)
        self.transfer.finetune_target_last_layer(x_tgt, y_tgt)
        emb_tgt = self.transfer.extract_embeddings(x_tgt)
        emb_aug, y_aug = simple_smote(emb_tgt, y_tgt, target_size=max(800, len(y_tgt) * 5))
        return self.proto.fit_prototypes(emb_aug, y_aug)

    def infer_target_gene(self, prototypes: Dict[int, np.ndarray], x_query: np.ndarray, gene_name: str) -> Dict[str, Any]:
        emb_q = self.transfer.extract_embeddings(x_query)
        preds, conf = self.proto.predict(emb_q, prototypes)
        raw_score = float(np.mean(np.max(conf, axis=1)))
        kg_score, kg_meta = self.kg.gate_score(gene_name, raw_score)
        memory_prior = self.reflection_agent.memory_prior(gene_name)
        audit_stub = {"risk_score": 0.0, "audit_passed": True}
        decision = self.reason_agent.reason(gene_name, raw_score, kg_score, memory_prior, audit_stub)
        return {
            "pred_class": preds.tolist(),
            "raw_score": raw_score,
            "kg_score": kg_score,
            "memory_prior": memory_prior,
            "decision": decision,
            "kg_meta": kg_meta,
        }

    def run_agents(self, raw_text: str, analysis: Dict[str, Any], gene_name: str) -> Dict[str, Any]:
        parsed = self.parser_agent.parse(raw_text)
        audit = self.audit_agent.audit(parsed)

        memory_prior = self.reflection_agent.memory_prior(gene_name)
        final_decision = self.reason_agent.reason(
            gene=gene_name,
            raw_score=float(analysis["raw_score"]),
            kg_score=float(analysis["kg_score"]),
            memory_prior=memory_prior,
            audit=audit,
        )

        result = {
            "parsed_record": parsed,
            "audit": audit,
            "analysis": analysis,
            "decision": final_decision,
        }
        result["report"] = self.report_agent.generate(result)

        if final_decision["priority"] in {"HIGH", "MEDIUM"} and audit["audit_passed"]:
            self.reflection_agent.update(gene=gene_name, delta=+0.04, validated=True, source="auto_followup")
        else:
            self.reflection_agent.update(gene=gene_name, delta=-0.03, validated=False, source="auto_guardrail")
        result["memory_size"] = len(self.reflection_agent.memory)
        return result


YanoMatrixEngine = YanoMatrixPipeline


def build_demo_kg() -> Dict[str, List[str]]:
    return {
        "KRAS": ["MAPK_pathway", "PI3K_pathway"],
        "BRAF": ["MAPK_pathway"],
        "PIK3CA": ["PI3K_pathway"],
        "TP53": ["DNA_repair"],
        "MAPK_pathway": ["LGSOC"],
        "PI3K_pathway": ["LGSOC"],
        "DNA_repair": ["OtherCancer"],
    }


def default_pipeline() -> YanoMatrixPipeline:
    return YanoMatrixPipeline(
        transfer=TransferLearningLayer(in_dim=64, hidden=96, emb_dim=24, src_classes=4, tgt_classes=3),
        proto=PrototypicalLayer(),
        kg=KnowledgeGraphConstraint(graph=build_demo_kg(), disease_node="LGSOC", max_hop=3, penalty=0.55),
        parser_agent=SemanticParserAgent(),
        audit_agent=LogicAuditAgent(),
        reason_agent=ReasoningDecisionAgent(),
        report_agent=ReportGeneratorAgent(),
        reflection_agent=ReflectionEvolutionAgent(),
    )


def save_prototypes(path: Path, prototypes: Dict[int, np.ndarray]) -> None:
    payload = {str(k): v.tolist() for k, v in prototypes.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_prototypes(path: Path) -> Dict[int, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): np.array(v, dtype=np.float32) for k, v in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="YANO Matrix LGSOC capability pipeline")
    parser.add_argument("mode", choices=["build-kg", "train", "infer", "run-agents"], nargs="?", default="run-agents")
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--gene", default="KRAS")
    parser.add_argument("--raw-text", default="患者术后卵巢切除，KRAS突变，100mg治疗，仍诉腹痛。")
    args = parser.parse_args()

    seed_all(42)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipeline = default_pipeline()
    memory_path = outdir / "memory.json"
    pipeline.reflection_agent.load(memory_path)

    if args.mode == "build-kg":
        (outdir / "kg.json").write_text(json.dumps(build_demo_kg(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[build-kg] wrote {outdir / 'kg.json'}")
        return

    if args.mode == "train":
        x_src, y_src = make_synthetic_omics(800, 64, 4, shift=0.25)
        x_tgt, y_tgt = make_synthetic_omics(180, 64, 3, shift=0.02)
        prototypes = pipeline.train(x_src, y_src, x_tgt, y_tgt)
        pipeline.transfer.save(outdir / "model_weights.npz")
        save_prototypes(outdir / "prototypes.json", prototypes)
        np.savez(outdir / "train_bundle.npz", x_src=x_src, y_src=y_src, x_tgt=x_tgt, y_tgt=y_tgt)
        print(f"[train] wrote model_weights/prototypes/train_bundle into {outdir}")
        return

    if args.mode == "infer":
        pipeline.transfer.load(outdir / "model_weights.npz")
        prototypes = load_prototypes(outdir / "prototypes.json")
        x_query, _ = make_synthetic_omics(32, 64, 3, shift=0.05)
        analysis = pipeline.infer_target_gene(prototypes, x_query, gene_name=args.gene)
        (outdir / "infer_result.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[infer] wrote {outdir / 'infer_result.json'}")
        return

    # run-agents
    if (outdir / "infer_result.json").exists() and (outdir / "model_weights.npz").exists():
        analysis = json.loads((outdir / "infer_result.json").read_text(encoding="utf-8"))
    else:
        x_src, y_src = make_synthetic_omics(800, 64, 4, shift=0.25)
        x_tgt, y_tgt = make_synthetic_omics(180, 64, 3, shift=0.02)
        prototypes = pipeline.train(x_src, y_src, x_tgt, y_tgt)
        analysis = pipeline.infer_target_gene(prototypes, make_synthetic_omics(32, 64, 3, 0.05)[0], gene_name=args.gene)

    result = pipeline.run_agents(args.raw_text, analysis, gene_name=args.gene)
    (outdir / "agents_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pipeline.reflection_agent.save(memory_path)
    print(f"[run-agents] wrote {outdir / 'agents_result.json'} and {memory_path}")


if __name__ == "__main__":
    main()
