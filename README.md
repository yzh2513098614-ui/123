# 123

LGSOC 神经符号 AI 示例代码：`yano_matrix_lgsoc.py`（保留 `yano_matrix_lgsoc_demo.py` 作为最小演示版）。

## 运行

```bash
python yano_matrix_lgsoc.py train
python yano_matrix_lgsoc.py run-agents
python yano_matrix_lgsoc.py build-kg --literature-dir ./papers --report-every 5
```

常用开关：

- `--progress on/off`：统一进度显示（默认 `on`）
- `--verbose`：输出详细诊断

脚本包含：迁移学习、原型网络小样本学习、知识图谱约束门控、多智能体流程示例。

## 进度条与诊断输出示例

以下为真实终端风格样例（非 TTY 时自动降级为纯文本百分比）：

```text
$ python yano_matrix_lgsoc.py --verbose train
[diag] progress_mode=text tty=False
[pretrain_source]  10% (2/20)
[pretrain_source] loss=1.4274 best=1.4274
...
[finetune_target_classifier] 100% (15/15)
[finetune_target_classifier] loss=1.0710 val_auc=0.6124 best=0.6413
{
  "prototype_classes": [0, 1, 2],
  "count": 3
}

$ python yano_matrix_lgsoc.py run-agents
[解析] 成功 | 耗时 0.000s
[审查] 告警 | 耗时 0.000s
[推理] 成功 | 耗时 0.000s
[反馈] 成功 | 耗时 0.000s
[报告] 成功 | 耗时 0.000s
[记忆更新] 成功 | 耗时 0.000s
```
