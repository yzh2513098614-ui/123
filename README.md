# YANO

LGSOC 神经符号 AI 示例代码：`yano_matrix_lgsoc_demo.py`

## 运行

```bash
python yano_matrix_lgsoc_demo.py
```

脚本包含：迁移学习、原型网络小样本学习、知识图谱约束门控、多智能体流程示例。

## Converted R blocks to Python (真实文件运行)

`lgsoc_r_to_py_blocks.py` 已改为优先读取你原始 R 代码中的路径/文件名进行执行：

- 默认工作目录：`L:/BaiduNetdiskDownload/半城生信`
- 默认 eQTL：`blood.txt`
- 默认基因列表：`gene2.csv`
- 默认 GWAS：`gwas2.tsv`
- 默认 PLINK：`L:/BaiduNetdiskDownload/半城生信/plink_win64/plink.exe`
- 默认 bfile：`L:/bio_data/data_maf0.01_rs_ref/data_maf0.01_rs_ref`

运行（可按需覆盖路径参数）：

```bash
python lgsoc_r_to_py_blocks.py \
  --workdir "L:/BaiduNetdiskDownload/半城生信" \
  --plink-exe "L:/BaiduNetdiskDownload/半城生信/plink_win64/plink.exe" \
  --bfile "L:/bio_data/data_maf0.01_rs_ref/data_maf0.01_rs_ref" \
  --gene-file "gene2.csv" \
  --eqtl-file "blood.txt" \
  --gwas-file "gwas2.tsv" \
  --outdir "py_real_outputs"
```

会输出：
- `Intersected_Significant_eQTL_Raw.csv`
- `Supplementary_Table_2_Final.csv`
- `Supplementary_Table_4_Final_MR.csv`
- `MR_Final_Manhattan_Plot.png`
- `MR_Summary_Forest_Plot_Fixed.png`
- `MR_Volcano_Plot_Fixed.png`
- `Top50_Genes_for_STRING.txt`
- `workflow_completeness.json`
- `manifest.json`
