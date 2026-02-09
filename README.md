# 123

LGSOC 文件驱动诊断流水线：`yano_matrix_lgsoc_demo.py`

## 运行

```bash
python yano_matrix_lgsoc_demo.py \
  --raw-note-file <path_or_url_to_txt> \
  --candidates-file <path_or_url_to_json> \
  --kg-file <path_or_url_to_json> \
  --report-out report_out/report.md
```

说明：
- 所有输入文件会先进行一次联网识别（HTTP 调用）再读取。
- 支持本地路径与 HTTP/HTTPS URL。
