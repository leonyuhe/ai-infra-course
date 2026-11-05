# 修订日志（本地阶段）

> 工作模式：**修订期间只在本地改与提交，不 push GitHub、不重建/上传 PDF**
> （README/站点/Release 全部保持最后同步状态）。等全书修订完成后一次性发布。

## 当前同步状态（remote）

- 最后推送 commit：`7b17e37`（ch01 的 prefill/decode 重写、ch05 措辞、站点 CI 提示条）
- 在线站与 Release PDF（227 页）已包含到该 commit 为止的内容
- 之后的改动都只在本地

## 待发布清单

| 日期 | 文件 | 改动 | 状态 |
| --- | --- | --- | --- |
| 2026-11-05 | `notebooks/01_forward_and_sampling.py` | 01.1 重写：并行/串行、对照表、诊断口诀、两个边界条件；带宽段硬件自适应 | 已推送（7b17e37） |
| 2026-11-05 | `notebooks/05_scheduler.py` | decode 与 batch 的措辞对齐 | 已推送（7b17e37） |
| 2026-11-05 | `tools/build_site.py` | 章节页 CI 执行提示条 | 已推送（7b17e37） |
| 2026-11-05 | `REVISION_LOG.md` | 建立修订日志（本文件） | 本地 |
| 2026-11-05 | `tools/build_site.py`、`.github/workflows/pages.yml` | 新增 `--ci-banner` 开关：本地预览不显示 CI 提示，线上构建才显示 | 本地 |
| 2026-11-05 | `notebooks/01_forward_and_sampling.py`、`assets/figures/ch01_pipeline.svg` | 01.2 重写：全貌流水线图、参数账本（2,753,792 × 4 = 11,015,168 B ≈ 10.5 MiB）、前向形状追踪、attention/SDPA 讲解、采样整体逻辑、与带宽/算力的关系 | 本地 |
| 2026-11-05 | `notebooks/01_forward_and_sampling.py` | 实测 2 重写：prefill 三阶段（固定开销/算力屋顶/attention 反噬），tiny vs wide 双模型对照，加入 attention 分数矩阵大小列；说明 14800 这类超长 T 无参考价值 | 本地 |
|  |  |  |  |

## 一次性发布 checklist（全部修订完成后）

1. `make all`：build + check + 15 章无头执行全绿
2. `python tools/build_site.py --out _site --pdf dist/ai-infra-course-v0.1.1.pdf`
   - 如需升版本号：先改 `tools/make_cover.py` 里的 `v0.1.0` → `v0.1.1` 并重新生成封面
3. `git push`（触发 CI + Pages 重建）
4. `gh release create v0.1.1 <pdf> --title ... --notes-file ...`
5. 验证：在线站、PDF 首页/第 11 页、Release 资产大小
6. 更新 README 里的版本号/链接（如有）
