# JiT 条件分支实验进度与表格说明

更新于 2026-11-25（进度核对以最近日志为准）。

本文对应 [exp1113.tex](exp1113.tex) 的前三张表。表格记录已经回填的实验结果；正在训练或采样但尚未回填的结果单独列在“当前实验进度”中，二者不混用。

## 当前实验进度

- **GPU1：per-file cached_recon E320 续训中。** 旧分片缓存已全部转换为每张图一个 `.npy` 文件的 v3 格式，已从 E45 恢复训练。当前日志索引为 285，即正在训练第 286 轮，最近约 0.13 秒/iter，说明 per-file 已消除旧分片随机读取的主要瓶颈。E280 在线结果为 FID 5.4851、IS 264.0797，下一次评估为 E300。
- **GPU2：pred_drop Stage 2 已完成 E200。** 最终结果为 FID 5.3077、IS 267.5210；过程中最低 FID 为 E160 的 5.2742。E160、JiT CFG 2.1 的补充采样结果为 **FID 4.74、IS 273.0**，已加入表 3。
- **noflip Stage 1 已在 E160 停止。** 最终重建验证为 MSE 0.021832、Relative L2 0.148482、Patch Cosine 0.988753，checkpoint 和日志均已保留。
- **two-stage noised 已停止。** clean-image reference（E_D=E_P=160、`t_dino_pixel=1`）结果为 FID 3.45、IS 207.5；普通 DinoDiT 采样结果为 FID 5.2038、IS 268.6。
- `JiT`、`JiT_cond` 和 `JiT_cond_cls` 主要作为表 1 的已有代码与历史结果保留。

后续新启动或续训的实验使用 `save_interval=80` 保存持久 checkpoint；`save_last_freq=5` 和在线评估频率保持不变。已经在运行的进程不因此重启。

## 表 1：不同语义条件与 AdaLN 注入方式

表 1 比较 JiT-B/16 在 clean 或 noisy 图像语义条件下的生成结果，核心变量是**条件内容**和**条件注入方式**：

- **JiT**：原始类别条件基线，不使用图像 grounding。
- **clean patch + Patch-level AdaLN**：从真实 clean 图的 DINOv3-B patch tokens 构造逐 patch 条件。采样时使用验证集真实图作为参考，因此属于 oracle/reference conditioning，用来估计理想语义条件的收益，不是无参考图生成的严格上界。
- **Class-level grounding**：每类只选一张验证图，并由该类 50 个生成样本共享，用于检验类别级参考与实例级参考的差异。
- **clean CLS + Block-level AdaLN**：使用 clean 图的全局 CLS token，而不是逐 patch token；各 Transformer block 通过 AdaLN 使用这个全局条件。
- **noisy patch + projector + Patch-level AdaLN**：从当前 noisy image 提取冻结 DINO 特征，再经 projector 构造 patch 条件。采样不读取真实参考图，但每个轨迹状态都需要重新提取条件。

表中目前较有代表性的结果包括：实例级 clean patch 在 E101 的 FID 为 3.58；class-level 对照在 E101 为 28.26；clean CLS 在 E80 为 6.53；noisy projector 分支在 E80 为 14.31。它们说明不同条件信息和空间分配会显著影响结果，但不能把差距只归因于 AdaLN 名称，因为参考图信息、token 内容和采样计算路径也同时改变。

## 表 2：DinoDiT 与 Pixel JiT 的联合条件生成

表 2 对应 `c = t_pixel + y + t_dino_pixel + W_g z_cond`，其中 `z_cond` 是 DINOv3-B/16 patch 空间中的条件。

- **联合训练组**：DinoDiT 和 Pixel JiT 一起训练，所以在线评估时 `E_D = E_P`。联合损失为 `L = L_JiT + 0.333 L_DINO`，两列 loss 分别记录，不用总 loss 替代。
- **D_heun25 / P_heun50**：分别表示 DinoDiT 的 25 步 Heun 采样和 Pixel JiT 的 50 步 Heun 采样；每列同时给出 CFG scale 与 interval。
- **E80 的 CFG 对照**：同一个联合训练 checkpoint 下，JiT CFG 2.9 时 FID 为 5.35，CFG 2.3 时为 5.21。空白的 epoch 和 loss 表示沿用同组上一行，不是缺失模型。
- **clean-image dino patch tokens 行**：不经过 DinoDiT 采样，直接使用真实验证图的 DINO patch tokens，并将 `t_dino_pixel` 固定为 1，即 `c = t_pixel + y + 1 + W_g z_cond`。此前 JoinDino E80 参考结果为 FID 3.65；noised E_D=160、E_P=160 的结果为 **FID 3.45、IS 207.5**。这是真实 clean 语义参考，用来衡量 DinoDiT 条件误差带来的差距，不能与采样条件行当作完全相同的推理设置。
- **冻结 DinoDiT E160 → Pixel E80 组**：DinoDiT 与 Pixel 的训练量不再相等；Pixel 阶段使用冻结的 E160 DinoDiT。JiT CFG 2.9 时 FID 为 5.28，最佳记录的 CFG 2.1 为 4.91。这里 `loss_jit` 是 Stage 2 的 Pixel loss，`loss_dino` 是 Stage 1 的历史参考值。

表 2 的重点是联合训练能否让 DinoDiT 提供适合 Pixel JiT 的条件，以及更充分的 DinoDiT 训练和 CFG 调整能带来多少收益。

## 表 3：冻结 DinoDiT 的单步预测条件

表 3 对应独立两阶段训练：先训练并冻结 DinoDiT，再训练 Pixel JiT。

训练时，对 clean latent `z` 按 `t_dino_pixel` 加噪，并使用冻结 DinoDiT 做一次单步预测得到 `z_cond`；不是直接把加噪后的 clean latent 当作条件。Pixel JiT 只优化 `L_JiT`，所以早期 E_D = 80 组的 `loss_dino` 记为 `--`。推理时仍使用同一个冻结 DinoDiT checkpoint 完成 DINO Heun-25 采样，再进行 Pixel Heun-50 采样。

- **E_D = 80 组**：Pixel 从 E20 到 E80 逐步改善；Pixel E80 时，JiT CFG 2.9 的 FID 为 5.58，CFG 2.3 为 5.18。
- **E_D = 160 组**：更强的冻结 DinoDiT 用于训练和推理；Pixel E80 时，JiT CFG 2.9 的 FID 为 5.43，最佳记录的 CFG 2.1 为 4.78。Pixel E160 时，CFG 2.9 的 FID 为 5.25，CFG 2.1 为 4.71（IS 272.6）。`loss_dino = 0.3687` 是该 Stage 1 checkpoint 的历史 loss。另有 clean-image reference（E_D=E_P=160、`t_dino_pixel=1`）结果 **FID 4.85、IS 193.3**；由于 pred 训练使用单步预测 token，该 clean-reference 不是严格上界。
- **pred_drop（P1 修复）**：训练时对传给 DinoDiT 的类别标签独立执行 dropout，使 Pixel 同时见过 DinoDiT 的有条件和无条件 token。E_D=E_P=160、DINO CFG 2.0、JiT CFG 2.1 时为 **FID 4.74、IS 273.0**；旧 pred 在相同训练轮数和 CFG 下为 FID 4.71、IS 272.6。FID 仅相差 0.03，IS 相差 0.4，说明修复后结果与修复前基本相当：P1 解决了训练—推理条件分布不一致，但没有带来显著的生成指标改善。

表 3 与表 2 的主要区别是：表 3 在 Pixel 训练时使用**冻结 DinoDiT 的单步预测条件**；表 2 的主体是联合训练，其后续 noised 组使用直接加噪条件。比较两组时应同时核对 `E_D`、`E_P`、条件生成方式和两边的 CFG，而不能只比较最佳 FID。

## 表格阅读约定与代码入口

- Epoch 表示已经完成的训练轮数；`E_D` 和 `E_P` 分别对应 DinoDiT 与 Pixel JiT。
- FID 越低越好，IS 越高越好；除特别说明外，生成指标基于 50,000 张图。
- 同一 checkpoint 的多组 CFG 采样只在第一行保留 epoch 和 loss；`.tex` 中注释掉的行不算当前已采用结果。
- [JiT](JiT)：原始 JiT 基线。
- [JiT_cond](JiT_cond)：clean DINO patch tokens 与 Patch-level AdaLN。
- [JiT_cond_cls](JiT_cond_cls)：clean DINO CLS token 与 Block-level AdaLN。
- [jit_cond_dino_two_stage_1122](jit_cond_dino_two_stage_1122)：冻结 DinoDiT 的 noised / pred 两阶段 Pixel 训练。
- [jit_cond_dino_two_stage_1122_pred_drop](jit_cond_dino_two_stage_1122_pred_drop)：独立 DinoDiT label dropout 的 pred 修复版。
- [jit_cond_dino_two_stage_1122_ddim_recon_noflip](jit_cond_dino_two_stage_1122_ddim_recon_noflip)：带重建约束、无水平翻转的 Stage 1，以及对应的 latent 导出流程。
- [jit_cond_dino_two_stage_1122_ddim_recon_noflip_perfile](jit_cond_dino_two_stage_1122_ddim_recon_noflip_perfile)：per-file 缓存转换、校验与 cached_recon 续训。

表 4 的 B/L 模型对比和表 5 的相关工作汇总不属于本文这里的“表 1、2、3”说明范围。
