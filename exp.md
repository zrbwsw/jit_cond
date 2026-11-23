# JiT 条件分支实验进度与表格说明

更新于 2026-11-23（进度核对时间约为 11:15 UTC）。

本文对应 [exp1113.tex](exp1113.tex) 的前三张表。表格记录已经回填的实验结果；正在训练或采样但尚未回填的结果单独列在“当前实验进度”中，二者不混用。

## 当前实验进度

- **GPU1：noflip E80 训练集 latent 导出中。** 使用 8 卡，将 `stage1-dinov3b-dinodit-b-recon-noflip-e200/checkpoint-80.pth` 的 EMA1 权重用于 ImageNet train 重建特征导出。当前完成 1,077 / 10,009 个分片，即 137,856 / 1,281,167 张图，约 10.8%。最终 `manifest.json` 尚未生成，因此缓存还不能作为完整数据集使用。
- **GPU2：two-stage pred 训练中。** 使用冻结的 DinoDiT E160，Pixel JiT 目标为 200 epoch；日志索引为 153，即正在训练第 154 轮。最近一次已完成的在线评估是 E140，FID 5.2553、IS 263.0。表 3 目前只回填到 Pixel E80，后续结果尚未写入。
- **GPU2：noflip Stage 1 续训中。** 从实际的 E115 checkpoint 恢复，目标为 200 epoch；日志索引为 116，即正在训练第 117 轮。此前 E120 重建验证只汇总到 31,250 / 50,000 张并发生集合通信超时，续训版本将通信超时提高到 60 分钟，并增加各 rank 的验证计数日志和同步。是否通过原故障点仍需等待新的 E120 验证。
- **two-stage noised 已停止。** 停止前完成 E160 在线评估，FID 5.2038、IS 268.6；该结果尚未作为新一行回填到表 2。
- `JiT`、`JiT_cond` 和 `JiT_cond_cls` 当前主要作为表 1 的已有代码与历史结果保留，没有在本进度核对中启动新训练。

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
- **clean-image dino patch tokens 行**：不经过 DinoDiT 采样，直接使用真实验证图的 DINO patch tokens，FID 为 3.65。这是真实 clean 语义参考，用来衡量 DinoDiT 条件误差带来的差距，不能与采样条件行当作完全相同的推理设置。
- **冻结 DinoDiT E160 → Pixel E80 组**：DinoDiT 与 Pixel 的训练量不再相等；Pixel 阶段使用冻结的 E160 DinoDiT。JiT CFG 2.9 时 FID 为 5.28，最佳记录的 CFG 2.1 为 4.91。这里 `loss_jit` 是 Stage 2 的 Pixel loss，`loss_dino` 是 Stage 1 的历史参考值。

表 2 的重点是联合训练能否让 DinoDiT 提供适合 Pixel JiT 的条件，以及更充分的 DinoDiT 训练和 CFG 调整能带来多少收益。

## 表 3：冻结 DinoDiT 的单步预测条件

表 3 对应独立两阶段训练：先训练并冻结 DinoDiT，再训练 Pixel JiT。

训练时，对 clean latent `z` 按 `t_dino_pixel` 加噪，并使用冻结 DinoDiT 做一次单步预测得到 `z_cond`；不是直接把加噪后的 clean latent 当作条件。Pixel JiT 只优化 `L_JiT`，所以早期 E_D = 80 组的 `loss_dino` 记为 `--`。推理时仍使用同一个冻结 DinoDiT checkpoint 完成 DINO Heun-25 采样，再进行 Pixel Heun-50 采样。

- **E_D = 80 组**：Pixel 从 E20 到 E80 逐步改善；Pixel E80 时，JiT CFG 2.9 的 FID 为 5.58，CFG 2.3 为 5.18。
- **E_D = 160 组**：更强的冻结 DinoDiT 用于训练和推理；Pixel E80 时，JiT CFG 2.9 的 FID 为 5.43，最佳记录的 CFG 2.1 为 4.78。`loss_dino = 0.3687` 是该 Stage 1 checkpoint 的历史 loss。

表 3 与表 2 的主要区别是：表 3 在 Pixel 训练时使用**冻结 DinoDiT 的单步预测条件**；表 2 的主体是联合训练，其后续 noised 组使用直接加噪条件。比较两组时应同时核对 `E_D`、`E_P`、条件生成方式和两边的 CFG，而不能只比较最佳 FID。

## 表格阅读约定与代码入口

- Epoch 表示已经完成的训练轮数；`E_D` 和 `E_P` 分别对应 DinoDiT 与 Pixel JiT。
- FID 越低越好，IS 越高越好；除特别说明外，生成指标基于 50,000 张图。
- 同一 checkpoint 的多组 CFG 采样只在第一行保留 epoch 和 loss；`.tex` 中注释掉的行不算当前已采用结果。
- [JiT](JiT)：原始 JiT 基线。
- [JiT_cond](JiT_cond)：clean DINO patch tokens 与 Patch-level AdaLN。
- [JiT_cond_cls](JiT_cond_cls)：clean DINO CLS token 与 Block-level AdaLN。
- [jit_cond_dino_two_stage_1122](jit_cond_dino_two_stage_1122)：冻结 DinoDiT 的 noised / pred 两阶段 Pixel 训练。
- [jit_cond_dino_two_stage_1122_ddim_recon_noflip](jit_cond_dino_two_stage_1122_ddim_recon_noflip)：带重建约束、无水平翻转的 Stage 1，以及对应的 latent 导出流程。

表 4 的 B/L 模型对比和表 5 的相关工作汇总不属于本文这里的“表 1、2、3”说明范围。
