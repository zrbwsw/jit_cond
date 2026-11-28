# PixelCond 实验结果与表格说明

更新于 2026-11-28。本文按顺序对应 [exp1113.tex](exp1113.tex) 的 **Table 1–8**，记录已经由日志确认的实验结果。FID 越低越好，IS 越高越好；除特别说明外，指标均来自 50,000 张生成图片。

空白项表示尚未形成有效的对应指标，不用训练中的估计值代替。不同表的语义 encoder、条件构造、训练轮数、采样器和 CFG 不完全相同，比较时必须先核对这些设置。

## 最新结果摘要

- **RAEv2 + JiT-B noised**：E80、Pixel CFG 2.0 时当前最佳为 **FID 3.2272、IS 233.6044**，表中写为 3.23 / 233.6。有效采样明确使用 `condition_mode_pixel=noised`；此前误用 clean 模式的失败采样不计入结果。
- **RAEv2 + JiT-B pred**：目前有 E20、E40 指标，E40 为 **FID 4.25、IS 223.6**。E60、E80 尚无有效生成指标，保持空白。
- **RAEv2 + PixelDiT2-H/16**：E60 为 **FID 5.52、IS 197.3**。同 epoch 下，clean JiT-B 为 3.98 / 230.0，noised JiT-B 为 3.78 / 229.9，因此当前 H 的 FID 更高、IS 更低，**60 epoch 的 H 明显不如 B**。E80 指标尚未形成，不能提前判断最终结果。
- 表 4–6 的 `_no_tdinopixel` 版本统一使用 `cond = E_pixel(t_pixel) + E_y(y) + W_g(grounding_tokens)`。`t_dino_pixel` 只用于训练时的 latent corruption 或 pred teacher 调用，不再作为额外 Pixel embedding 注入。

## Table 1：语义内容与 AdaLN 注入方式

Table 1 是早期 JiT-B/16 条件分支对照，核心变量是语义内容和注入方式：

- 原始 JiT 基线：E80、CFG 5.1 时 FID 6.77；E600 时 FID 3.66。
- clean patch + Patch-level AdaLN：E101、使用真实参考图时为 FID 3.58，属于 oracle/reference conditioning，不是自由生成结果。
- class-level grounding：每类共享一张验证图，E101 为 FID 28.26，明显差于实例级参考。
- clean CLS + Block-level AdaLN：E80 为 FID 6.53。
- noisy image → frozen DINO → projector：E80 为 FID 14.31，采样时每个轨迹状态都需重新提取条件。

这些结果说明语义的空间粒度、参考图信息和注入方式都会影响生成，但不能把差距只归因于 AdaLN 的名称。

## Table 2：DinoDiT noised / 联合与两阶段 Pixel JiT

Table 2 使用 DINOv3-B/16 patch latent，主体条件为：

`c = t_pixel + y + t_dino_pixel + W_g z_cond`。

主要结果：

- 联合训练 E80：JiT CFG 2.9 时 FID 5.35，CFG 2.3 时 5.21。
- 冻结 DinoDiT E160 → Pixel E80：CFG 2.9 时 FID 5.28。
- 同组 Pixel E160：CFG 2.9 时 FID 5.13；DINO CFG 1.8、JiT CFG 2.1 时 FID 4.76。
- clean-val reference：FID 3.45；clean-train reference：FID 2.28。它们是使用真实图像语义的参考结果，与 DinoDiT 自由采样条件不是同一种推理设置。
- noised clean-train：随机 `t_dino_pixel` 时 FID 2.36，固定 0.75 时为 2.49。

该表用于衡量 DinoDiT 条件误差以及真实语义参考与自由生成之间的差距。

## Table 3：冻结 DinoDiT 的单步 pred 条件

Table 3 在训练时先对 clean latent 加噪，再由冻结 DinoDiT 做一次单步预测，Pixel 使用预测后的 tokens 和 `t_dino=1`。Inference 仍由冻结 DinoDiT 完成多步语义采样，再执行 Pixel sampling。

- DinoDiT E80 → Pixel E80：JiT CFG 2.9 时 FID 5.58，CFG 2.3 时 5.18。
- DinoDiT E160 → Pixel E80：CFG 2.9 时 FID 5.43。
- DinoDiT E160 → Pixel E160：CFG 2.9 时 FID 5.25；DINO CFG 1.8、JiT CFG 2.1 时 FID 4.62；DINO Heun-50 后进一步达到 FID 4.32。
- clean-val reference：FID 4.85；clean-train reference：FID 3.52。
- pred clean-train：随机 corruption time 时 FID 2.51，固定 0.75 时为 2.39。

Table 3 与 Table 2 的关键区别是：这里训练注入的是 DinoDiT 的单步预测，而 Table 2 的 noised 主体直接使用加噪 latent。

## Table 4：RAEv2 clean latent + JiT-B/16

Table 4 使用 RAEv2 normalized DINOv3-L k=7 clean patch tokens。Inference 改由冻结的 RAEv2 DDT-XL EMA 采样 normalized latent，再执行 Pixel Heun-50。

| Pixel epoch | 采样设置 | FID | IS |
| --- | --- | ---: | ---: |
| 40 | JiT CFG 2.9 | 4.54 | 224.2 |
| 60 | JiT CFG 2.9 | 3.98 | 230.0 |
| 80 | JiT CFG 2.9 | 3.88 | 231.7 |
| 80 | JiT CFG 2.0 | 3.50 | 231.9 |
| 160 | JiT CFG 2.0 | 3.38 | 232.1 |
| 240 | JiT CFG 2.0 | **3.28** | **233.8** |

该组是后续 RAEv2 noised、pred 以及 PixelDiT2-H 的 clean-B 对照。E60 的 3.98 / 230.0 也用于当前 H 的同 epoch 比较。

## Table 5：RAEv2 noised latent + JiT-B/16（无 DINO time embedding）

Training 使用：

`z_cond = t_dino_pixel * clean + (1 - t_dino_pixel) * noise`，其中 `t_dino_pixel ~ U[0.75,1)`。

Pixel 只接收 `E_pixel(t_pixel) + E_y(y) + W_g(z_cond)`，不额外接收 `t_dino_pixel` embedding。Inference 使用完整 RAEv2 sampler 生成的 latent。

| Pixel epoch | 采样设置 | FID | IS |
| --- | --- | ---: | ---: |
| 20 | JiT CFG 2.9 | 14.91 | 147.2 |
| 40 | JiT CFG 2.9 | 4.39 | 223.9 |
| 60 | JiT CFG 2.9 | 3.78 | 229.9 |
| 80 | JiT CFG 2.9 | 3.64 | 232.7 |
| 80 | JiT CFG 2.0 | **3.23** | **233.6** |

更精确的 E80、CFG 2.0 日志值为 FID 3.2272、IS 233.6044。当前结果说明 noised 训练没有破坏自由生成，且在 E80 达到了略优于 clean-B E240 的 FID；但训练轮数不同，不能据此直接断言 noised 一定优于 clean。

## Table 6：RAEv2 DDT 单步 pred + JiT-B/16（无 DINO time embedding）

Training 先按 Table 5 的方式构造 noised latent，再由冻结的 RAEv2 DDT-XL 做一次 clean x-prediction。RAEv2 的时间方向为 `x_t=(1-t)*clean+t*noise`，因此 DDT 接收 `t_rae=1-t_dino_pixel`。Pixel 只接收预测 tokens，不接收额外 DINO 时间 embedding。

| Pixel epoch | 采样设置 | FID | IS |
| --- | --- | ---: | ---: |
| 20 | JiT CFG 2.9 | 15.71 | 152.1 |
| 40 | JiT CFG 2.9 | 4.25 | 223.6 |
| 60 | 尚无有效指标 | — | — |
| 80 | 尚无有效指标 | — | — |

E40 的 pred FID 略优于同 epoch noised 的 4.39，IS 则非常接近（223.6 对 223.9）。目前只能说明早期趋势，E60/E80 未完成前不能与 Table 5 的最终结果直接比较。

## Table 7：RAEv2 clean latent + PixelDiT2-H/16 + REPA

该组保留 PixelDiT2-H 主干和 REPA，损失为 `L = L_fm + 0.5 * L_cos`。Pixel sampling 使用 Heun-50、CFG 2.4、interval `[0.125,0.9]`。

| Pixel epoch | fm loss | cos loss | FID | IS |
| --- | ---: | ---: | ---: | ---: |
| 20 | 0.0293 | 0.0960 | 10.61 | 162.4 |
| 40 | 0.0269 | 0.0751 | 6.94 | 183.6 |
| 60 | 0.0260 | 0.0677 | 5.52 | 197.3 |
| 80 | 尚无有效指标 | — | — | — |

**需要特别注意：当前 E60 的 H 不如 B。**

- H E60：FID 5.52、IS 197.3；
- clean-B E60：FID 3.98、IS 230.0；
- noised-B E60：FID 3.78、IS 229.9。

因此 H 在相同 epoch 下 FID 分别高 1.54 和约 1.74，IS 低约 33。虽然 H 的 FID 仍在随训练下降，但目前没有表现出模型规模优势。该比较同时存在模型容量、REPA、Pixel CFG 和训练配置差异，不能仅凭这一项指标归因于某一个因素；E80 之后仍需按同一评估协议复核。

## Table 8：Cached reconstruction Pixel JiT

Table 8 使用离线 25-step Heun-reconstructed DINO patch tokens 作为 cached conditions：

- E_D=80、E_P=80、JiT CFG 2.9：FID 5.59、IS 259.8。
- 同 checkpoint、JiT CFG 2.1：FID 5.43、IS 260.6。

该组用于评估缓存重建语义的有效性，与 RAEv2 系列使用的 latent space 和采样路径不同，不与 Table 4–7 直接混排排名。

## 表格阅读与结果使用约定

- `E_D`、`E_P` 分别表示语义模型和 Pixel 模型的训练轮数；只有单一 Pixel 训练阶段时重点看 `E_P`。
- 同一 checkpoint 的多组 CFG 结果不能当作不同训练模型；表格中的最佳值也必须同时保留 CFG 和 interval。
- clean-val / clean-train reference 使用真实图片语义，不能与自由生成行直接视为相同任务。
- `noised` 是直接注入加噪 latent；`pred` 是注入冻结语义模型的一步预测；两者 Inference 都使用完整语义采样结果。
- `_no_tdinopixel` 表示不注入额外的 DINO 时间 embedding，并不表示取消 noised/pred 所需的 corruption time。
- 当前 H 的结论仅限于已完成的 E60 指标；pred-B 的结论仅限于 E20/E40，均不外推。
- `.tex` 在前述 Table 1–8 之后还保留 B/L comparison 和 related work 两张参考附表；它们不计入本文整理的前 8 张实验表，也未在本次更新中删除。

主要代码入口：

- [jit_cond_dino_two_stage_1122](jit_cond_dino_two_stage_1122)：DinoDiT noised / pred。
- [jit_cond_dino_two_stage_1122_raev2](jit_cond_dino_two_stage_1122_raev2)：RAEv2 clean / noised / pred。
- [jit_cond_dino_two_stage_1122_raev2_no_tdinopixel](jit_cond_dino_two_stage_1122_raev2_no_tdinopixel)：无额外 DINO time embedding 的 RAEv2 JiT-B 变体。
- [PixelDiTv2_cond_RAEv2](PixelDiTv2_cond_RAEv2)：PixelDiT2-H + RAEv2 grounding + REPA。
- [jit_cond_dino_two_stage_1122_ddim_recon_noflip_perfile](jit_cond_dino_two_stage_1122_ddim_recon_noflip_perfile)：per-file cached reconstruction。
