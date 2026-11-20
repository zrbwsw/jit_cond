# JiT 条件分支实验分析

记录日期：2026-11-19。

本记录依据当前 [exp1113.tex](exp1113.tex) 和本目录下的代码，区分已确认的实现、已有实验结果和待验证的解释。代码与实验表通过 GitHub 在本机和 H200 集群之间同步，checkpoint 通过 Hugging Face 交换。

表中 epoch 沿用 `.tex` 的报告值；注释掉的 101–221 epoch 行不作为当前结果。严格比较训练量时还应核对日志与 optimizer step，避免旧 checkpoint 的零基 epoch 索引与新格式“已完成轮数”混用。

## 1. 现有实验结果

所有结果使用表中报告的 CFG interval `[0.1,1]`；空缺的 JiT baseline 训练 loss 未补填。

| 方法 | Epoch | CFG | Train loss | FID ↓ | IS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| JiT | 80 | 2.9 | — | 8.64 | 214.0 |
| JiT | 80 | 5.1 | — | 6.77 | 218.8 |
| JiT | 600 | 3.0 | — | 3.66 | 275.1 |
| Clean patch → patch-level AdaLN | 21 | 2.9 | 0.0474 | 21.29 | 110.2 |
| Clean patch → patch-level AdaLN | 41 | 2.9 | 0.0449 | 4.82 | 192.9 |
| Clean patch → patch-level AdaLN | 61 | 2.9 | 0.0441 | 3.78 | 203.1 |
| Clean patch → patch-level AdaLN | 81 | 2.9 | 0.0437 | 3.67 | 204.5 |
| Clean CLS → block-level AdaLN | 20 | 2.9 | 0.0673 | 118.79 | 10.7 |
| Clean CLS → block-level AdaLN | 40 | 2.9 | 0.0654 | 22.42 | 72.6 |
| Clean CLS → block-level AdaLN | 60 | 2.9 | 0.0648 | 9.80 | 123.2 |
| Noisy DINO + time-conditioned projector → patch-level AdaLN | 20 | 2.9 | 0.0679 | 162.49 | 3.9 |
| Noisy DINO + time-conditioned projector → patch-level AdaLN | 40 | 2.9 | 0.0665 | 43.70 | 33.4 |
| Noisy DINO + time-conditioned projector → patch-level AdaLN | 60 | 2.9 | 0.0660 | 22.96 | 68.2 |
| Noisy DINO + time-conditioned projector → patch-level AdaLN | 80 | 2.9 | 0.0656 | 14.31 | 100.8 |

## 2. 第一项分析：clean 条件实验能否作为上限

对应代码：

- [JiT_cond](JiT_cond)：DINOv3 clean patch tokens，逐 patch AdaLN。
- [jit_cond_cls](jit_cond_cls)：DINOv3 clean CLS token，全局条件广播到各 block 的 token。

### 2.1 训练与采样实际提供了什么信息

记 `D(s)` 为包含输入预处理的冻结 DINO 编码器，`x` 为训练目标图，`epsilon` 为独立高斯噪声。当前时间约定为 `t=0` 是纯噪声，`t=1` 是 clean endpoint：

```text
z_t = t * x + (1 - t) * epsilon
g_clean = D(x)
prediction = JiT(z_t, t, y, g_clean)
```

训练时 grounding 来自目标图 `x` 本身，不是同类的另一张图。代码先提取 `D(x)`，再用同一张 `x` 构造 noisy state 和监督目标。因此，条件中包含目标样本特有的信息。见 [JiT_cond/denoiser.py](JiT_cond/denoiser.py) 的 `forward`；CLS 版本采用相同的数据流。

以 JiT-B/16、256×256 输入为例：

| 分支 | DINO 输出 | AdaLN 条件 | 空间分配 |
| --- | --- | --- | --- |
| Clean patch | `[B,256,768]` | `c[b,n] = t_embed[b] + y_embed[b] + W_g * patch[b,n]` | 每个 image patch 有自己的条件 |
| Clean CLS | `[B,768]` | `c[b] = t_embed[b] + y_embed[b] + W_g * cls[b]` | 一个样本内所有位置共用全局条件 |

Patch token 可以携带与目标图对应的局部外观和空间结构，同时 DINO 的自注意力也使它包含全局上下文；CLS 是整图信息的汇总，不提供逐位置条件。CLS 版本仍然是每个 Transformer block 用自己的 AdaLN 参数处理这个全局条件，不是不同 block 各获得一个独立 CLS token。

采样时，`grounding_source=val` 按类别和类内序号选择真实验证图：

```text
x_ref = 同类验证集真实图片
g_ref = D(x_ref)             # 采样循环前只编码一次
z_0 = random noise
整个 Euler/Heun 轨迹共用 g_ref
```

配对按文件排序固定进行；在每类 50 张验证图、共生成 50,000 张的设置下，参考条件覆盖整个验证集。起始噪声仍是随机的。见 [val_grounding.py](JiT_cond/val_grounding.py) 和 [engine_jit.py](JiT_cond/engine_jit.py)。

CFG 训练时同时丢弃类别和 grounding；采样无条件分支也同时使用 null label 和关闭的 grounding。因此，这里的 CFG 联合引导类别与参考图语义，与 baseline 仅有类别条件的路径存在差异。

### 2.2 现有结果支持的判断

Clean patch 配置显示出明显收益：61 epochs 的 FID 为 3.78，81 epochs 为 3.67，loss 也明显低于另外两种条件分支。丰富的目标特征降低了训练条件下的预测不确定性，这可以解释较低 loss，但不能直接外推为无参考图生成同样容易。

在接近的 60/61 epochs，clean patch 为 FID 3.78 / IS 203.1，clean CLS 为 9.80 / 123.2。这个比较支持当前“patch 特征＋空间调制”整套配置更有效；它同时改变了特征内容、信息量和调制空间分配，不能将全部差距单独归因于 patch-level AdaLN。两版对 in-context token 的处理也不同：patch 版 prefix 条件保持全局 `t+y`，CLS 版的全局语义条件也调制 prefix。

Clean patch 的 81-epoch FID 接近 JiT 600-epoch 的 3.66，但两者推理信息和计算预算不同，不能据此声称同任务下获得了等比例训练加速。其 IS 204.5 也没有超过 JiT 80-epoch 的 214.0，更低于 JiT 600-epoch 的 275.1。FID 改善不能直接解释为所有生成性质都改善；仅凭 IS 下降也不能断言模式坍塌或判断多样性的具体变化。

### 2.3 “上限”的适用含义

这两组可以作为 **提供真实 clean 语义时的 oracle conditioning reference**：它们检验 JiT 是否能利用理想语义条件，以及潜在收益有多大。

它们不构成未来 noisy-token 方法 FID 的严格数值界限，原因包括：

1. 采样额外得到真实参考图的实例信息，而未来方法只获得噪声、类别和模型已有的知识。
2. 当前 clean token 全轨迹固定，预测 token 随当前状态变化，采样动力学不同。
3. 有限训练、条件尺度、模型容量和 CFG 设置都会影响实测指标；额外信息不保证某次训练得到的 FID 必然优于所有替代方法。
4. FID 使用预计算的 `jit_in256_stats.npz`。当前代码不足以确认该统计来自 train 还是 val，不能仅凭文件名断言与参考图是同一集合。

因此，3.67 可作为有价值的经验参照，不能被当成未来 noisy 模块必然能接近、或必然无法超越的界限。

### 2.4 用训练期 clean 图监督 noisy 模块的可行性

训练时有 clean 图，可以建立教师目标；采样时只保留读取 noisy state 的学生：

```text
teacher target: g_clean = stop_gradient(D(x))
student input:  D(z_t), t，以及可选的类别 y
student output: g_hat_t
generation:    JiT(z_t, t, y, g_hat_t)
```

关键是生成模型在训练时也接触预测条件。若始终用真实 clean token 训练生成模型，仅在采样时替换成有误差的预测 token，会引入条件分布偏移。Clean 图可以仅用于教师监督，不需要成为学生的输入。

纯噪声端存在不可忽视的不确定性：在训练配对中，`z_0=epsilon` 与目标图 `x` 独立。即使提供 `y`，也无法唯一恢复该张配对目标图的姿态、背景和各 patch 的内容。对 clean token 做普通平方误差回归时，纯噪声端的理想解是 `E[g_clean | y]`；如果学生没有类别输入，则是跨类别的总体均值。Patch 特征的平均可能混合不同空间布局。

从噪声生成合理且有随机性的语义是可学习的；逐样本恢复随机配对图的唯一特征，在纯噪声端没有足够依据。随着 `t` 增大、图像信号增强，实例和局部特征的预测才有更多观测依据。因此应分噪声强度评估预测质量，不要求所有时间步达到同样的精度。

Clean-teacher / noisy-student 的表示监督有相关研究依据：[REPA](https://arxiv.org/abs/2410.06940) 对齐去噪网络的 noisy-input 隐藏表示与预训练编码器的 clean 表示。它支持这一监督思路，但不能直接保证“冻结 DINO(noisy image) 后接 projector”这一具体结构足够。

## 3. 第二项分析：当前 noisy PixelDiT2 projector 实验

对应代码：[JiT_cond_pixelditv2proj](JiT_cond_pixelditv2proj)。

### 3.1 Ground truth 的使用边界和输入一致性

准确的表述是：**该版本不把 ground-truth clean 图或其 token 作为 grounding 条件；训练仍使用 clean 图构造 noisy state 和生成监督，这与原版 JiT 一致。** 它也没有另外计算 `D(clean image)` 作为特征教师目标。

训练链路为：

```text
训练图 uint8
  → /255，再 *2-1，得到 x∈[-1,1]
  → z_t = t*x + (1-t)*epsilon
  ├─→ JiT 图像输入
  └─→ DINO 预处理 → 冻结 DINO → time-conditioned projector
                                  → patch-level AdaLN 条件
```

JiT-B/16、256×256 的形状为：

```text
z_t                         [B,3,256,256]
DINO patch tokens           [B,256,768]
t-conditioned projector     [B,256,768]
t_embed + y_embed           [B,768]
patch condition             [B,256,768]
```

采样入口仅为 `generate(labels)`，从高斯噪声开始。每次速度评估都把**当前同一个 `z_t`**分别送往 JiT 主干与 DINO 分支，后者使用当前 `t`。Heun 第二次评估会对 Euler predictor 得到的新状态重新编码，没有复用上一状态的 token，也不读取验证参考图。见 [denoiser.py](JiT_cond_pixelditv2proj/denoiser.py) 的 `forward`、`generate`、`_forward_sample`、`_heun_step`。

因此，它与原版 JiT 具有相同的推理外部输入种类：类别和随机噪声。DINO 分支引入的是预训练知识、额外网络与计算，没有获得目标图的额外观测。JiT 读取原始 `z_t`；DINO 读取经过固定预处理的同一 `z_t`，二者并非读取完全相同数值的张量。

还需区分“接口一致”和“分布一致”：训练的 `z_t` 由真实图和噪声插值构成，采样的 `z_t` 来自模型数值积分。模型尚不完美时，这两种状态分布不必相同；固定 DINO 对这些状态的响应是否稳定，需要实测。

### 3.2 已经做了 `(z_t+1)/2` 和 ImageNet normalization

**确实做了，训练和采样使用同一个函数。** [grounding.py](JiT_cond_pixelditv2proj/grounding.py) 的 `DINOv3GroundingEncoder.forward` 中有：

```python
x = noisy_images.float().add(1.0).mul(0.5)
x = (x - self.image_mean.to(device=x.device)) / self.image_std.to(device=x.device)
features = self.model.forward_features(x)
tokens = features["x_norm_patchtokens"]
```

所用常数是：

```python
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

完整的通道变换是：

$$
\mathcal P_c(z_t)=\frac{(z_{t,c}+1)/2-\mu_c}{\sigma_c}.
$$

训练图的基础归一化也存在：[main_jit.py](JiT_cond_pixelditv2proj/main_jit.py) 进行 center crop、随机水平翻转和 `PILToTensor`；[engine_jit.py](JiT_cond_pixelditv2proj/engine_jit.py) 的 `train_one_epoch` 再执行 `/255` 和 `*2-1`。所以这里不存在“遗漏训练图范围转换”或“遗漏 DINO ImageNet 标准化”的问题。

对于当前 B/16、256×256，DINO 需要的输入网格就是 16×16，输入保持 256×256，编码器内不额外 resize。其他 JiT patch size 则会按 `JiT grid × DINO patch size` 调整尺寸。当前不进行 clamp；只返回 patch tokens，未使用 CLS token。

### 3.3 相同的预处理公式不会把噪声变成自然图像

对 clean 图 `x∈[-1,1]`，`(x+1)/2` 的输出确实位于 `[0,1]`。但高斯噪声无界，`(z_t+1)/2` 对 noisy state 只是仿射变换，不保证输出位于 `[0,1]`。

以当前 `noise_scale=1`、`t=0` 为例，每个输入标量满足：

$$
z_0\sim\mathcal N(0,1),\qquad (z_0+1)/2\sim\mathcal N(0.5,0.25).
$$

其中约 **31.7% 的标量值**仍位于 `[0,1]` 之外；这不是图像张数的比例。再做 ImageNet 标准化后：

| 通道 | 理论均值 `(0.5-mean)/std` | 理论标准差 `0.5/std` |
| --- | ---: | ---: |
| R | 0.0655 | 2.1834 |
| G | 0.1964 | 2.2321 |
| B | 0.4178 | 2.2222 |

这些数值是纯噪声端的公式推导，不是实测 DINO 激活统计，也不意味着 clean 输入经标准化后必然恰好是零均值、单位方差。固定 ImageNet 常数不是逐图重新估计均值和方差。

给定 clean `x` 和 `t`，标准化后高斯噪声项在通道 `c` 的标准差为 `(1-t)*noise_scale/(2*std[c])`。因此，DINO 接收的数据统计与结构会随时间明显变化。

当前“不 clamp”保留了真实 noisy state 的幅度信息；clamp 会改变噪声分布并截断极值。移除 `(z_t+1)/2` 或 ImageNet normalization 也会改变 DINO 的输入尺度和均值。现有结果不能证明这些修改会改善质量，应作为独立消融评估。

能够确认的是：**预处理没有遗漏，但预处理也没有完成语义去噪或自然图分布恢复。** 冻结 DINO 是否能在各噪声区间保留有用信息，仍是待验证问题。

### 3.4 当前效果应怎样解释

在表中相同的 80 epochs、CFG=2.9 和 interval 下：

- FID 从 JiT 的 **8.64** 变为 **14.31**，增加 5.67，约 **65.6%**。
- IS 从 **214.0** 变为 **100.8**，减少 113.2，约 **52.9%**。

因此，在当前报告设置下，增加 frozen noisy-DINO + time-conditioned projector 没有取得超过原版 JiT 的生成收益。两者都是无参考图采样，这个比较比与 clean oracle 的比较更直接；同时仍需计入额外预训练模型、计算量，并确认训练、采样步数、EMA、评估统计和随机种子等实验条件。

Noisy 分支自身从 20 到 80 epochs 的 FID 持续改善：`162.49 → 43.70 → 22.96 → 14.31`；IS 为 `3.9 → 33.4 → 68.2 → 100.8`。这表明采样结果在持续改善，不能解释成训练完全没有起作用。训练 loss 仅从 `0.0679` 降至 `0.0656`，约下降 **3.4%**，但同期 FID 改善很大；平均训练 loss 不是采样质量的充分指标，不能仅凭 loss 趋平认定已经收敛。

当前还没有足够证据判断延长到 600 epochs 是否能追平 baseline，也没有 baseline 20/40/60 epochs 的配套结果来精确比较早期学习速度。

与 clean patch 的 81-epoch FID 3.67 相比，差距还包括真实参考条件、固定 token 与动态 token、projector 结构及初始化等因素，不能把差值全部归于 noisy-token 翻译能力。

### 3.5 当前训练目标并未要求“翻译成 clean token”

当前损失仅为：

```text
v_target = (x - z_t) / clamp_min(1-t, t_eps)
x_pred = JiT(z_t, t, y, projector(D(z_t), t))
v_pred = (x_pred - z_t) / clamp_min(1-t, t_eps)
loss = mean((v_pred - v_target)^2)
```

没有 `D(x)` 特征目标、token 重建损失或显式语义对齐损失。Projector 的“翻译”当前指把 noisy DINO 输出变换成 JiT 可用的条件；它可以学到任务特征，但没有被约束为 clean DINO token，也不保证输出更干净。

Projector 显式接收 `tokens, t, grid_hw`，没有类别输入 `y`；类别只在 projector 外相加。DINO 冻结，无法通过生成损失学习适配噪声。默认 projector 为 1 个 Transformer block、hidden size 768、12 heads，包含输入/输出 Linear 和时间 AdaLN。见 [pixeldit2_projector.py](JiT_cond_pixelditv2proj/pixeldit2_projector.py)。

Projector 的输出空间是 JiT 条件空间，DINO 输出空间是视觉表示空间。它们同为 768 维，也不自动具有逐维对应关系。

### 3.6 当前可提出的原因假设及其证据边界

| 已确认的实现 | 可能影响；尚未被实验定位 |
| --- | --- |
| DINO 冻结并读取不同噪声强度的图像 | 低信噪比或模型积分产生的状态可能使特征质量下降；需要按时间测量，不能直接断言 DINO 已失效 |
| 只有生成 loss，没有 clean-token 教师目标 | 联合训练需同时学条件提取与条件利用；增加监督是否有效需要实测，缺少辅助 loss 本身不构成实现错误 |
| Projector 有 `t`、没有直接的 `y` 输入 | 高噪声端不能直接利用类别先验预测语义；加入类别可能有帮助，但无法消除实例布局的不确定性 |
| Clean 线性投影全零初始化；noisy projector 输入/输出 Linear 保留默认随机初始化 | 两版初始条件尺度和优化路径不同，需要统一初始化的消融；目前没有测量支持“语义分支压过 t+y”或“激活爆炸” |
| Noisy projector 仅内部 block AdaLN 清零 | 初始残差 block 为恒等，但整个输出约为 `output_proj(input_proj(tokens))`，通常非零，且初始尚不依赖 `t` |
| CFG 的 null 分支同时关闭 grounding | 引导同时改变类别与辅助条件路径；相同 CFG 数值不代表各模型都处于最佳设置 |
| 每次采样评估都重算 token | 条件会随生成轨迹变化，误差可能反馈给后续状态；是否影响稳定性需要测量 |

如果要研究“无条件分支仍保留 noisy grounding”的 CFG 变体，应同步修改训练的 dropout 路径；不应仅在现有模型的采样阶段改变条件组合，然后将结果当成纯 CFG 系数对比。

额外计算也应记录：当前 50-step Heun 使用前 49 步两次评估、末步 Euler 一次评估，即每批生成有 **99 次 DINO 编码**。Clean-reference 分支每批只编码参考图一次。这里是调用次数推导，未测量端到端耗时比，不能直接换算成固定倍数的减速。

## 4. 下一步需要取得的证据

以下是建议的验证方向，尚未在本记录中执行或产生新结果：

1. **分时间段检查特征。** 在相同 `(x,z_t,t)` 上比较 clean DINO、noisy DINO 与学生预测表示，记录 cosine/MSE、跨样本方差和条件分支尺度。对 projector 输出做对齐时先明确特征空间或增加预测头；不能因为维度相同就直接比较。Patch 教师与学生必须对应同一裁剪、翻转和空间网格。
2. **检查特征对生成的实际作用。** 在匹配样本和模型条件下比较真实 token、预测 token、关闭 grounding 的去噪误差；离线 ablation 只用于诊断。方法收益还需由采用相应训练协议的无参考图生成 FID/IS 验证，不能仅依据特征相似度。
3. **一次控制一个因素。** 分别验证 clean-feature 辅助监督、条件门控/初始化、时间权重或注入强度，以及 CFG 设置。不要同时改变所有因素后再将改善归因于某一项。
4. **保证训练条件与部署接口一致。** 若增加 clean 教师，生成分支仍训练使用预测条件；clean 图提供监督。高噪声阶段考虑较弱全局条件、后期加强局部条件，是候选策略，不是已有结论。

一个可讨论的辅助监督形式为：

$$
\hat g_t=M_\phi(D(z_t),t,y),\qquad
\mathcal L=\mathcal L_{\mathrm{flow}}+
\lambda\,\mathbb E\left[w(t)\,d\left(H(\hat g_t),\operatorname{stopgrad}(D(x))\right)\right].
$$

`H` 表示必要时使用的特征预测头，用于明确与 DINO 教师的比较空间；条件映射与预测头可以属于同一个模块。`d`、`w(t)`、是否输入类别、梯度路径和注入门控仍需实验确定。辅助目标应允许高噪声端存在不确定性，避免把纯噪声对应的条件均值误认为完整的实例语义恢复。

## 5. 代码核对入口

| 核对项 | 文件与函数 |
| --- | --- |
| 实验数值 | [exp1113.tex](exp1113.tex) |
| Clean patch 训练与固定参考图采样 | [JiT_cond/denoiser.py](JiT_cond/denoiser.py)：`forward`、`generate` |
| 验证图配对与 FID 调用 | [JiT_cond/val_grounding.py](JiT_cond/val_grounding.py)、[JiT_cond/engine_jit.py](JiT_cond/engine_jit.py) |
| Clean CLS 特征及全局 AdaLN | [jit_cond_cls/grounding.py](jit_cond_cls/grounding.py)、[jit_cond_cls/model_jit.py](jit_cond_cls/model_jit.py) |
| Noisy 训练图范围转换 | [JiT_cond_pixelditv2proj/engine_jit.py](JiT_cond_pixelditv2proj/engine_jit.py)：`train_one_epoch` |
| Noisy DINO 预处理、冻结与 patch 输出 | [JiT_cond_pixelditv2proj/grounding.py](JiT_cond_pixelditv2proj/grounding.py)：`DINOv3GroundingEncoder.forward` |
| Noisy flow loss 与采样 | [JiT_cond_pixelditv2proj/denoiser.py](JiT_cond_pixelditv2proj/denoiser.py)：`forward`、`generate`、`_forward_sample` |
| 时间 projector 与初始化 | [JiT_cond_pixelditv2proj/pixeldit2_projector.py](JiT_cond_pixelditv2proj/pixeldit2_projector.py)：`TimestepProjection` |
| Projector 输出加入条件的位置 | [JiT_cond_pixelditv2proj/model_jit.py](JiT_cond_pixelditv2proj/model_jit.py)：`_prepare_grounding`、`forward` |
