import torch
import torch.nn as nn
from model_jit import JiT_models
from grounding import DINOv3GroundingEncoder


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        # use_grounding 是 Python/API 名称，enable_grounding 是启动脚本的兼容
        # 别名；打开后 JiT 会接收 clean image 的 DINOv3 patch token。
        use_grounding = bool(getattr(args, "use_grounding", getattr(args, "enable_grounding", False)))
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            grounding_dim=(getattr(args, "grounding_dim", 768) if use_grounding else None),
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        self.use_grounding = use_grounding

        # 采样时的语义 token 来源。本实验要验证的是"始终有干净语义"的
        # 上限，因此默认取验证集同类真实图片；probe 只在拿不到参考图时
        # 才用（它需要被 ground_drop_prob 单独训练覆盖），none 即原始 t+y。
        source = getattr(args, "grounding_source", None)
        if source is None:  # 兼容旧脚本的 --auto_grounding
            source = "probe" if getattr(args, "auto_grounding", None) else "none"
        self.grounding_source = source
        self.auto_grounding = source == "probe"
        self.ground_drop_prob = float(getattr(args, "ground_drop_prob", 0.0))
        self._val_provider = None  # 懒加载：训练阶段不读验证集

        # DINOv3 是冻结的语义提取器，独立于 JiT 网络保存，扩散 loss 不会
        # 更新它的参数。
        if self.use_grounding:
            self.grounding_encoder = DINOv3GroundingEncoder(
                model_name=getattr(args, "grounding_model", "dinov3_vitb16"),
                weights_path=getattr(args, "grounding_weights", None) or None,
                repo=getattr(args, "grounding_repo", None) or None,
            )
            self.grounding_encoder.eval()
        else:
            self.grounding_encoder = None

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def drop_labels_with_mask(self, labels):
        """返回 (dropped_labels, keep_mask)。

        ``keep_mask`` 为 False 的样本是 CFG 的无条件样本；它们的语义条件
        也必须一并关闭，否则干净图的 DINO token 会把类别信息泄漏进无条件
        分支，模型永远学不到真正的 null 条件。
        """
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out, ~drop

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    @torch.no_grad()
    def encode_grounding(self, clean_images):
        if not self.use_grounding or self.grounding_encoder is None:
            return None
        # 输入仍是 [-1,1]；grounding.py 会做 ImageNet 标准化并返回
        # [B,N,768] patch token。
        return self.grounding_encoder(clean_images)

    def forward(self, x, labels, clean_images=None, grounding_tokens=None, grounding_mask=None):
        """计算 flow loss；可选地加入 clean image 的 grounding。

        ``x`` 是原始 JiT API 中的 clean training image；显式传入
        ``clean_images`` 或 ``grounding_tokens`` 可支持缓存 DINO token，
        同时不改变原有的加噪路径。

        ``grounding_mask`` 是 [B] 的 bool，默认由两部分合成：

        * label 被丢掉的样本（CFG 无条件样本）同时关闭语义条件，否则
          干净图的 DINO token 会把类别信息泄漏进无条件分支；
        * ``ground_drop_prob`` 的样本保留 label、只关闭语义条件，用来训练
          ``grounding_source=probe`` 采样时用到的那条纯 t+y 路径。

        语义条件被关闭时 ``patch_condition`` 退化为广播后的全局 t+y。
        """
        # 训练时从干净目标图像取 token，而不是从加噪 z 取 token，
        # 这样验证的是 clean semantic grounding 的作用。
        if not self.use_grounding:
            grounding_tokens = None
            grounding_mask = None
        elif grounding_tokens is None:
            grounding_tokens = self.encode_grounding(x if clean_images is None else clean_images)

        if self.training:
            labels_dropped, keep_mask = self.drop_labels_with_mask(labels)
            if self.ground_drop_prob > 0:
                # 保留 label，只额外关掉部分样本的语义条件。
                ground_keep = torch.rand(labels.shape[0], device=labels.device) >= self.ground_drop_prob
                keep_mask = keep_mask & ground_keep
        else:
            labels_dropped, keep_mask = labels, None

        if grounding_tokens is not None and grounding_mask is None:
            grounding_mask = keep_mask

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(
            z, t.flatten(), labels_dropped,
            grounding_tokens=grounding_tokens, grounding_mask=grounding_mask,
        )
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    def _provider(self, data_path):
        """懒加载验证集 provider；只有 grounding_source=val 时才需要。"""
        if self._val_provider is None:
            from val_grounding import ValGroundingProvider
            self._val_provider = ValGroundingProvider(
                root=data_path, img_size=self.img_size, num_classes=self.num_classes
            )
        return self._val_provider

    @torch.no_grad()
    def generate(self, labels, clean_images=None, grounding_tokens=None,
                 intra_indices=None, data_path=None, grounding_source=None):
        """生成 class-conditional 样本。

        ``grounding_source`` 决定语义 token 从哪来：

        * ``val``   —— 验证集同类真实图片的 DINO token。这是本实验的设定：
          假设采样时始终有干净语义，用来测 patch-AdaLN 的上限。
        * ``probe`` —— 先用 t+y 路径估计 x_hat，再编码它的 token；没有参考
          图时的兜底，需要训练时用 ``ground_drop_prob`` 覆盖这条路径。
        * ``none``  —— 原始 t+y 采样，用于对照。

        也可以直接传入 ``clean_images`` / ``grounding_tokens`` 覆盖上述逻辑。
        """
        device = labels.device
        bsz = labels.size(0)
        source = self.grounding_source if grounding_source is None else grounding_source

        if not self.use_grounding:
            grounding_tokens, source = None, "none"
        if grounding_tokens is None and clean_images is not None:
            grounding_tokens = self.encode_grounding(clean_images)
        if grounding_tokens is None and source == "val":
            if intra_indices is None:
                raise ValueError(
                    "grounding_source=val 需要 intra_indices（每个样本在其类别验证集内的序号）"
                )
            imgs = self._provider(data_path).load_batch(
                labels.detach().cpu().numpy(), intra_indices
            ).to(device, non_blocking=True)
            grounding_tokens = self.encode_grounding(imgs)

        auto_grounding = source == "probe"
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels, grounding_tokens, auto_grounding)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels, grounding_tokens, auto_grounding)
        return z

    @torch.no_grad()
    def _probe_grounding(self, z, t, labels):
        """用不带语义条件的 t+y 路径估计 clean image，再提取 DINO token。

        仅在 ``grounding_source=probe`` 时使用；它对应的 t+y 路径需要在训练
        中由 ``ground_drop_prob`` 覆盖，否则面对的是分布外输入。
        """
        x_probe = self.net(z, t.flatten(), labels, grounding_tokens=None)
        return self.encode_grounding(x_probe.clamp(-1.0, 1.0))

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, grounding_tokens=None, auto_grounding=False):
        if self.use_grounding and auto_grounding and grounding_tokens is None:
            grounding_tokens = self._probe_grounding(z, t, labels)

        null_labels = torch.full_like(labels, self.num_classes)

        # conditional：带语义条件。
        x_cond = self.net(z, t.flatten(), labels, grounding_tokens=grounding_tokens)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional：训练时 label 被丢弃的样本同时关闭了语义条件，
        # 因此这里也必须走不带 grounding 的 t+y 路径，否则无条件分支会
        # 通过 DINO token 拿到干净图的类别语义，CFG 差分失去意义。
        x_uncond = self.net(
            z, t.flatten(), null_labels,
            grounding_tokens=grounding_tokens, grounding_mask=False,
        )
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, grounding_tokens=None, auto_grounding=False):
        v_pred = self._forward_sample(z, t, labels, grounding_tokens, auto_grounding)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, grounding_tokens=None, auto_grounding=False):
        v_pred_t = self._forward_sample(z, t, labels, grounding_tokens, auto_grounding)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, grounding_tokens, auto_grounding)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        # DINOv3 冻结，不进入 EMA；只保存 JiT 和 grounding_proj 的可训练
        # 参数，避免每个 checkpoint 重复保存两份 DINOv3-B 权重。
        source_params = [p for p in self.parameters() if p.requires_grad]
        if self.ema_params1 is None or self.ema_params2 is None:
            raise RuntimeError("EMA buffers must be initialized before update_ema")
        if len(source_params) != len(self.ema_params1):
            raise RuntimeError("EMA buffer does not match trainable parameters")
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
