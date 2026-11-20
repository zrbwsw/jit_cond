import torch
import torch.nn as nn
from model_jit import JiT_models
from latent_force.dino_dit import DinoDiT_models
from latent_force.dinov2_encoder import DINOv2LatentEncoder, DEFAULT_STAT_PATH


class Denoiser(nn.Module):
    """JiT + 独立 dino DiT 的联合去噪器（join-dino）。

    * JiT path（pixel）：原始 JiT，grounding 通过 patch-level AdaLN 注入，
      ``c = t + y + t_dino + W_g·grounding``。
    * dino path：独立的 DinoDiT，去噪 DINOv2 latent（LF 的 x_dino 空间）。
    * 两个网络参数独立，不共享 trunk；每个样本两路都训。
    * grounding 来源：训练时用 clean 图经 DINOv2 编码并按 LF cascaded_noised 加噪；
      采样时先用 dino DiT 从纯噪声跑 ``dino_steps`` 步生成，再喂给 JiT。
    """

    def __init__(self, args):
        super().__init__()
        use_grounding = bool(getattr(args, "use_grounding", getattr(args, "enable_grounding", False)))
        grounding_dim = getattr(args, "grounding_dim", 768)
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            grounding_dim=(grounding_dim if use_grounding else None),
            use_t_dino=use_grounding,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.use_grounding = use_grounding
        self.grounding_dim = grounding_dim

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ---- dino path（独立 DiT + 冻结 DINOv2 编码器，与 LF 对齐） ----
        self.latent_size = args.img_size // 16          # DINOv2-B/14 在 224 下为 16×16
        self.dino_channels = grounding_dim              # 768
        self.D_mean = getattr(args, "D_mean", -1.2)     # LF 默认
        self.D_std = getattr(args, "D_std", 1.0)
        self.dino_weight = getattr(args, "dino_weight", 0.333)
        self.dino_noise_beta = getattr(args, "dino_noise_beta", 0.25)   # pixel 阶段 t_dino∈[1-β,1)
        self.dino_steps = getattr(args, "dino_steps", 25)
        self.dino_cfg_scale = getattr(args, "dino_cfg", 1.0)
        self.dino_cfg_interval = (getattr(args, "dino_interval_min", 0.0),
                                  getattr(args, "dino_interval_max", 1.0))
        self.dino_method = getattr(args, "dino_sampling_method", args.sampling_method)

        if self.use_grounding:
            self.dino_net = DinoDiT_models[getattr(args, "dino_model", "DinoDiT-B")](
                grid_size=self.latent_size,
                dino_in_channels=self.dino_channels,
                num_classes=args.class_num,
                attn_drop=args.attn_dropout,
                proj_drop=args.proj_dropout,
            )
            self.dino_encoder = DINOv2LatentEncoder(
                dinov2_path=getattr(args, "dinov2_path", "facebook/dinov2-with-registers-base"),
                normalization_stat_path=getattr(args, "dino_stat_path", None) or DEFAULT_STAT_PATH,
            )
            self.dino_encoder.eval()
        else:
            self.dino_net = None
            self.dino_encoder = None

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params（JiT path）
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    # ---------------------------------------------------------------- 工具

    def drop_labels_with_mask(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out, ~drop

    def sample_t(self, n, device=None, mean=None, std=None):
        mean = self.P_mean if mean is None else mean
        std = self.P_std if std is None else std
        return torch.sigmoid(torch.randn(n, device=device) * std + mean)

    @torch.no_grad()
    def encode_grounding(self, clean_images):
        """clean 图 [-1,1] -> LF 的 x_dino 空间 [B,768,16,16]。"""
        if not self.use_grounding or self.dino_encoder is None:
            return None
        return self.dino_encoder.encode(clean_images * 0.5 + 0.5)

    @staticmethod
    def _as_tokens(latent):
        """[B,768,16,16] -> [B,256,768]，供 JiT 的 patch-level AdaLN 使用。"""
        return latent.flatten(2).transpose(1, 2).contiguous()

    # ---------------------------------------------------------------- 训练

    def forward(self, x, labels):
        """联合 flow loss：JiT path 用带轻噪声的 dino token 作 grounding，dino path 独立去噪。

        与 LF 的 dino_first_cascaded_noised 对齐：每个样本两路都监督，JiT 见到的
        grounding 是 t_dino∈[1-β,1) 的带噪 clean latent（β=dino_noise_beta）。
        """
        bsz, device = x.size(0), x.device
        x_dino = self.encode_grounding(x)                       # [B,768,16,16]，teacher 目标

        labels_dropped, keep_mask = self.drop_labels_with_mask(labels)

        # ---- JiT (pixel) path ----
        t = self.sample_t(bsz, device=device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        # grounding 的噪声时间：pixel 阶段取 U[1-β,1)，对 clean latent 加轻噪声。
        t_dino_pixel = torch.rand(bsz, device=device) * self.dino_noise_beta + (1.0 - self.dino_noise_beta)
        td = t_dino_pixel.view(-1, 1, 1, 1)
        e_d = torch.randn_like(x_dino) * self.noise_scale
        z_dino_cond = td * x_dino + (1 - td) * e_d              # 带轻噪声的 grounding
        grounding_tokens = self._as_tokens(z_dino_cond)

        x_pred = self.net(
            z, t.flatten(), labels_dropped,
            grounding_tokens=grounding_tokens, grounding_mask=keep_mask,
            t_dino=t_dino_pixel,
        )
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        loss_jit = ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()

        # ---- dino path（独立去噪 DINOv2 latent，follow LF 的 t 调度） ----
        t_d = self.sample_t(bsz, device=device, mean=self.D_mean, std=self.D_std).view(-1, 1, 1, 1)
        e_d2 = torch.randn_like(x_dino) * self.noise_scale
        z_d = t_d * x_dino + (1 - t_d) * e_d2
        v_d = (x_dino - z_d) / (1 - t_d).clamp_min(self.t_eps)
        xd_pred = self.dino_net(z_d, t_d.flatten(), labels_dropped)
        vd_pred = (xd_pred - z_d) / (1 - t_d).clamp_min(self.t_eps)
        loss_dino = ((v_d - vd_pred) ** 2).mean(dim=(1, 2, 3)).mean()

        loss = loss_jit + self.dino_weight * loss_dino
        return loss, {"loss_jit": loss_jit.item(), "loss_dino": loss_dino.item()}

    # ---------------------------------------------------------------- dino path 采样

    @torch.no_grad()
    def _dino_velocity(self, z_d, t_d, labels):
        """dino DiT 的 guidance 速度；z_d:[B,768,16,16]，t_d:[B,1,1,1]。"""
        x_cond = self.dino_net(z_d, t_d.flatten(), labels)
        v_cond = (x_cond - z_d) / (1.0 - t_d).clamp_min(self.t_eps)
        if self.dino_cfg_scale == 1.0:
            return v_cond
        null = torch.full_like(labels, self.num_classes)
        x_unc = self.dino_net(z_d, t_d.flatten(), null)
        v_unc = (x_unc - z_d) / (1.0 - t_d).clamp_min(self.t_eps)
        low, high = self.dino_cfg_interval
        mask = (t_d < high) & ((low == 0) | (t_d > low))
        scale = torch.where(mask, self.dino_cfg_scale, 1.0)
        return v_unc + scale * (v_cond - v_unc)

    @torch.no_grad()
    def sample_dino_latent(self, labels):
        """从纯噪声采样 dino latent；返回 [B,768,16,16]（t_dino=1 的干净端）。"""
        bsz, device = labels.size(0), labels.device
        z_d = self.noise_scale * torch.randn(bsz, self.dino_channels, self.latent_size,
                                             self.latent_size, device=device)
        ts = torch.linspace(0.0, 1.0, self.dino_steps + 1, device=device).view(-1, 1, 1, 1, 1).expand(-1, bsz, -1, -1, -1)
        heun = self.dino_method == "heun"
        for i in range(self.dino_steps - 1):
            z_d = self._dino_step(z_d, ts[i], ts[i + 1], labels, heun)
        z_d = self._dino_step(z_d, ts[-2], ts[-1], labels, heun=False)   # 末步 Euler
        return z_d

    @torch.no_grad()
    def _dino_step(self, z_d, t, t_next, labels, heun):
        v = self._dino_velocity(z_d, t, labels)
        z_euler = z_d + (t_next - t) * v
        if not heun:
            return z_euler
        v2 = self._dino_velocity(z_euler, t_next, labels)
        return z_d + (t_next - t) * 0.5 * (v + v2)

    # ---------------------------------------------------------------- JiT path 采样

    @torch.no_grad()
    def generate(self, labels, grounding_tokens=None):
        """生成图像。grounding 由 dino DiT 采样得到，整条 JiT 轨迹共用（instance-level）。"""
        device = labels.device
        bsz = labels.size(0)

        if self.use_grounding and grounding_tokens is None:
            z_dino = self.sample_dino_latent(labels)             # [B,768,16,16]，t_dino=1
            grounding_tokens = self._as_tokens(z_dino)

        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)
        stepper = self._heun_step if self.method == "heun" else self._euler_step
        for i in range(self.steps - 1):
            z = stepper(z, timesteps[i], timesteps[i + 1], labels, grounding_tokens)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels, grounding_tokens)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, grounding_tokens=None):
        null_labels = torch.full_like(labels, self.num_classes)
        # grounding 由 dino DiT 采到 t_dino=1，故 JiT 的 t_dino 条件取 1。
        t_dino = torch.ones_like(t.flatten()) if self.use_grounding else None

        x_cond = self.net(z, t.flatten(), labels, grounding_tokens=grounding_tokens, t_dino=t_dino)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # 无条件分支：丢 y 同时丢 grounding（grounding_mask=False 退回纯 t+y，t_dino 也一并关闭）。
        x_uncond = self.net(z, t.flatten(), null_labels,
                            grounding_tokens=grounding_tokens, grounding_mask=False, t_dino=t_dino)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)
        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, grounding_tokens=None):
        v_pred = self._forward_sample(z, t, labels, grounding_tokens)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, grounding_tokens=None):
        v_pred_t = self._forward_sample(z, t, labels, grounding_tokens)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, grounding_tokens)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def update_ema(self):
        # 冻结的 DINOv2 不进 EMA；JiT 和 dino DiT 的可训练参数都纳入。
        source_params = [p for p in self.parameters() if p.requires_grad]
        if self.ema_params1 is None or self.ema_params2 is None:
            raise RuntimeError("EMA buffers must be initialized before update_ema")
        if len(source_params) != len(self.ema_params1):
            raise RuntimeError("EMA buffer does not match trainable parameters")
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
