"""Latent Forcing 的 dino path，拆成独立 DiT 后作为 JiT 的 grounding 生成器。

* ``lf_modules.py``：从 LatentForcing/model_cot.py 逐字摘出的构件；
* ``model_util.py``：LatentForcing/util/model_util.py（RoPE buffer 去掉硬编码 .cuda()）；
* ``dinov2_encoder.py``：与 LF ``RAE.encode`` 一致的 DINOv2-reg-B/14 latent 编码；
* ``dino_dit.py``：独立的 dino DiT（JiT-B 同尺寸）；
* ``dinov2_wreg_base_imagenet1k_stat.pt``：RAE 发布的 ImageNet-1k latent 方差（mean=None）。
"""
