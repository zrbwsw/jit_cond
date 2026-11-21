"""Latent Forcing 的 dino path，拆成独立 DiT 后作为 JiT 的 grounding 生成器。

* ``lf_modules.py``：从 LatentForcing/model_cot.py 逐字摘出的构件；
* ``model_util.py``：LatentForcing/util/model_util.py（RoPE buffer 去掉硬编码 .cuda()）；
* ``dinov2_encoder.py``：与 LF ``RAE.encode`` 一致的 DINOv2-reg-B/14 latent 编码；
* ``dino_dit.py``：独立的 dino DiT（JiT-B 同尺寸）；
* ``dinov2_encoder.py``：统一的冻结 latent 编码器（``dinov2b`` / ``dinov3b``），
  两者最后 norm 去 affine、归一化不乘 0.485，统计读自项目 ``stat/`` 目录；
* ``dinov3_backbone.py``：本地 DINOv3 加载器（dinov3b 时使用）；
* ``dino_dit.py``：独立的 dino DiT（JiT-B 同尺寸）。
统计文件（``stat/dinov2b_*.pt``、``stat/dinov3b_*.pt``）在项目 ``stat/`` 目录。
"""
