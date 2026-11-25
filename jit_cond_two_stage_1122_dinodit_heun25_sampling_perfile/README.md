# DINO DiT / Pixel JiT · noflip

此目录从头训练带实例重建监督的 Stage 1 DinoDiT，每 20 个 epoch 对 ImageNet val 全部 50,000 张做 DINO 特征重建验证。Stage 2 默认按图加载单视图重建缓存，自由采样使用 DinoDiT 生成的 grounding。

[中文修改指南与完整运行命令](修改代码指南.md)涵盖数据准备、训练损失、TensorBoard 指标、分片导出、Pixel 训练和采样。
