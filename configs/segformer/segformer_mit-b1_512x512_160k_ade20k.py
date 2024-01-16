_base_ = ['./segformer_mit-b0_512x512_160k_ade20k.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa

# model settings
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[2, 2, 2, 2]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
#  20240113
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(2048, 512),
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             # resize image to multiple of 32, improve SegFormer by 0.5-1.0 mIoU.
#             dict(type='ResizeToMultiple', size_divisor=32),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
