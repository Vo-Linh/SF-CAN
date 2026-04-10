# UNetFormer with ResNet-18 backbone (from GeoSeg)
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='UNetFormerBackbone',
        backbone_name='swsl_resnet18',
        pretrained=True,
        out_indices=(1, 2, 3, 4),
        output_stride=32),
    decode_head=dict(
        type='UNetFormerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=64,
        decode_channels=64,
        window_size=8,
        num_heads=8,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        vib_params=None,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
