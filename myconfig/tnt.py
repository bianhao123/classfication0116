root_path = '/data111/bianhao/code/xy/mmclassification'
_base_ = [
    f'{root_path}/configs/_base_/schedules/mydata_bs128.py',
    f'{root_path}/configs/_base_/datasets/mydata_bs128_resize224.py', f'{root_path}/configs/_base_/default_runtime.py'
]

evaluation = dict(interval=1, metric=[
                  "accuracy", "precision", "recall", "f1_score"])

checkpoint_config = dict(interval=10)
log_config = dict(interval=100, hooks=[dict(
    type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
img_norm_cfg = dict(
    mean=[0, 0, 0],
    std=[255.0, 255.0, 255.0],
    to_rgb=False)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='TNT',
        arch='s',
        img_size=224,
        patch_size=16,
        in_channels=3,
        ffn_ratio=4,
        qkv_bias=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        first_stride=4,
        num_fcs=2,
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=.02),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=9,
        in_channels=384,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        init_cfg=dict(type='TruncNormal', layer='Linear', std=.02)))
