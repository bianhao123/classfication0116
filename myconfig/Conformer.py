root_path = '/data111/bianhao/code/xy/mmclassification'
_base_ = [
    f'{root_path}/configs/_base_/schedules/mydata_bs128_resize224.py',
    f'{root_path}/configs/_base_/datasets/mydata_bs128_resize224.py', f'{root_path}/configs/_base_/default_runtime.py'
]

evaluation = dict(interval=1, metric=[
                  "accuracy", "precision", "recall", "f1_score"])

checkpoint_config = dict(interval=10)
log_config = dict(interval=100, hooks=[dict(
    type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Conformer', arch='base', drop_path_rate=0.1, init_cfg=None),
    neck=None,
    head=dict(
        type='ConformerHead',
        num_classes=9,
        in_channels=[1536, 576],
        init_cfg=None,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
)
