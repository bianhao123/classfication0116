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
img_norm_cfg = dict(
    mean=[0, 0, 0],
    std=[255.0, 255.0, 255.0],
    to_rgb=False)

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='VAN', arch='b0', drop_path_rate=0.1),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=9,
        in_channels=256,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ])
