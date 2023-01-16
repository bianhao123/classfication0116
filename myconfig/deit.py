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
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=9,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision'),
    ))
