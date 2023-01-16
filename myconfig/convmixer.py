root_path = '/data111/bianhao/code/xy/mmclassification'
_base_ = [
    f'{root_path}/configs/_base_/schedules/mydata_bs128.py',
    f'{root_path}/configs/_base_/datasets/mydata_bs128.py', f'{root_path}/configs/_base_/default_runtime.py'
]

evaluation = dict(interval=1, metric=[
                  "accuracy", "precision", "recall", "f1_score"])

checkpoint_config = dict(interval=10)
log_config = dict(interval=100, hooks=[dict(
    type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

model = dict(
    type='ImageClassifier',
    backbone=dict(type='ConvMixer', arch='768/32', act_cfg=dict(type='ReLU')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=9,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
