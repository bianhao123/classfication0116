optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[100, 500])
runner = dict(type='EpochBasedRunner', max_epochs=500)
dataset_type = 'CustomDataset'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    dict(
        type='Normalize',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    dict(
        type='Normalize',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type='CustomDataset',
        data_prefix='/data111/bianhao/code/xy/yinhe/myData64/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255.0, 255.0, 255.0],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CustomDataset',
        data_prefix='/data111/bianhao/code/xy/yinhe/myData64/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255.0, 255.0, 255.0],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='CustomDataset',
        data_prefix='/data111/bianhao/code/xy/yinhe/myData64/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255.0, 255.0, 255.0],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
root_path = '/data111/bianhao/code/xy/mmclassification'
evaluation = dict(
    interval=1, metric=['accuracy', 'precision', 'recall', 'f1_score'])
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientFormer',
        arch='l1',
        drop_path_rate=0,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=0.02,
                bias=0.0),
            dict(type='Constant', layer=['GroupNorm'], val=1.0, bias=0.0),
            dict(type='Constant', layer=['LayerScale'], val=1e-05)
        ]),
    neck=dict(type='GlobalAveragePooling', dim=1),
    head=dict(
        type='EfficientFormerClsHead',
        in_channels=448,
        num_classes=9,
        distillation=False))
work_dir = './work_dirs/efficientformer'
gpu_ids = [0]
