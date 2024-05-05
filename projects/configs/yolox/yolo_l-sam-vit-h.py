_base_ = [
    '../_base_/datasets/coco_panoptic.py', '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/instance_segment_anything/'

model = dict(
    type='DetWrapperInstanceSAM',
    det_wrapper_type='generalized_detector',
    det_wrapper_cfg=dict(
                det_config = './mmdetection/configs/yolox/yolox_l_8x8_300e_coco.py',
                det_weight = './ckpt/yolox_l.pth',
        ),
    num_classes=80,
    model_type='vit_h',
    sam_checkpoint='ckpt/sam_vit_h_4b8939.pth',
    use_sam_iou=True,
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

