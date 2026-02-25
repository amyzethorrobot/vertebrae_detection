# NOTE: 
# CHANGE FILEPATHS TO DATASETS AND CHECKPOINTS 
# BEFORE USING!!!

auto_scale_lr = dict(base_batch_size=1024)
backend_args = dict(backend='local')
base_lr = 0.005
checkpoint = '/home/user/cool_checkpoints/20251704_back_348ep.pth'
codec = dict(
    decode_visibility=True,
    input_size=(
        768,
        1024,
    ),
    no_fine=[
        [
            22,
            70,
        ],
        [
            92,
            94,
        ],
    ],
    normalize=False,
    sigma=(
        4.9,
        5.66,
    ),
    simcc_split_ratio=2.0,
    type='SimCCFineInvisible',
    use_dark=False)
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
data_mode = 'topdown'
data_root = '/home/user/dataset_folder/spine_1000flfs/'
dataset_type = 'CocoSpineHipCenter'
default_hooks = dict(
    badcase=dict(
        _scope_='mmpose',
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        _scope_='mmpose',
        interval=10,
        max_keep_ckpts=5,
        rule='greater',
        save_best='10p/PCKVert',
        type='CheckpointHook'),
    logger=dict(_scope_='mmpose', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmpose', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmpose', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmpose', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmpose', enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmpose',
    by_epoch=True,
    num_digits=6,
    type='LogProcessor',
    window_size=50)
loss = dict(
    beta=10.0,
    label_softmax=True,
    type='KLDiscretLoss',
    use_target_weight=True)
max_epochs = 1200
model = dict(
    backbone=dict(
        _scope_='mmdet',
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.67,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint='/home/user/cool_checkpoints/20251704_back_348ep.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=0.75),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            decode_visibility=True,
            input_size=(
                768,
                1024,
            ),
            no_fine=[
                [
                    22,
                    70,
                ],
                [
                    92,
                    94,
                ],
            ],
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCFineInvisible',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=768,
        in_featuremap_size=(
            24,
            32,
        ),
        input_size=(
            768,
            1024,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=94,
        simcc_split_ratio=2.0,
        type='RTMCCHead'),
    test_cfg=dict(flip_test=True),
    type='TopdownPoseEstimator')
optim_wrapper = dict(
    clip_grad=dict(max_norm=150, norm_type=2),
    optimizer=dict(lr=0.005, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=500,
        by_epoch=True,
        end=1200,
        end_factor=0.05,
        start_factor=1.0,
        type='LinearLR'),
]
randomness = dict(seed=2021)
resume = False
test_ann_file = 'annotations/COCO_1000FLFS_Test.json'
test_batch_size = 20
test_cfg = dict()
test_dataloader = dict(
    batch_size=20,
    dataset=dict(
        ann_file='annotations/COCO_1000FLFS_Test.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='/home/user/dataset_folder/spine_1000flfs/',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(padding=1.0, type='GetBBoxCenterScale'),
            dict(input_size=(
                768,
                1024,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoSpineHipCenter'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        segments=dict(
            cervical=[
                0,
                22,
            ],
            full=[
                0,
                92,
            ],
            lumbar=[
                70,
                92,
            ],
            thoracic=[
                22,
                70,
            ]),
        thr=0.01,
        type='PCKSegments'),
    dict(
        prefix='10p_GLOBAL',
        radius='global',
        segments=dict(
            cervical=[
                0,
                22,
            ],
            full=[
                0,
                92,
            ],
            lumbar=[
                70,
                92,
            ],
            thoracic=[
                22,
                70,
            ]),
        thr=0.1,
        type='PCKVertSegments'),
    dict(
        prefix='25p_GLOBAL',
        radius='global',
        segments=dict(
            cervical=[
                0,
                22,
            ],
            full=[
                0,
                92,
            ],
            lumbar=[
                70,
                92,
            ],
            thoracic=[
                22,
                70,
            ]),
        thr=0.25,
        type='PCKVertSegments'),
    dict(
        prefix='10p_LOCAL',
        radius='local',
        segments=dict(
            cervical=[
                0,
                22,
            ],
            full=[
                0,
                92,
            ],
            lumbar=[
                70,
                92,
            ],
            thoracic=[
                22,
                70,
            ]),
        thr=0.1,
        type='PCKVertSegments'),
    dict(
        prefix='25p_LOCAL',
        radius='local',
        segments=dict(
            cervical=[
                0,
                22,
            ],
            full=[
                0,
                92,
            ],
            lumbar=[
                70,
                92,
            ],
            thoracic=[
                22,
                70,
            ]),
        thr=0.25,
        type='PCKVertSegments'),
    dict(
        error_treshold=0.05,
        segments=dict(
            cervical=[
                0,
                22,
            ],
            full=[
                0,
                92,
            ],
            lumbar=[
                70,
                92,
            ],
            thoracic=[
                22,
                70,
            ]),
        type='DistanceSegments'),
]
train_ann_file = 'annotations/COCO_1000FLFS_Train.json'
train_batch_size = 20
train_cfg = dict(by_epoch=True, max_epochs=1200, val_interval=1)
train_dataloader = dict(
    batch_size=20,
    dataset=dict(
        ann_file='annotations/COCO_1000FLFS_Train.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='/home/user/dataset_folder/spine_1000flfs/',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(padding=1.0, type='GetBBoxCenterScale'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                min_lower_keypoints=20,
                min_total_keypoints=92,
                padding=1.25,
                prob=0.35,
                type='RandomHalfBody',
                upper_prioritized_prob=0.0),
            dict(
                rotate_factor=15,
                rotate_prob=0.75,
                scale_factor=[
                    0.75,
                    1.1,
                ],
                scale_prob=0.5,
                shift_factor=0.15,
                shift_prob=0.5,
                type='RandomBBoxTransform'),
            dict(input_size=(
                768,
                1024,
            ), type='TopdownAffine'),
            dict(
                transforms=[
                    dict(
                        color_shift=[
                            0.01,
                            0.05,
                        ],
                        intensity=[
                            0.1,
                            0.5,
                        ],
                        p=0.1,
                        type='ISONoise'),
                    dict(
                        allow_shifted=False,
                        angle_range=(
                            0,
                            180,
                        ),
                        blur_limit=(
                            5,
                            11,
                        ),
                        p=0.25,
                        type='MotionBlur'),
                ],
                type='Albumentation'),
            dict(
                encoder=dict(
                    decode_visibility=True,
                    input_size=(
                        768,
                        1024,
                    ),
                    no_fine=[
                        [
                            22,
                            70,
                        ],
                        [
                            92,
                            94,
                        ],
                    ],
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCFineInvisible',
                    use_dark=False),
                type='GenerateTarget',
                use_dataset_keypoint_weights=True),
            dict(type='PackPoseInputs'),
        ],
        type='CocoSpineHipCenter'),
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        min_lower_keypoints=20,
        min_total_keypoints=92,
        padding=1.25,
        prob=0.35,
        type='RandomHalfBody',
        upper_prioritized_prob=0.0),
    dict(
        rotate_factor=15,
        rotate_prob=0.75,
        scale_factor=[
            0.75,
            1.1,
        ],
        scale_prob=0.5,
        shift_factor=0.15,
        shift_prob=0.5,
        type='RandomBBoxTransform'),
    dict(input_size=(
        768,
        1024,
    ), type='TopdownAffine'),
    dict(
        transforms=[
            dict(
                color_shift=[
                    0.01,
                    0.05,
                ],
                intensity=[
                    0.1,
                    0.5,
                ],
                p=0.1,
                type='ISONoise'),
            dict(
                allow_shifted=False,
                angle_range=(
                    0,
                    180,
                ),
                blur_limit=(
                    5,
                    11,
                ),
                p=0.25,
                type='MotionBlur'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            decode_visibility=True,
            input_size=(
                768,
                1024,
            ),
            no_fine=[
                [
                    22,
                    70,
                ],
                [
                    92,
                    94,
                ],
            ],
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCFineInvisible',
            use_dark=False),
        type='GenerateTarget',
        use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs'),
]
val_ann_file = 'annotations/COCO_1000FLFS_Val.json'
val_cfg = dict()
val_dataloader = dict(
    batch_size=20,
    dataset=dict(
        ann_file='annotations/COCO_1000FLFS_Val.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='/home/user/dataset_folder/spine_1000flfs/',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(padding=1.0, type='GetBBoxCenterScale'),
            dict(input_size=(
                768,
                1024,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoSpineHipCenter'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(thr=0.05, type='PCKAccuracy'),
    dict(exclude_keypoints=[
        92,
        93,
    ], prefix='05p', thr=0.05, type='PCKVert'),
    dict(exclude_keypoints=[
        92,
        93,
    ], prefix='10p', thr=0.1, type='PCKVert'),
    dict(exclude_keypoints=[
        92,
        93,
    ], prefix='25p', thr=0.25, type='PCKVert'),
    dict(type='DistanceVert'),
]
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(
        768,
        1024,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(_scope_='mmpose', type='LocalVisBackend'),
    dict(_scope_='mmpose', type='TensorboardVisBackend'),
]
visualizer = dict(
    _scope_='mmpose',
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'work_dir/rtmpose/vertescan_spatial_aug_kld_newpipeline'
workers = 10
