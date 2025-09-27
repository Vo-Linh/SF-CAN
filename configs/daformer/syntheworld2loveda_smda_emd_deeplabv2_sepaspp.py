# Obtained from: https://github.com/lhoyer/DAFormer
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/deeplabv2_r50-d8.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/smda_syntheworld_Xloveda_to_loveda_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/smda_emd.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 111
# Modifications to Basic UDA

data = dict(
    train=dict(
        # # Rare Class Sampling
        # rare_class_sampling=dict(
        #     min_pixels=3000, class_temp=0., min_crop_ratio=0.5)
        )) 
uda = dict(
    # Pseudo Labeling Configuration
    pseudo_threshold=0.968,
    # Coefficient for Trust Weight Adjustment
    coefficient=0.5,
    trust_update_interval=100,
    )
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
log_interval = 2000
checkpoint_config = dict(by_epoch=False, interval=log_interval*10, max_keep_ckpts=10)
evaluation = dict(interval=log_interval, metric='mIoU')
# Meta Information for Result Analysis
name = '5percent-smda-emd-deeplabv2_r50_d8-512x512_40k'
exp = 'EMD_Baseline'
name_dataset = ''
name_architecture = ''
name_encoder = ''
name_decoder = ''
name_uda = ''
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
