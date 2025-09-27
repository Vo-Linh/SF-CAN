# Obtained from: https://github.com/lhoyer/DAFormer
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_vib_mitb5.py',
    '../_base_/datasets/smda_R2U_5percent_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/smda_emd.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 111
uda = dict(
    alpha=0.999,
    # Pseudo Labeling Configuration
    pseudo_threshold=0.968,
    # Coefficient for Trust Weight Adjustment
    coefficient=0.5,
    trust_update_interval=100,
    )

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
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
checkpoint_config = dict(by_epoch=False, interval=log_interval*5, max_keep_ckpts=1)
evaluation = dict(interval=log_interval, metric='mIoU', save_best='mIoU')
# Meta Information for Result Analysis
name = 'R2U_5percent_smda_emd_daformer_vib_sepaspp'
exp = 'EMD_Baseline'
name_dataset = 'rural2urban'
name_architecture = 'daformer_sepaspp_vib_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = ''
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
