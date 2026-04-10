# DeepLabV3+ R101 + Confidence-Aware (no VIB)
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/deeplabv3plus_r101-d8.py',
    '../_base_/datasets/smda_R2U_5percent_512x512.py',
    '../_base_/uda/smda_emd.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]
seed = 111
uda = dict(
    alpha=0.999,
    pseudo_threshold=0.968,
    coefficient=0.5,
    trust_update_interval=100,
)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)
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
log_interval = 2000
checkpoint_config = dict(
    by_epoch=False, interval=log_interval * 5, max_keep_ckpts=1)
evaluation = dict(interval=log_interval, metric='mIoU', save_best='mIoU')
name = 'R2U_5percent_smda_emd_deeplabv3plus_r101'
exp = 'ablation_deeplabv3plus'
name_dataset = 'rural2urban'
name_architecture = 'deeplabv3plus_r101'
name_encoder = 'r101'
name_decoder = 'deeplabv3plus'
name_uda = 'smda_emd'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
