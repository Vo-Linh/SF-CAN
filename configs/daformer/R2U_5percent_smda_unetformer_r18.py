# UNetFormer R18 Baseline (no VIB, no confidence-aware)
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/unetformer_r18.py',
    '../_base_/datasets/smda_R2U_5percent_512x512.py',
    '../_base_/uda/smda_base.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]
seed = 111
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
name = 'R2U_5percent_smda_unetformer_r18'
exp = 'ablation_unetformer'
name_dataset = 'rural2urban'
name_architecture = 'unetformer_r18'
name_encoder = 'swsl_resnet18'
name_decoder = 'unetformer'
name_uda = 'smda'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
