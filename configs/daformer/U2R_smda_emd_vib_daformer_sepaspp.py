# Obtained from: https://github.com/lhoyer/DAFormer
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_vib_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/smda_U2R_5percent_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/smda_emd.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 111

# # Model Settings
model = dict(
    auxiliary_head=dict(
        type='FPN_VIB_Head',
        loss_decode=dict(
            type='KLLoss', loss_weight=0.1)),  # VIB Bottleneck with KL Divergence Loss
)  
# Modifications to Basic UDA
uda = dict(
    # Pseudo Labeling Configuration
    pseudo_threshold=0.968,
    # Coefficient for Trust Weight Adjustment
    coefficient=0.5,
    trust_update_interval=100,
    # increased Alpha for Pseudo Labeling
    alpha=0.999,
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
checkpoint_config = dict(by_epoch=False, interval=log_interval*10, max_keep_ckpts=1)
evaluation = dict(interval=log_interval, metric='mIoU', save_best='mIoU')
# Meta Information for Result Analysis
name = 'U2R_5percent_smda_emd_vib_daformer_sepaspp_512x512_40k'
exp = 'EMD_Baseline'
name_dataset = ''
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = ''
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
