# optimizer
optimizer = dict(type='Adam', lr=5e-5, weight_decay=0.0001, amsgrad=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8])
total_epochs = 14
