module = DataModule(
    datafiles=['data.h5ad'],
    labelfiles=['labels.csv'],
    class_label='Cell Type',
)

logger = WandbLogger(
    project="Single-Cell Classifier",
    name=name,
)

lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
early_stopping_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
)

trainer = pl.Trainer(
    gpus=1,
    logger=wandb_logger,
    gradient_clip_val=0.5,
    callbacks=[
        lr_callback, 
        upload_callback,
        early_stopping_callback,
    ]
)

model = SIMSClassifier(
    input_dim=module.num_features,
    output_dim=module.num_labels,
)

trainer.fit(model, datamodule=module)