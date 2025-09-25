import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.dataset import MidiDataModule
from src.model import MidiModel
from src.config import VOCAB_SIZE, Model, Training, MidiTokenization

data_module = MidiDataModule(
    shard_files=list(MidiTokenization.OUT_PATH.glob("*.pt")),
    batch_size=Training.BATCH_SIZE,
    seq_len=Model.SEQ_LEN
)

model = MidiModel(
    vocab_size=VOCAB_SIZE,
    d_model=Model.D_MODEL,
    n_layers=Model.N_LAYERS,
    n_heads=Model.N_HEADS,
    d_ff=Model.D_FF,
    seq_len=Model.SEQ_LEN,
    lr=Training.LR,
    weight_decay=Training.WEIGHT_DECAY
)

checkpoint_callback = ModelCheckpoint(
    dirpath=Training.CHECKPOINT_PATH,
    filename="model_{step}",
    every_n_train_steps=Training.CHECKPOINT_EVERY,
    save_top_k=-1,
    save_weights_only=False
)

trainer = Trainer(
    accelerator="auto", devices=1, precision="16-mixed",
    max_steps=Training.TOTAL_STEPS,
    val_check_interval=Training.VAL_EVERY,
    callbacks=[checkpoint_callback],
    log_every_n_steps=Training.PRINT_EVERY
)
torch.set_float32_matmul_precision('high')
trainer.fit(model, datamodule=data_module)