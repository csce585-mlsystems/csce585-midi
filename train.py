import torch
from src.dataset import ShardedMidiDataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from src.config import Model, Training, TOKENIZER, VOCAB_SIZE, MidiTokenization
from src.model import build_midi_model

def train():
   dataset = ShardedMidiDataset()
   model = build_midi_model()

   data_collator = DataCollatorForLanguageModeling(tokenizer=TOKENIZER, mlm=False)

   args = TrainingArguments(
      output_dir=Training.CHECKPOINT_PATH,
      per_device_train_batch_size=Training.BATCH_SIZE,
      gradient_accumulation_steps=Training.ACCUMULATION_STEPS,
      learning_rate=Training.LR,
      weight_decay=Training.WEIGHT_DECAY,
      warmup_steps=Training.WARMUP_STEPS,
      max_steps=Training.TOTAL_STEPS,
      logging_steps=Training.PRINT_EVERY,
      save_steps=Training.CHECKPOINT_EVERY,
      remove_unused_columns=False,
    #   eval_strategy="steps",
    #   eval_steps=Training.VAL_EVERY,
      save_total_limit=5,
      fp16=torch.cuda.is_available(),
      dataloader_num_workers=4,
      dataloader_prefetch_factor=2,
      dataloader_pin_memory=True
   )

   trainer = Trainer(
      model=model,
      args=args,
      train_dataset=dataset,
      eval_dataset=None,
      data_collator=data_collator
   )

   trainer.train()

if __name__ == "__main__":
   train()