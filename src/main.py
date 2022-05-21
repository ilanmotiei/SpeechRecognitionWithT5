from pytorch_lightning.callbacks import ModelCheckpoint

from model import SpeechTranscriptor
from dataset import STTDataset, LetterTokenizer
import params
import pytorch_lightning as pl
from torch.utils.data import DataLoader


tokenizer = LetterTokenizer()

train_dataset = STTDataset(data_root=params.training_data_root,
                           train=True,
                           tokenizer=tokenizer,
                           sample_rate=params.sample_rate,
                           n_mels=params.n_mels,
                           n_fft=params.n_fft,
                           power=params.power,
                           window_length=params.window_length,
                           hop_len=params.hop_len,
                           augment_rate=params.augment_rate,
                           augment_frequency_mask_param=params.augment_frequency_mask_param,
                           augment_time_mask_rate=params.augment_time_mask_rate,
                           augment_time_mask_amount=params.augment_time_mask_amount,
                           max_audio_duration=params.max_audio_duration)

validation_dataset = STTDataset(data_root=params.validation_data_root,
                                train=False,
                                tokenizer=tokenizer,
                                sample_rate=params.sample_rate,
                                n_fft=params.n_fft,
                                n_mels=params.n_mels,
                                power=params.power,
                                window_length=params.window_length,
                                hop_len=params.hop_len)

train_dataloader = DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=params.batch_size,
                              num_workers=params.num_workers,
                              collate_fn=train_dataset.collate_fn)

validation_dataloader = DataLoader(dataset=validation_dataset,
                                   shuffle=False,
                                   batch_size=1,
                                   num_workers=params.num_workers,
                                   collate_fn=validation_dataset.collate_fn)

model = SpeechTranscriptor(tokenizer=tokenizer,
                           text_classes=tokenizer.num_tokens(),
                           device=params.device,
                           embedding_dim=params.embedding_dim,
                           n_mels=params.n_mels,
                           num_beams=params.num_beams,
                           lr=params.lr)

checkpoint_callback = ModelCheckpoint(dirpath="model_starting_from_magenta_t5/", save_top_k=2, monitor="Validation WER %")
trainer = pl.Trainer(gpus=params.n_gpus,
                     max_epochs=params.epochs,
                     check_val_every_n_epoch=5,
                     callbacks=[checkpoint_callback],
                     resume_from_checkpoint='./model_starting_from_magenta_t5/epoch=59-step=527340.ckpt'
                     )
trainer.fit(model, train_dataloader, validation_dataloader)
