

from model import SpeechTranscriptor
from dataset import STTDataset, LetterTokenizer
import params
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

evaluated_data_root = 'LibriSpeech/test-other_ordered'

tokenizer = LetterTokenizer()

evaluation_dataset = STTDataset(data_root=evaluated_data_root,
                                train=False,
                                tokenizer=tokenizer,
                                sample_rate=params.sample_rate,
                                n_fft=params.n_fft,
                                n_mels=params.n_mels,
                                power=params.power,
                                window_length=params.window_length,
                                hop_len=params.hop_len)

evaluation_dataloader = DataLoader(dataset=evaluation_dataset,
                                   shuffle=False,
                                   batch_size=1,
                                   num_workers=params.num_workers,
                                   collate_fn=evaluation_dataset.collate_fn)

model = SpeechTranscriptor(tokenizer=tokenizer,
                           text_classes=tokenizer.num_tokens(),
                           device=params.device,
                           embedding_dim=params.embedding_dim,
                           n_mels=params.n_mels,
                           num_beams=params.num_beams,
                           lr=params.lr)

model = model.load_from_checkpoint(checkpoint_path='./model_starting_from_magenta_t5/epoch=59-step=527340.ckpt',
                                   tokenizer=tokenizer,
                                   text_classes=tokenizer.num_tokens(),
                                   device=params.device)

model = model.to(params.device)

wer = 0
cer = 0
items = 0

for eval_batch in tqdm(evaluation_dataloader):
    spec = eval_batch['spec_tensor'].squeeze(0)  # shape = (spec_len, self.n_mels)
    transcription = model.transcribe(spectrogram=spec.to(params.device))
    labels = eval_batch['text_tensor'].squeeze(0)  # shape = (text_len, )
    gt_transcription = model.tokenizer.decode(labels.tolist())

    predictions = [transcription]
    references = [gt_transcription]

    curr_wer = 100 * model.wer_metric.compute(predictions=predictions, references=references)
    curr_cer = 100 * model.cer_metric.compute(predictions=predictions, references=references)

    wer += curr_wer
    cer += curr_cer
    items += 1


wer /= items
cer /= items

print(f"Evaluated dataset: {evaluation_data_root}")
print(f"Avg. WER: {wer}")
print(f"Avg. CER: {cer}")
