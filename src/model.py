
import torch
import pytorch_lightning as pl
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration, LongformerConfig, LongformerForTokenClassification
from dataset import LetterTokenizer
from torch.optim.adamw import AdamW
from datasets import load_metric
from torch.nn.utils.rnn import pad_sequence


class SpeechTranscriptor(pl.LightningModule):

    def __init__(self,
                 tokenizer: LetterTokenizer,
                 text_classes: int,  # amount of text classes
                 device: torch.device,
                 embedding_dim: int = 480,
                 n_mels: int = 80,  # amount of features at spectrograms
                 num_beams: int = 5,  # num of beams for beam search
                 lr: float = 1e-04):

        super().__init__()
        self.tokenizer = tokenizer
        self.num_beams = num_beams
        self.lr = lr

        # self.pre_encoder = nn.Sequential(nn.Conv1d(in_channels=n_mels, out_channels=embedding_dim, kernel_size=3, stride=1, padding='same'),
        #                                  nn.ReLU(),
        #                                  nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=1, padding='same'),
        #                                  nn.ReLU())

        # self.projector = nn.Linear(in_features=n_mels, out_features=embedding_dim)

        config = T5Config.from_pretrained('t5-small')
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
        # config = T5Config.from_pretrained('t5-small')
        # self.t5 = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.t5.lm_head = nn.Linear(config.d_model, text_classes, bias=False)

        # config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        # self.longformer = LongformerForTokenClassification.from_pretrained('allenai/longformer-base-4096')
        # self.longformer.classifier = nn.Linear(in_features=config.hidden_size, out_features=text_classes)
        # self.longformer.num_labels = text_classes

        self.max_positions = config.n_positions

        self.wer_metric = load_metric('wer')
        self.cer_metric = load_metric('cer')

    def forward(self, spectrograms, attention_mask, labels):
        """
        :param spectrograms: shape == (B, max_spectrogram_length, self.n_mels)
        :param specs_lens: shape == (B, )
        :param labels: (B, max_label_length)
        :return:
        """

        # B = spectrograms.size(0)
        # L = spectrograms.size(1)
        # positional_embeddings = self.positional_embedding(torch.stack([torch.arange(L) for _ in range(B)]).to(spectrograms.device))
        # inputs_embeds = spectrograms + positional_embeddings

        inputs_embeds = spectrograms

        return {'output': self.t5(inputs_embeds=inputs_embeds,
                                  attention_mask=attention_mask,
                                  labels=labels),
                'input_embeds': inputs_embeds}

    def training_step(self, train_batch, batch_idx):
        spectrograms = train_batch['spec_tensor']
        attention_mask = train_batch['attention_mask']
        labels = train_batch['text_tensor']
        forward_output = self(spectrograms, attention_mask, labels)
        loss = forward_output['output'].loss

        self.log('Training Loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # spectrograms = val_batch['spec_tensor']
        # attention_mask = val_batch['attention_mask']
        # labels = val_batch['text_tensor']
        # forward_output = self(spectrograms, attention_mask, labels)
        # loss = forward_output['output'].loss
        # self.log('Validation Loss', loss)
        #
        # model_outputs = self.t5.generate(inputs_embeds=forward_output['input_embeds'],
        #                                  attention_mask=attention_mask,
        #                                  bos_token_id=self.tokenizer.decoder_bos_id,
        #                                  num_beams=self.num_beams,
        #                                  max_length=100,
        #                                  do_sample=True)
        # predictions = []
        # references = []
        #
        # for prediction, gt in zip(model_outputs, labels):
        #     gt = gt[gt != -100]  # removing the padding from the gt's
        #     predictions.append(self.tokenizer.decode(prediction.tolist()))
        #     references.append(self.tokenizer.decode(gt.tolist()))
        #     print("=======================================")
        #     print(f"{predictions[-1]}, {references[-1]}")

        # new implementation --- batch size is always 1

        spec = val_batch['spec_tensor'].squeeze(0)  # shape = (spec_len, self.n_mels)
        transcription = self.transcribe(spectrogram=spec)
        labels = val_batch['text_tensor'].squeeze(0)  # shape = (text_len, )
        gt_transcription = self.tokenizer.decode(labels.tolist())

        predictions = [transcription]
        references = [gt_transcription]

        print("====================")
        print(f"Transcription: {transcription}")
        print(f"GT Transcription: {gt_transcription}")

        self.log('Validation WER %', 100 * self.wer_metric.compute(predictions=predictions, references=references))
        self.log('Validation CER %', 100 * self.cer_metric.compute(predictions=predictions, references=references))

    def transcribe(self, spectrogram):
        # spectrogram.shape = (spec_len, self.n_mels)

        splits = torch.split(spectrogram, dim=0, split_size_or_sections=self.max_positions)

        if len(splits) >= 5:
            seqs = [splits[:int(len(splits)//2)], splits[int(len(splits)//2):]]
        else:
            seqs = [splits]

        prediction = ''

        for seq in seqs:
            batch = pad_sequence(sequences=seq,
                                 batch_first=True,
                                 padding_value=0)
            # batch.shape = (len(seq), self.max_positions, self.n_mels)

            # B = batch.size(0)
            # L = batch.size(1)
            # positional_embeddings = self.positional_embedding(torch.stack([torch.arange(L) for _ in range(B)]).to(batch.device))
            # input_embeds = batch + positional_embeddings

            input_embeds = batch

            model_outputs = self.t5.generate(inputs_embeds=input_embeds,
                                             attention_mask=pad_sequence(sequences=[torch.ones(s.size(0)) for s in seq], batch_first=True, padding_value=0).to(batch.device),
                                             bos_token_id=self.tokenizer.decoder_bos_id,
                                             num_beams=self.num_beams,
                                             max_length=100,
                                             do_sample=True)

            for t in model_outputs:
                prediction += self.tokenizer.decode(t.tolist())

        return prediction


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


if __name__ == "__main__":

    SpeechTranscriptor(text_classes=30,
                       device=torch.device('cuda'))


