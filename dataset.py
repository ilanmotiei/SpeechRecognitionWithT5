
import torchaudio
from torch.utils.data import DataLoader, Dataset
import os
import tqdm
from torch import nn
import torch
from typing import List
import random
from torch.nn.utils.rnn import pad_sequence


class LetterTokenizer:

    def __init__(self):
        self.mapping = {'|': 2, 'A': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7, 'F': 8, 'G': 9, 'H': 10, 'I': 11, 'J': 12,
                        'K': 13, 'L': 14, 'M': 15, 'N': 16, 'O': 17, 'P': 18, 'Q': 19, 'R': 20, 'S': 21, 'T': 22,
                        'U': 23, 'V': 24, 'W': 25, 'X': 26, 'Y': 27, 'Z': 28, "'": 29}

        self.inverse_mapping = {v: k for k, v in self.mapping.items()}

        self.eos_token_id = 1

        self.decoder_bos_id = 0
        self.silence_token_id = self.mapping['|']

    def char_to_int(self, c: chr):
        return self.mapping[c]

    def int_to_char(self, i: int):
        return self.inverse_mapping[i]

    def num_tokens(self):
        return 2 + len(self.mapping.values())

    def decode(self, labels):
        text = ''

        for i in labels:
            if i == self.eos_token_id:
                break
            else:
                try:
                    text += self.inverse_mapping[i]
                except KeyError:
                    continue

        text = text.replace('|', ' ').strip()

        return text


class STTDataset(Dataset):

    def __init__(self,
                 data_root: str,
                 train: bool,
                 tokenizer: LetterTokenizer,
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 n_fft: int = 2048,
                 power: int = 2,
                 window_length: int = 400,
                 hop_len: int = 128,
                 augment_rate: float = 0.5,
                 augment_frequency_mask_param: int = 27,
                 augment_time_mask_rate: float = 0.05,
                 augment_time_mask_amount: int = 10,
                 max_audio_duration: float = 4,  # maximum audio duration at training
                 ):
        """
        :param data_root: --- speaker_id_#1
                              --- file_#1.wav
                              --- file_#1.aligned.txt
                              ...
                              --- file_#n.wav
                              --- file_#n.aligned.txt
                          --- speaker_id_#2
                              --- file_#1.wav
                              --- file_#1.aligned.txt
                              ...
                              --- file_#n.wav
                              --- file_#n_.aligned.txt
        """
        super().__init__()

        self.data_root = data_root
        self.train = train
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.audio_slice_duration = max_audio_duration

        self.items_paths = []
        for speaker_id in tqdm.tqdm(os.listdir(self.data_root), desc='Creating Dataset'):
            speaker_path = f'{self.data_root}/{speaker_id}'
            audio_files = [f for f in os.listdir(speaker_path) if f.endswith('wav')]

            for audio_file in audio_files:
                curr_audio_path = f'{speaker_path}/{audio_file}'
                curr_transcription_path = f"{data_root}/{speaker_id}/{audio_file[:-4]}._aligned_with_mfa.txt"
                self.items_paths.append((curr_audio_path, curr_transcription_path))

        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate,
                                                  n_mels=n_mels,
                                                  power=power,
                                                  n_fft=n_fft,
                                                  window_length=window_length,
                                                  hop_length=hop_len)

        self.spec_augmentor = SpecAugment(rate=augment_rate,
                                          freq_mask_param=augment_frequency_mask_param,
                                          time_mask_rate=augment_time_mask_rate,
                                          time_mask_amount=augment_time_mask_amount)

    def __len__(self):
        return len(self.items_paths)

    def __getitem__(self, item):
        audio_filepath, transcription_filepath = self.items_paths[item]

        audio_tensor, sample_rate = torchaudio.load(audio_filepath)

        if sample_rate != self.sample_rate:
            return self[(item+1) % len(self)]

        audio_len = int(audio_tensor.size(1))

        if self.train:
            # take a small cut from the audio
            slice_len = self.audio_slice_duration * sample_rate
            if slice_len <= audio_len:
                audio_start = random.randint(0, audio_len - slice_len)
                audio_end = audio_start + slice_len
                audio_tensor = audio_tensor[:, audio_start: audio_end]
            else:
                audio_start = 0
                audio_end = audio_len
        else:
            # take all the audio
            audio_start = 0
            audio_end = audio_len

        audio_features = self.feature_extractor(audio_tensor)
        if self.train:
            # augment
            audio_features = self.spec_augmentor(audio_features)
        audio_features = audio_features.squeeze(dim=0)

        audio_start /= self.sample_rate
        audio_end /= self.sample_rate

        try:
            transcription = open(transcription_filepath, 'r').readlines()
        except FileNotFoundError:
            # we don't have an alignment for the queried audio file
            return self[(item+1) % len(self)]

        transcription = [[a.strip() for a in line.split(' | ')] for line in transcription]
        # y = self.transcript_to_tensor(transcription,
        #                               num_bins=audio_features.size(1),
        #                               audio_start=audio_start/self.sample_rate,
        #                               audio_end=audio_end/self.sample_rate)

        def overlap(min1, max1, min2, max2):
            return max(0, min(max1, max2) - max(min1, min2))

        y = torch.Tensor([self.tokenizer.char_to_int(c) for c, s, e in transcription if overlap(float(s), float(e), audio_start, audio_end) > 0] +
                         [self.tokenizer.eos_token_id])

        # if int(y[0]) == self.tokenizer.silence_token_id:
        #     y = y[1:]
        # if int(y[-1]) == self.tokenizer.silence_token_id:
        #     y = y[:-1]

        y = y.long()

        return audio_features, y

    def transcript_to_tensor(self, transcription: List[List], num_bins, audio_start, audio_end) -> torch.Tensor:
        """
        :param transcription: A list of the following form:
                                [char#1, start_time, end_time]
                                [char#2, start_time, end_time]
                                ...
                                [char#n, start_time, end_time]

                            where char can be any character from the english alphabet (in upper-case)
                            or '|' (which indicated silence).

                            This list contains the information for the transcription of all the audio that contains this slice.

        :param num_bins: Amount of bins in the target tensor.
        :param audio_start: The starting time of the extracted audio slice.
        :param audio_end: The ending time of the extracted audio slice.

        :return: A vector of shape (audio_duration // self.token_bin_duration + 1, ) where each entry contains
                 a class index, that represents a character in the english alphabet, or silence ('|').
                 where audio_duration == audio_end - audio_end.
        """

        def overlap(min1, max1, min2, max2):
            return max(0, min(max1, max2) - max(min1, min2))

        y = torch.ones(size=(num_bins, )).long() * self.tokenizer.silence_token_id
        transcription = [(c, s, e) for c, s, e in transcription if (overlap(float(s), float(e), audio_start, audio_end) > 0)]

        for char, start_time, end_time in transcription:
            start_time = float(start_time)
            end_time = float(end_time)
            start_bin = int((start_time - audio_start) // self.token_duration)
            end_bin = int((end_time - audio_start) // self.token_duration)

            for bin_idx in range(start_bin, end_bin):
                bin_start_time = bin_idx * self.token_duration
                bin_end_time = bin_start_time + self.token_duration

                if overlap(start_time, end_time, bin_start_time, bin_end_time) >= 0.4 * self.token_duration:
                    y[bin_idx] = self.tokenizer.char_to_int(char)

        silence_bins_indexes = [i for i in range(num_bins) if y[i] == self.tokenizer.silence_token_id]

        # if min(silence_bins_indexes) == 0:
        #     # the current audio slice starts with silence
        #     # we'll put a bos token in the last bin of the first silence slice
        #
        #     idx = min([i for i in range(num_bins-1) if
        #                y[i] == self.tokenizer.silence_token_id and y[i+1] != self.tokenizer.silence_token_id])
        #     y[idx] = self.tokenizer.decoder_bos_id

        if max(silence_bins_indexes) < num_bins:
            # the current audio slice ends with silence
            # we'll put a eos token in the first bin of the last silence slice

            idx = max([i for i in range(1, num_bins) if
                       y[i] == self.tokenizer.silence_token_id and y[i-1] != self.tokenizer.silence_token_id])
            y[idx] = self.tokenizer.eos_token_id

        return y

    def collate_fn(self, tuples):
        """
        :param tuples: A list of B tuples of the shape (spectrogram, y).
        :return: A tuple of:
                  * the spectrograms stacked (after padding with zeros). shape = (B, self.n_mels, Max_i(spectrogram_i.shape[1])).
                  * the labels (y's) stacked (after padded with the silence token - zeros). shape = (B, Max_i(y_i.shape[0])).
        """

        spectrograms = [i[0].contiguous().transpose(0, 1) for i in tuples]
        ys = [i[1] for i in tuples]
        specs_lens = [int(s.size(0)) for s in spectrograms]

        padded_specs = pad_sequence(sequences=spectrograms,
                                    batch_first=True,
                                    padding_value=0)
        # ^ : shape = (B, Max_i(spectrogram_i.shape[1]), self.n_mels)

        padded_ys = pad_sequence(sequences=ys,
                                 batch_first=True,
                                 padding_value=-100)
        # ^ : shape = (B, Max_i(y_i.shape[0]))

        output = {'spec_tensor': padded_specs,
                  'text_tensor': padded_ys,
                  'attention_mask': pad_sequence([torch.ones(length) for length in specs_lens], batch_first=True, padding_value=0)}

        return output


class FeatureExtractor(nn.Module):
    def __init__(self, sample_rate, n_mels, power, n_fft, window_length, hop_length):
        super().__init__()

        self.transform = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                            n_mels=n_mels,
                                                                            n_fft=n_fft,
                                                                            power=power,
                                                                            win_length=window_length,
                                                                            hop_length=hop_length,
                                                                            center=True,
                                                                            pad_mode="reflect",
                                                                            norm="slaney",
                                                                            onesided=True,
                                                                            mel_scale="htk",
                                                                            ))

    def forward(self, x):
        x = self.transform(x)
        return x


class SpecAugment(nn.Module):

    def __init__(self, rate, freq_mask_param=27, time_mask_rate=0.05, time_mask_amount=1):
        super(SpecAugment, self).__init__()

        self.rate = rate
        self.time_mask_rate = time_mask_rate
        self.time_mask_amount = time_mask_amount
        self.frequency_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)

    def forward(self, x):
        seq_len = x.size(1)

        if random.random() < self.rate:
            # augment
            x = self.frequency_mask(x)
            for _ in range(self.time_mask_amount):
                x = torchaudio.transforms.TimeMasking(time_mask_param=self.time_mask_rate * seq_len)(x)

        return x


# if __name__ == "__main__":
#     audio_filepath = './LibriSpeech/sanity-check-train/19/19-198-0022.wav'
#     transcription_filepath = './LibriSpeech/sanity-check-train/19/19-198-0022._aligned.txt'
#
#     audio_tensor, sample_rate = torchaudio.load(audio_filepath)
#     audio_duration = audio_tensor.shape[1] / sample_rate
#
#     transcription = open(transcription_filepath, 'r').readlines()
#     transcription = [[a.strip() for a in line.split(' | ')] for line in transcription]
#
#     ds = STTDataset(data_root='./LibriSpeech/sanity-check-train', train=True)
#     y = ds.transcript_to_tensor(transcription, audio_duration=audio_duration)
#
#     char_to_int = {'|': 1}  # token for spaces
#     char_to_int.update({chr(c): i + 2 for i, c in enumerate(range(65, 91))})  # english alphabet (big letters)
#     char_to_int.update({"'": char_to_int['Z'] + 1})
#     int_to_char = {v: k for k, v in char_to_int.items()}
#
#     y = y[y.nonzero()]
#     ids = torch.unique_consecutive(y)
#
#     sentence = []
#
#     for id in ids:
#         sentence.append(int_to_char[id.item()])
#
#     sentence = "".join(sentence)
#     sentence = sentence.replace("|", " ")
#     sentence = sentence.strip()
#
#     print(sentence)
