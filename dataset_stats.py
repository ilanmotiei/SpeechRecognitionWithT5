
import torch
import torchaudio
import os
from tqdm import tqdm
import numpy as np


def stats(root: str):
    speakers = os.listdir(root)

    start_time = []
    end_time = []
    audios_lengths = []

    for speaker_id in tqdm(speakers, desc=f'Calculating Dataset Stats'):
        speaker_path = f'{root}/{speaker_id}'
        speaker_audio_files = [file for file in os.listdir(speaker_path) if file.endswith('wav')]

        for speaker_audio_file in speaker_audio_files:
            transcript_file_path = f"{speaker_path}/{speaker_audio_file[:-3]}_aligned.txt"
            transcript = [[a.strip() for a in line.split(' | ')] for line in open(transcript_file_path, 'r').readlines()]

            start_time += [float(line[1]) for line in transcript]
            end_time += [float(line[2]) for line in transcript]

            wav_file_path = f"{speaker_path}/{speaker_audio_file}"
            audio, sr = torchaudio.load(wav_file_path)
            audio_length_in_seconds = audio.size(1) / sr
            audios_lengths.append(audio_length_in_seconds)

    start_time = np.array(start_time)
    end_time = np.array(end_time)
    audios_lengths = np.array(audios_lengths)
    char_lens = end_time - start_time

    average_char_length = np.mean(char_lens).item()
    variance_char_length = np.var(char_lens).item()

    average_audio_length = np.mean(audios_lengths).item()
    variance_audio_length = np.var(audios_lengths).item()

    print(f"Average Char Length: {round(average_char_length, 3)}")
    print(f"Char Length Variance: {round(variance_char_length, 3)}")
    print(f"Average Audio Length: {round(average_audio_length, 3)}")
    print(f"Variance Audio Length: {round(variance_audio_length, 3)}")

    print(f"Maximum Char Length: {char_lens.max().item()}")
    print(f"Minimum Char Length: {char_lens.min().item()}")
    print(f"Maximum Audio Length: {audios_lengths.max().item()}")
    print(f"Minimum Audio Length: {audios_lengths.min().item()}")


if __name__ == "__main__":
    stats('./LibriSpeech/train-clean-100_ordered')