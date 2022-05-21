
import os
from pathlib import PurePath
from pydub import AudioSegment
import tqdm


def order_librispeech_dir(root):
    """
    :param root: The directory to order for the forced aligner.
    :return:
    """

    ordered_root = f'{root}_ordered'
    os.mkdir(ordered_root)

    speakers_ids = os.listdir(root)

    for speaker_id in tqdm.tqdm(speakers_ids, desc='Ordering the LibriSpeech data folder'):
        current_speaker_path = f'{root}/{speaker_id}'
        current_speaker_ordered_path = f'{ordered_root}/{speaker_id}'
        os.mkdir(current_speaker_ordered_path)

        texts_ids = os.listdir(current_speaker_path)

        for text_id in texts_ids:
            current_text_path = f'{current_speaker_path}/{text_id}'

            files = os.listdir(current_text_path)

            text_file = f"{current_text_path}/{[file for file in files if file.endswith('txt')][0]}"
            transcriptions = {line.split()[0]: ' '.join(line.split()[1:])
                              for line in open(text_file, 'r').readlines()}

            audio_files = [file for file in files if not file.endswith('txt')]
            for audio_file in audio_files:
                current_audio_file_path = f'{current_text_path}/{audio_file}'
                audio_id = audio_file.split('.')[0]
                transcript = transcriptions[audio_id]

                ordered_transcript_filepath = f"{current_speaker_ordered_path}/{audio_id}.txt"
                open(ordered_transcript_filepath, "w").write(transcript)

                file_path = PurePath(current_audio_file_path)
                flac_tmp_audio_data = AudioSegment.from_file(file_path, file_path.suffix[1:])
                current_wav_audio_filepath = file_path.name.replace(file_path.suffix, "") + ".wav"
                flac_tmp_audio_data.export(current_wav_audio_filepath, format="wav")

                os.rename(current_wav_audio_filepath,
                          f"{current_speaker_ordered_path}/{audio_file.replace('.flac', '.wav')}")



def order_alignments_dir(alignment_dir_root):
    """

    :param alignment_dir_root: Has to be a path of a directory with alignments generated from
                                this repo: https://github.com/CorentinJ/librispeech-alignments.git
    :return:
    """

    def reformat_row(row: str):
        data = [i.strip() for i in row.split("\"")]
        data = [i for i in data if i != '']
        audio_id = data[0].split('-')[-1]
        speaker_id = data[0].split('-')[0]
        story_id = data[0].split('-')[1]

        words = data[1][1:-1].split(',')
        starting_times = [float(t) for t in data[2].split(',')][:-1]

        aligns = []

        for i, word in enumerate(words):
            starting_time = starting_times[i]
            ending_time = starting_times[i + 1]

            if len(word) == 0:
                # word == ''
                assert word == ''
                # aligns.append(('|', starting_time, ending_time))
            else:
                word += '|'
                char_len = (ending_time - starting_time) / len(word)
                curr_starting_time = starting_time
                for char in word:
                    aligns.append((char, starting_time, starting_time + char_len))
                    starting_time += char_len

        return {
            'audio_id': audio_id,
            'alignments': aligns,
            'alignment_string': '\n'.join([f"{c} | {s} | {e}" for c, s, e in aligns])
        }

    for subset_dir in os.listdir(alignment_dir_root):
        for speaker_id in os.listdir(f"{alignment_dir_root}/{subset_dir}"):
            for story_id in os.listdir(f"{alignment_dir_root}/{subset_dir}/{speaker_id}"):
                curr_path = f"{alignment_dir_root}/{subset_dir}/{speaker_id}/{story_id}"
                alignment_file = [f for f in os.listdir(curr_path) if len(f.split('.')) >= 2][0]
                alignment_file = f"{curr_path}/{alignment_file}"
                alignment = open(alignment_file, 'r').readlines()

                for r in alignment:
                    reformatted_row_alignment = reformat_row(r)
                    audio_id = reformatted_row_alignment['audio_id']
                    audio_filepath = f"{curr_path}/{audio_id}"
                    open(audio_filepath, 'w').write(reformatted_row_alignment['alignment_string'])


def join_alignments_and_audio(alignment_dir_root, ordered_librispeech_dir_root, relevant_subsets):

    for subset_dir in os.listdir(alignment_dir_root):
        if subset_dir not in relevant_subsets:
            continue

        for speaker_id in os.listdir(f"{alignment_dir_root}/{subset_dir}"):
            for story_id in os.listdir(f"{alignment_dir_root}/{subset_dir}/{speaker_id}"):
                for audio_id in os.listdir(f"{alignment_dir_root}/{subset_dir}/{speaker_id}/{story_id}"):
                    curr_path = f"{alignment_dir_root}/{subset_dir}/{speaker_id}/{story_id}/{audio_id}"

                    os.rename(curr_path,
                              f"{ordered_librispeech_dir_root}/{speaker_id}/{speaker_id}-{story_id}-{audio_id}._aligned_with_mfa.txt")


def correct():

    train_500_speakers = os.listdir('./LibriSpeech/train-other-500_ordered')
    s = os.listdir("./LibriSpeech/train-clean-100+360+500_ordered")

    for dir in train_500_speakers:
        if dir in s:
            print(dir)
            os.rename(f"./LibriSpeech/train-clean-100+360+500_ordered/{dir}", f"./LibriSpeech/train-other-500/{dir}")


if __name__ == "__main__":

    # order_librispeech_dir(root='./LibriSpeech/train-other-500')

    join_alignments_and_audio(alignment_dir_root='./LibriSpeech - Alignments',
                              ordered_librispeech_dir_root='./LibriSpeech/train-clean-100+360+500_ordered',
                              relevant_subsets=['train-other-500'])

    # correct()

    # order_alignments_dir(alignment_dir_root='./LibriSpeech - Alignments')

    # TODO: Align the data using Wav2Vec, check again all the code, test it, and train!
