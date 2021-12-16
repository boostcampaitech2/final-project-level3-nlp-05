import soundfile as sf
import numpy as np
import pandas as pd
import json
import os
import timeit

import tensorflow as tf

from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel
from tqdm import tqdm
#audio segment
from pydub import AudioSegment

import argparse

parser = argparse.ArgumentParser()

def get_args():
    """ retrieve arguments for tts inference """
    parser = argparse.ArgumentParser(description="Args for TTS Generation")
    parser.add_argument('--text2mel', 
        choices = ['tacotron2', 'fastspeech2'],
        default = 'tacotron2',
        type=str,
        help="text2mel model")

    parser.add_argument('--sample_rate', 
        default = '22050',
        type=int,
        help="sample rate")

    parser.add_argument('--pause', 
        default = '2000',
        type=int,
        help="pause between audio files")

    args = parser.parse_args()
    return args


def generate_conjunction(text2mel_model, text2mel_processor, mel2wav_model, sample_rate):
    """ drop conjunction audio files  """
    lines = ['처음은 경제 뉴스입니다.','다음은 사회 뉴스입니다.','마지막으로 정치 뉴스입니다.']

    for category,lines in zip(['economics', 'social', 'politics'], lines):
        input_ids = text2mel_processor.text_to_sequence(lines)
        if text2mel_model == 'fastspeech2':
            _, mel_output, _, _, _ = text2mel_model.inference(
                input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                speaker_ids=tf.convert_to_tensor([1], dtype=tf.int32),
                speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
                energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
            )
        else: 
            _, mel_output,_,_ = text2mel_model.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            )
        audio = mel2wav_model.inference(mel_output)[0, :, 0]
        sf.write(f"./voice_files/conjunction/{category}_0.wav", audio, sample_rate, "PCM_16")
    

def text_fill(date):
    """ retrieve opening statement for specific date """
    year,month,day = date.split('-')
    opening = f'안녕하세요 {year}년 {month}월 {day}일자 핵심뉴스 요약입니다.'
    return opening 


def audio_drop(summary, date, id, text2mel_model, text2mel_processor, mel2wav_model, sample_rate):
    """ audio drop using text2mel & mel2wav """
    input_ids = text2mel_processor.text_to_sequence(summary)
    if text2mel_model == 'fastspeech2':
        _, mel_output, _, _, _ = text2mel_model.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([1], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
            energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
        )
    else: 
        _, mel_output,_,_ = text2mel_model.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    )
    audio = mel2wav_model.inference(mel_output)[0, :, 0]
    sf.write(f'./voice_files/{date}/{id}.wav', audio, sample_rate, "PCM_16")


##############################################################################
def main():
    # measure time
    start = timeit.default_timer()

    # parse args
    args = parser.parse_args()
    args = get_args()
    
    # load models
    text2mel_processor = AutoProcessor.from_pretrained(f"tensorspeech/tts-{args.text2mel}-kss-ko")
    text2mel_model = TFAutoModel.from_pretrained(f"tensorspeech/tts-{args.text2mel}-kss-ko")
    mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-kss-ko")
    print('#'*5, 'Model Loaded' ,'#'*5)
    print(f'Selected {args.text2mel} for text2mel...')

    # prepare conjunction files 
    if os.path.isdir("./voice_files/conjunction/"):
        pass
    else:
        os.mkdir("./voice_files/conjunction/")
        generate_conjunction(text2mel_model, text2mel_processor, mb_melgan, args.sample_rate)


    # load json
    with open(f"./summary.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not os.path.isdir(f'./voice_files/{data["date"]}'):
        os.mkdir(f'./voice_files/{data["date"]}')
        
    opening = text_fill(data['date'])
    audio_drop(opening, data['date'], 'opening_',text2mel_model, text2mel_processor, mb_melgan, args.sample_rate)
    
    for idx,summary in tqdm(data['summary'].items()):
        audio_drop(summary, data['date'], data['id'][idx],text2mel_model, text2mel_processor, mb_melgan, args.sample_rate)
        print('###### Summary of article - ',data['id'][idx], '######')
        print('Summary Input...')
        print(summary,'\n','-'*100)

    # audio concat
    ordered_list = ['opening_.wav']
    file_list = os.listdir(f'./voice_files/{data["date"]}') + os.listdir(f'./voice_files/conjunction')
    category_list = ['economics','social', 'politics']
    for category in category_list :
        for audio in sorted(file_list):
            if category in audio:
                ordered_list.append(audio)    
    sounds = []
    silence = AudioSegment.silent(duration=args.pause)
    for path in ordered_list:
        if path.split('_')[1] == '0.wav':
            sounds.append(AudioSegment.from_file(os.path.join(f'./voice_files/conjunction/',path), format="wav"))
        else:
            sounds.append(AudioSegment.from_file(os.path.join(f'./voice_files/{data["date"]}/',path), format="wav"))
        sounds.append(silence)

    overlay = sum(sounds)

    if not os.path.isdir('./tts_output/'):
        os.mkdir('./tts_output/')

    overlay.export(f"./tts_output/final_{data['date']}.wav", format="wav")
    print('#'*5, 'Final audio files generated at ./tts_output folder','#'*5)

    execution_time = timeit.default_timer() - start
    print(f"Program Executed in {execution_time:.2f}s", '\n')


##############################################################################
if __name__ == "__main__":
    main()

### 추가 구현
# 1. 그냥 긴 요약문 안전하게 나눠서 concat -> 이거 필요없다는건가...? -> 하기
# 2. Category 추가하기 -> 하기 (일단)
# 3. pause 필요한데 -> 줄일필요도 + 다양하게 쉬기..?
# 4. directory root/data 수정 <- data/날짜/tts/voice_files (final 밖에)
# 5. 받는 json 파일 수정되면 맞춰서 수정
# 6. 경어: 했습니다 <-> 했다 
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 7. fine-tuning