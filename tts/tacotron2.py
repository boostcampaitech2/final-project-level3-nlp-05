import soundfile as sf
import numpy as np
import pandas as pd
import json
import os

import tensorflow as tf

from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel
from tqdm import tqdm
#audio segment
from pydub import AudioSegment

processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-kss-ko")
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-kss-ko")
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-kss-ko")


def text_fill(date):
    year,month,day = date.split('-')
    opening = f'안녕하세요 {year}년 {month}월 {day}일자 핵심뉴스 요약입니다.'
    return opening 

def audio_drop(summary, date, id):
    input_ids = processor.text_to_sequence(summary)

    mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([1], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
    
    audio = mb_melgan.inference(mel_after)[0, :, 0]
    sf.write(f'./voice_files/{date}/{id}.wav', audio, 20000, "PCM_16")

def main():
    
    # # json 가져오기
    with open(f"./summary.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not os.path.isdir(f'./voice_files/{data["date"]}'):
        os.mkdir(f'./voice_files/{data["date"]}')
        
    opening = text_fill(data['date'])
    audio_drop(opening, data['date'], 'opening_')
    
    for idx,summary in tqdm(data['summary'].items()):
        audio_drop(summary, data['date'], data['id'][idx])
    
    # audio concat
    ordered_list = ['opening_.wav']
    file_list = os.listdir(f'./voice_files/{data["date"]}') + os.listdir(f'./voice_files/conjunction')
    category_list = ['economics','social', 'politics']
    for category in category_list :
        for audio in sorted(file_list):
            if category in audio:
                ordered_list.append(audio)    
    sounds = []
    silence = AudioSegment.silent(duration=2000)
    for path in ordered_list:
        if path.split('_')[1] == '0.wav':
            sounds.append(AudioSegment.from_file(os.path.join(f'./voice_files/conjunction/',path), format="wav"))
        else:
            sounds.append(AudioSegment.from_file(os.path.join(f'./voice_files/{data["date"]}/',path), format="wav"))
        sounds.append(silence)

    overlay = sum(sounds)
    overlay.export(f"./final_{data['date']}.wav", format="wav")
    print('#'*5, 'Final audio files generated','#'*5)

    
if __name__ == "__main__":
    main()