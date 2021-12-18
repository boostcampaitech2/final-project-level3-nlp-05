import soundfile as sf
import numpy as np
import pandas as pd
import json
import os
import timeit
import shutil

import tensorflow as tf

from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel
from datetime import date, timedelta

from tqdm import tqdm

#audio segment
from pydub import AudioSegment
from change_honorific import honorific_token_check, change_text

import argparse

parser = argparse.ArgumentParser()

dict_categories = {
        "economic":"경제",
        "society": "사회",
        "politics":"정치",
        "digital":"IT",
        "foreign":"국제", 
        "culture": "문화",
        "entertain":"연예", 
        "sports":"스포츠", 
    }

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
    
    parser.add_argument('--pause_long', 
        default = '2000',
        type=int,
        help="long pause between sentences")
        
    parser.add_argument('--pause_short', 
        default = '2000',
        type=int,
        help="short pause between comments")

    parser.add_argument('--split_length', 
        default = '300',
        type=int,
        help="split_length")
    
    parser.add_argument('--date', 
        default = (date.today() - timedelta(1)).strftime("%Y%m%d"), # '20211216'
        type=int,
        help="date of news summary")

    args = parser.parse_args()
    return args


def generate_conjunction(text2mel_model, text2mel_processor, mel2wav_model, sample_rate, dict_categories):
    """ drop conjunction audio files  """
    
    lines = []
    start, next, final, ending = ['처음은', '다음은', '마지막으로', '뉴스입니다.']
    for idx, value in enumerate(dict_categories.values()):
        if idx == 0: lines.append(f'{start} {value} {ending}')
        elif idx == len(dict_categories)-1: lines.append(f'{final} {value} {ending}')
        else: lines.append(f'{next} {value} {ending}')

    for category,lines in zip(dict_categories.keys(), lines):
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
        sf.write(f"./data/conjunction/{category}_0.wav", audio, sample_rate, "PCM_16")
    

def text_fill(date):
    """ retrieve opening statement for specific date """
    year,month,day = date[:4], date[4:6], date[6:]
    opening = f'안녕하세요 {year}년 {month}월 {day}일자 핵심뉴스 요약입니다.'
    return opening 


def audio_drop(summary, date, id, text2mel_model, text2mel_processor, mel2wav_model, sample_rate, split=False):
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
    if not split:
        sf.write(f'./data/{date}/tts/voice_files/{id}.wav', audio, sample_rate, "PCM_16")
    else :
        if not os.path.isdir(f"./data/{args.date}/tts/voice_files/temp"):
            os.makedirs(f"./data/{args.date}/tts/voice_files/temp")
        sf.write(f'./data/{date}/tts/voice_files/temp/{id}.wav', audio, sample_rate, "PCM_16")

##############################################################################
def main():
    # measure time
    start = timeit.default_timer()

    # parse args
    args = get_args()
    
    # load models
    text2mel_processor = AutoProcessor.from_pretrained(f"tensorspeech/tts-{args.text2mel}-kss-ko")
    text2mel_model = TFAutoModel.from_pretrained(f"tensorspeech/tts-{args.text2mel}-kss-ko")
    mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-kss-ko")
    print('#'*5, 'Model Loaded' ,'#'*5)
    print(f'Selected {args.text2mel} for text2mel...')

    # prepare conjunction files 
    if os.path.isdir("./data/conjunction/"):
        pass
    else:
        os.makedirs("./data/conjunction/")
        generate_conjunction(text2mel_model, text2mel_processor, mb_melgan, args.sample_rate, dict_categories)


    # load json args.date for the name
    with open(f"./data/{args.date}/summary.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # change to honorific
    for idx, summary in data['summary'].items():
        result = ''
        text_split = summary.split()
        if honorific_token_check(text_split[-1]): # check if honorific
            text_split[-1] = change_text(text_split[-1])
            result += " ".join(text_split)  
            data['summary'][idx] = result

    # check dir
    if not os.path.isdir(f"./data/{args.date}/tts/voice_files/"):
        os.makedirs(f"./data/{args.date}/tts/voice_files/")
    
    # audio drop opening statement
    opening = text_fill(args.date)
    audio_drop(opening, args.date, 'opening_',text2mel_model, text2mel_processor, mb_melgan, args.sample_rate)
    
    # split summary text if larger than split_length
    for idx,summary in tqdm(data['summary'].items()):
        if len(summary) > args.split_length :
            for i in range((len(summary)//args.split_length)+1):
                if args.split_length*(i+1) < len(summary):
                    summary_split = summary[args.split_length*i:args.split_length*(i+1)]
                else :
                    summary_split = summary[args.split_length*i:]
                audio_drop(
                    summary = summary_split, 
                    date = args.date, 
                    id = data['id'][idx]+f"-{i}", 
                    text2mel_model = text2mel_model, 
                    text2mel_processor = text2mel_processor, 
                    mel2wav_model = mb_melgan, 
                    sample_rate = args.sample_rate,
                    split = True
                )
            split_sounds = []
            path = sorted(os.listdir(f"./data/{args.date}/tts/voice_files/temp"))
            split_sounds.append(AudioSegment.from_file(os.path.join(f'./data/{args.date}/tts/voice_files/temp',path), format="wav"))
            split_overlay = sum(split_sounds)
            split_overlay.export(f"../data/{args.date}/tts/voice_files/{data['id'][idx]}.wav", format="wav")
            shutil.rmtree(f"./data/{args.date}/tts/voice_files/temp")

        else :
            audio_drop(
                summary = summary, 
                date = args.date, 
                id = data['id'][idx],
                text2mel_model = text2mel_model, 
                text2mel_processor = text2mel_processor, 
                mel2wav_model = mb_melgan, 
                sample_rate = args.sample_rate
            )

        print('###### Summary of article - ',data['id'][idx], '######')
        print('Summary Input...')
        print(summary,'\n','-'*100)
    
    

    # make ordered list of wav files
    ordered_list = ['opening_.wav']
    file_list = os.listdir(f"./data/{args.date}/tts/voice_files/") + os.listdir(f'./data/conjunction')
    for category in dict_categories.keys() :
        for audio in sorted(file_list):
            if category in audio:
                ordered_list.append(audio)
    
    # concat split audios
    split_sounds = []
    for path in ordered_list:
        if '-' in path:
            split_sounds.append(AudioSegment.from_file(os.path.join(f'./data/{args.date}/tts/voice_files/',path), format="wav"))
            split_overlay = sum(split_sounds)
            split_overlay.export(f"./data/{args.date}/tts/voice_files/{args.date}.wav", format="wav")
    # 1 2 3 4-1 4-2 5 6 7-1 7-2 8 9
    # concat wav files with corresponding silence
    sounds = []
    silence_long = AudioSegment.silent(duration=args.pause_long)
    silence_short= AudioSegment.silent(duration=args.pause_short)
    
    for i, path in enumerate(ordered_list):
        if path.split('_')[1] == '0.wav':
            sounds.append(AudioSegment.from_file(os.path.join(f'./data/conjunction/',path), format="wav"))
            sounds.append(silence_short)
        else:
            sounds.append(AudioSegment.from_file(os.path.join(f'./data/{args.date}/tts/voice_files/',path), format="wav"))
            if path == 'opening_.wav':
                sounds.append(silence_short)
            else :
                # 여기 바꿔야함 (노아님)
                current_category = path.split('_')[0]
                if i < len(ordered_list)-1:
                    next_category = ordered_list[i + 1].split('_')[0]
                    if current_category != next_category:
                        sounds.append(silence_short)
                    else :
                        sounds.append(silence_long)

    overlay = sum(sounds)
    overlay.export(f"./data/{args.date}/tts/final_{args.date}.wav", format="wav")
    print('#'*5, f"Final audio files generated at ./data/{args.date}/tts/ folder",'#'*5)

    execution_time = timeit.default_timer() - start
    print(f"Program Executed in {execution_time:.2f}s", '\n')

##############################################################################
if __name__ == "__main__":
    main()

