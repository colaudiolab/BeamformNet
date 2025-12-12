import os
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import webrtcvad
from scipy.signal import hilbert
from tqdm import tqdm


def process_angle(angle):
    if angle >= 180:
        angle = 360 - angle
    angle = angle - 90

    return angle

#读取csv文件中的file然后取关注的10个通道 以采样率16000读取 还要读取方位角
root_path = Path('./RealMAN_data')
csv_path = Path('./val_moving_source_location.csv')
save_path = Path('./RealMAN_data/process')
is_static = 'static' in csv_path.name
sr_target = 16000
frame_duration_ms = 30
mode = 2
vad = webrtcvad.Vad(mode)

df = pd.read_csv(root_path / csv_path)
result = df[['filename', 'angle(°)']]
for row in tqdm(result.itertuples(), total=len(result)):
    filename = root_path / Path(row.filename).parent / Path(row.filename).stem
    if is_static:
        angle = process_angle(row._2)
    else:
        angle = [process_angle(float(i)) for i in row._2.split(',')]

    audio_data_CH0, sample_rate = librosa.load(f'{filename}_CH0.flac', sr=sr_target)
    audio_data_CH1, _ = librosa.load(f'{filename}_CH1.flac', sr=sr_target) # sr=None保持原始采样率
    audio_data_CH17, _ = librosa.load(f'{filename}_CH17.flac', sr=sr_target)  # sr=None保持原始采样率
    audio_data_CH26, _ = librosa.load(f'{filename}_CH26.flac', sr=sr_target)  # sr=None保持原始采样率
    audio_data_CH27, _ = librosa.load(f'{filename}_CH27.flac', sr=sr_target) # sr=None保持原始采样率
    audio_data_CH5, _ = librosa.load(f'{filename}_CH5.flac', sr=sr_target) # sr=None保持原始采样率
    audio_data_CH13, _ = librosa.load(f'{filename}_CH13.flac', sr=sr_target)  # sr=None保持原始采样率
    audio_data_CH21, _ = librosa.load(f'{filename}_CH21.flac', sr=sr_target)  # sr=None保持原始采样率
    audio_data_CH25, _ = librosa.load(f'{filename}_CH25.flac', sr=sr_target)  # sr=None保持原始采样率
    audio_data_CH9, _ = librosa.load(f'{filename}_CH9.flac', sr=sr_target)  # sr=None保持原始采样率

    waveform = np.array([audio_data_CH25,audio_data_CH21,audio_data_CH13,audio_data_CH5,audio_data_CH0,audio_data_CH1,audio_data_CH9,audio_data_CH17,audio_data_CH26,audio_data_CH27])
    waveform = hilbert(waveform, axis=1)

    audio_int16 = (audio_data_CH0 * 32767).astype(np.int16)

    frame_size = int(sr_target * frame_duration_ms / 1000)

    try:
        for i in range(0, len(audio_int16), frame_size):
            #如果文件存在跳过

            frame = audio_int16[i:i + frame_size]

            if len(frame) != frame_size: # 只保留完整帧
                continue

            is_speech = vad.is_speech(frame.tobytes(), sample_rate=sr_target)

            if is_speech:
                if is_static:
                    label_angle = angle
                else:
                    #根据标签插值方位角
                    start_time = (i / frame_size) * frame_duration_ms
                    end_time = start_time + frame_duration_ms

                    label_index = int(start_time / 100)
                    if end_time >= (label_index + 1) * 100: #横跨标签的frame标签就为横跨的
                        label_angle = angle[label_index]
                    if label_index == 0:
                        label_angle = angle[label_index]
                    else:
                        label_angle = angle[label_index - 1] + (angle[label_index] - angle[label_index - 1]) / 100 * (end_time - label_index * 100)


                #保存文件
                real_frame = waveform[:,i:i + frame_size]
                file_save_path = save_path / f'{filename.stem}_{int(start_time)}_{label_angle:.4f}.npy'
                if not os.path.exists(file_save_path):
                    np.save(file_save_path, real_frame)

            else:
                continue
    except:
        print(f'error: {filename}')
        continue
