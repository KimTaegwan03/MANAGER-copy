import os
import subprocess
import pandas as pd
import whisper
from tqdm import tqdm

def download_video(url, output_path):
    """
    주어진 url에서 비디오를 다운로드하여 출력 경로에 저장합니다.

    매개변수:
    url (str): 다운로드할 비디오의 URL입니다.
    output_path (str): 비디오를 저장할 경로입니다.

    반환값:
    dict: 비디오의 메타데이터가 포함된 사전입니다.
    """
    # 경로 생성
    os.makedirs(output_path, exist_ok=True)

    

    # 비디오 재생 시간 확인
    command = f'yt-dlp --get-duration --cookies-from-browser firefox {url}'
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    duration = result.stdout.strip()

    # 재생 시간이 1시간 이상인 경우 함수 종료
    time_parts = duration.split(':')
    if len(time_parts) == 3:
        hours, minutes, seconds = map(int, time_parts)
    elif len(time_parts) == 2:
        hours = 0
        minutes, seconds = map(int, time_parts)
    elif len(time_parts) == 1:
        hours = 0
        minutes = 0
        seconds = int(time_parts[0])
    else:
        raise ValueError("Unexpected duration format")
    total_seconds = hours * 3600 + minutes * 60 + seconds

    return total_seconds

    if total_seconds > 3600:
        return total_seconds
    
    else:
        # yt-dlp 명령 실행
        try:
            command = f'yt-dlp --force-overwrites -f "bestaudio" --cookies-from-browser firefox -o "{output_path}/video" {url}'
            subprocess.run(command, shell=True, check=True)

            return total_seconds
        except:
            return 3601

whisperModel = whisper.load_model("base")

df_cnn = pd.read_csv("matching_questions_politics_economy.csv")
df_fox = pd.read_csv("matching_questions_politics_economy_fox.csv")

df_concat = pd.concat([df_cnn, df_fox], ignore_index=True)

concat_url = df_concat['url'].unique().tolist()
concat_questions = df_concat['matching_questions'].unique().tolist()

print(len(concat_url))
print(len(concat_questions))
print(len(df_concat))

duration_list = []

for url in tqdm(concat_url):
    output_path = f"output"
    duration = download_video(url, output_path)
    duration_list.append(duration)

df_duration = pd.DataFrame(duration_list, columns=['duration'])
df_duration.to_csv('video_durations.csv', index=False)

print(df_duration.describe())
