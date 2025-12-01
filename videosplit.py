import os
import re
import moviepy.editor as mp
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from tqdm import tqdm


def split_video_by_pause(video_path, output_base_folder, dataset_type="dev", min_silence_len=2000, silence_thresh=-50,
                         min_speech_len=3000, max_segments=14):
    # 从视频文件名提取基础名称（如 223_1）
    video_basename = re.sub(r'_cut_combined.*', '', os.path.basename(video_path))
    video_output_folder = os.path.join(output_base_folder, dataset_type, video_basename)

    # 加载视频并提取音频
    video = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, codec="pcm_s16le")

    # 加载音频并检测非静音部分
    audio = AudioSegment.from_wav(audio_path)
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # 过滤掉小于 min_speech_len 的片段，确保连续 5 秒以上才视为有效语音
    filtered_ranges = [r for r in nonsilent_ranges if (r[1] - r[0]) >= min_speech_len]

    # 计算切割点
    cut_points = []
    for i in range(1, len(filtered_ranges)):
        prev_end = filtered_ranges[i - 1][1]
        next_start = filtered_ranges[i][0]
        if next_start - prev_end > min_silence_len:
            cut_points.append(prev_end / 1000.0)  # 转换为秒

    # 确定是否标记父目录
    if len(cut_points) + 1 > max_segments:
        video_output_folder += "_*"
    elif len(cut_points) + 1 < 12:
        video_output_folder += "_<"
    os.makedirs(video_output_folder, exist_ok=True)

    # 视频切割并保存，加入分割进度条
    start = 0
    for i, end in enumerate(tqdm(cut_points + [video.duration], desc=f"Processing {video_basename}", unit="segment")):
        segment_filename = f"{video_basename}_task{i + 1}.mp4"
        segment_path = os.path.join(video_output_folder, segment_filename)

        subclip = video.subclip(start, end)
        subclip.write_videofile(segment_path, codec="libx264", audio_codec="aac")
        start = end

    print(f"视频 {video_basename} 切割完成！输出路径: {video_output_folder}")


def batch_process_videos(input_folder, output_base_folder, min_silence_len=3500, silence_thresh=-41.5,
                         min_speech_len=7000, max_segments=14):
    # 限制仅处理 dev 文件夹中的视频文件
    video_files = [
        (root, file)
        for root, _, files in os.walk(os.path.join(input_folder, "train"))
        for file in files if file.endswith(".mp4")
    ]

    for root, file in tqdm(video_files, desc="Batch Processing Videos in 'dev'", unit="video"):
        video_path = os.path.join(root, file)
        split_video_by_pause(video_path, output_base_folder, "train", min_silence_len, silence_thresh, min_speech_len,
                             max_segments)


# 设置路径并批量处理
input_folder = "/Users/cai/Documents/实验/DATASET/AVEC2013"
output_base_folder = "/Users/cai/Documents/实验/DATASET/AVEC2013_tasks"
batch_process_videos(input_folder, output_base_folder, max_segments=15)
