import os
import ssl
import subprocess
from src.translation import load_whisper, transcribe_arabic_and_translate
from src.subtitles import (
    create_bilingual_srt,
    load_srt_segments_with_transliteration,
    create_bilingual_srt_from_existing,
    escape_for_ffmpeg_subtitles,
    align_english_to_arabic_segments,
)

# Temporary workaround: allow model download despite SSL issues
try:
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
except Exception:
    pass

def extract_audio(video_path, audio_output_path):
    """
    Extracts audio from a video file using ffmpeg.
    """
    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-af', 'loudnorm',
        audio_output_path
    ]
    subprocess.run(command, check=True)

def transcribe_arabic_and_translate(model, audio_path: str):
    # shim import retained for backward compatibility; real implementation in src.translation
    from src.translation import transcribe_arabic_and_translate as _fn
    return _fn(model, audio_path)





def transliterate_arabic_to_latin(text: str) -> str:
    from src.subtitles import transliterate_arabic_to_latin as _tl
    return _tl(text)





def create_bilingual_srt(arabic_result, english_result, srt_output_path):
    from src.subtitles import create_bilingual_srt as _create
    return _create(arabic_result, english_result, srt_output_path)


def _parse_timecode(tc: str) -> float:
    hh, mm, rest = tc.split(":", 2)
    ss, ms = rest.split(",", 1)
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def load_srt_segments_with_transliteration(srt_path: str):
    """
    Parse an SRT where each block has two text lines:
    1) Arabic
    2) Transliteration (e.g., Buckwalter)
    Returns list of {start, end, arabic, transliteration}
    """
    segments = []
    with open(srt_path, "r", encoding="utf-8") as f:
        block = []
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if block:
                    if len(block) >= 3:
                        try:
                            times = block[1]
                            start_tc, end_tc = [t.strip() for t in times.split("-->")]
                            start = _parse_timecode(start_tc)
                            end = _parse_timecode(end_tc)
                            text_lines = [l for l in block[2:] if l.strip()]
                            arabic = text_lines[0] if len(text_lines) >= 1 else ""
                            translit = text_lines[1] if len(text_lines) >= 2 else ""
                            segments.append({
                                "start": start,
                                "end": end,
                                "arabic": arabic,
                                "transliteration": translit,
                            })
                        except Exception:
                            pass
                block = []
            else:
                block.append(line)
        # flush last block
        if block and len(block) >= 3:
            try:
                times = block[1]
                start_tc, end_tc = [t.strip() for t in times.split("-->")]
                start = _parse_timecode(start_tc)
                end = _parse_timecode(end_tc)
                text_lines = [l for l in block[2:] if l.strip()]
                arabic = text_lines[0] if len(text_lines) >= 1 else ""
                translit = text_lines[1] if len(text_lines) >= 2 else ""
                segments.append({
                    "start": start,
                    "end": end,
                    "arabic": arabic,
                    "transliteration": translit,
                })
            except Exception:
                pass
    return segments


def burn_subtitles_into_video(video_path, srt_path, output_path):
    escaped = escape_for_ffmpeg_subtitles(srt_path)
    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vf', f"subtitles={escaped}:force_style='Fontsize=24,Fontname=Arial Unicode MS'",
        '-c:a', 'copy',
        output_path
    ]
    subprocess.run(command, check=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process Arabic video with translation and transliteration')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output_folder', type=str, default='output_videos', help='Folder to save the output video and subtitles')
    parser.add_argument('--model_size', type=str, default='small', help='Whisper model size (tiny, base, small, medium, large)')
    parser.add_argument('--existing_srt_dir', type=str, default='', help='Directory to search for existing SRT with transliteration')

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Extract audio
    audio_output_path = os.path.join(args.output_folder, os.path.splitext(os.path.basename(args.video_path))[0] + '.wav')
    extract_audio(args.video_path, audio_output_path)

    # Load model
    model = load_whisper(args.model_size)

    # Transcribe and translate
    arabic_result, english_result = transcribe_arabic_and_translate(model, audio_output_path)

    # Try to use existing SRT transliteration for better sync
    srt_output_path = os.path.join(args.output_folder, os.path.splitext(os.path.basename(args.video_path))[0] + '.srt')

    used_existing = False
    search_dirs = [
        args.existing_srt_dir,
        os.path.join('resources', 'transliteration', 'subs'),
        os.path.join('resources', 'transliteration', 'subs'),
    ]
    for dir_path in search_dirs:
        if not dir_path:
            continue
        if os.path.isdir(dir_path):
            for name in os.listdir(dir_path):
                if name.lower().endswith('.srt'):
                    candidate = os.path.join(dir_path, name)
                    try:
                        ar_segments = load_srt_segments_with_transliteration(candidate)
                        if ar_segments:
                            create_bilingual_srt_from_existing(ar_segments, english_result, srt_output_path)
                            used_existing = True
                            break
                    except Exception:
                        pass
        if used_existing:
            break

    if not used_existing:
        create_bilingual_srt(arabic_result, english_result, srt_output_path)

    # Burn subtitles
    output_video_path = os.path.join(args.output_folder, os.path.splitext(os.path.basename(args.video_path))[0] + '_subtitled.mp4')
    burn_subtitles_into_video(args.video_path, srt_output_path, output_video_path)

if __name__ == '__main__':
    main()


