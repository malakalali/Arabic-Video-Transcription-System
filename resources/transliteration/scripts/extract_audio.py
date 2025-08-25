#!/usr/bin/env python3
import argparse
import sys
import subprocess
from pathlib import Path


def run_ffmpeg_extract(video_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(audio_path),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed for {video_path} -> {audio_path}") from exc


def extract_single(video_path: Path, output_dir: Path) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    stem = video_path.stem
    audio_path = output_dir / f"{stem}.wav"
    run_ffmpeg_extract(video_path, audio_path)
    return audio_path


def extract_batch(input_dir: Path, output_dir: Path, exts=(".mp4", ".mov", ".mkv", ".avi")) -> list[Path]:
    videos: list[Path] = []
    for ext in exts:
        videos.extend(sorted(input_dir.glob(f"**/*{ext}")))
    if not videos:
        print(f"No videos found in {input_dir}")
        return []
    outputs: list[Path] = []
    for video in videos:
        out = extract_single(video, output_dir)
        print(f"Extracted: {video} -> {out}")
        outputs.append(out)
    return outputs


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Extract audio to 16kHz mono WAV using ffmpeg.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Path to a single video file")
    group.add_argument("--videos-dir", type=Path, help="Directory containing video files")
    parser.add_argument("--out-dir", type=Path, default=Path("audio"), help="Output directory for WAV files")
    args = parser.parse_args(argv)

    try:
        if args.video:
            out = extract_single(args.video, args.out_dir)
            print(f"Saved: {out}")
        else:
            outs = extract_batch(args.videos_dir, args.out_dir)
            print(f"Finished. {len(outs)} files written to {args.out_dir}")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
