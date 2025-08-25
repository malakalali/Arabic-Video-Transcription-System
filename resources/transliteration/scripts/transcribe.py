#!/usr/bin/env python3
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path
from datetime import timedelta
from typing import Iterable, Optional

from faster_whisper import WhisperModel


arabic_to_latin = {
    "ا": "a",
    "أ": "a",
    "إ": "i",
    "آ": "aa",
    "ب": "b",
    "ت": "t",
    "ث": "th",
    "ج": "j",
    "ح": "ḥ",
    "خ": "kh",
    "د": "d",
    "ذ": "dh",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "ṣ",
    "ض": "ḍ",
    "ط": "ṭ",
    "ظ": "ẓ",
    "ع": "ʿ",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "و": "w",
    "ي": "y",
    "ى": "a",
    "ء": "'",
    "ئ": "'",
    "ؤ": "'",
    "ة": "h",  # or "t" depending on context
}


def buckwalter_transliterate(text: str) -> str:
    return "".join(arabic_to_latin.get(ch, ch) for ch in text)


def format_timestamp(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    delta = timedelta(seconds=seconds)
    # SRT format: HH:MM:SS,mmm
    total_ms = int(delta.total_seconds() * 1000)
    hours, rem_ms = divmod(total_ms, 3600 * 1000)
    minutes, rem_ms = divmod(rem_ms, 60 * 1000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(segments: Iterable, srt_path: Path, add_transliteration: bool) -> None:
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    with srt_path.open("w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            text = seg.text.strip()
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            if add_transliteration:
                f.write(text + "\n")
                f.write(buckwalter_transliterate(text) + "\n\n")
            else:
                f.write(text + "\n\n")


def transcribe_file(model: WhisperModel, audio_path: Path, out_dir: Path, language: str,
                    add_transliteration: bool, vad_filter: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_path = out_dir / (audio_path.stem + ".srt")
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=vad_filter,
        beam_size=5,
        condition_on_previous_text=True,
        word_timestamps=False,
    )
    write_srt(segments, srt_path, add_transliteration)
    return srt_path


def transcribe_batch(model: WhisperModel, audio_dir: Path, out_dir: Path, language: str,
                     add_transliteration: bool, vad_filter: bool) -> list[Path]:
    wavs = sorted(list(audio_dir.glob("**/*.wav")))
    outs: list[Path] = []
    for wav in wavs:
        srt = transcribe_file(model, wav, out_dir, language, add_transliteration, vad_filter)
        print(f"Wrote: {srt}")
        outs.append(srt)
    return outs


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio to SRT using faster-whisper.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=Path, help="Path to a single .wav file")
    group.add_argument("--audio-dir", type=Path, help="Directory containing .wav files")
    parser.add_argument("--out-dir", type=Path, default=Path("subs"), help="Output directory for SRT files")
    parser.add_argument("--model", type=str, default="small", help="Whisper model size (e.g., tiny, base, small, medium, large-v3)")
    parser.add_argument("--language", type=str, default="ar", help="Language code (default: ar)")
    parser.add_argument("--compute-type", type=str, default="int8", help="CTranslate2 compute type (e.g., int8, int8_float16, float16, float32)")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    parser.add_argument("--transliteration", action="store_true", help="Include Buckwalter transliteration lines in SRT")
    args = parser.parse_args(argv)

    print(f"Loading model {args.model} (compute_type={args.compute_type})...")
    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)

    try:
        if args.audio:
            out = transcribe_file(model, args.audio, args.out_dir, args.language, args.transliteration, not args.no_vad)
            print(f"Saved: {out}")
        else:
            outs = transcribe_batch(model, args.audio_dir, args.out_dir, args.language, args.transliteration, not args.no_vad)
            print(f"Finished. {len(outs)} files written to {args.out_dir}")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
