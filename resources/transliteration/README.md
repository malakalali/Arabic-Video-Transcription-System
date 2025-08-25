# Transliteration Resources

This directory contains organized resources for Arabic transcription and transliteration.

## Directory Structure

```
resources/transliteration/
├── audio/           # Audio files (WAV format)
├── logs/            # Transcription logs
├── scripts/         # Utility scripts
├── subs/            # Subtitle files (SRT format)
├── videos/          # Video files
└── requirements.txt # Python dependencies
```

## Files

### Subtitle Files
- `subs/Adam Wa Mishmish.srt` - Example SRT with Arabic text and transliteration

### Scripts
- `scripts/transcribe.py` - Transcription script using faster-whisper
- `scripts/extract_audio.py` - Audio extraction utility using ffmpeg

### Audio
- `audio/Adam Wa Mishmish.wav` - Example audio file

### Logs
- `logs/transcribe_Adam.txt` - Transcription log file

## Usage

The main application (`main.py`) automatically searches this directory for existing SRT files to use for better subtitle synchronization and transliteration accuracy.

## Notes

- All paths in the main application have been updated to reference this organized structure
- The old `TransliterationBulBul/` directory has been consolidated here
- Only one virtual environment (`venv/`) is maintained at the project root
