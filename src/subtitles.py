import os
from datetime import timedelta


# Simple, readable Latin transliteration - easy to read and pronounce
arabic_to_latin = {
    # Basic letters - simple Latin equivalents
    "ا": "a", "أ": "a", "إ": "i", "آ": "aa", "ب": "b", "ت": "t", "ث": "th",
    "ج": "j", "ح": "h", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z",
    "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "a",
    "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "و": "w", "ي": "y", "ى": "a", "ء": "'", "ئ": "'", "ؤ": "'",
    "ة": "h",
    
    # Vowel marks - simple Latin vowels
    "َ": "a", "ُ": "u", "ِ": "i", "ّ": "", "ْ": "", "ً": "an", "ٌ": "un", "ٍ": "in",
    
    # Extended Arabic letters - use simple equivalents
    "ڪ": "k", "ګ": "k", "ڬ": "k", "ڭ": "k", "ڮ": "k", "ڰ": "k", "ڱ": "k", "ڲ": "k",
    "ڳ": "k", "ڴ": "k", "ڵ": "l", "ڶ": "l", "ڷ": "l", "ڸ": "l", "ڹ": "n", "ں": "n",
    "ڻ": "n", "ڼ": "n", "ڽ": "n", "ھ": "h", "ڿ": "h", "ہ": "h", "ۂ": "h", "ۃ": "h",
    "ۄ": "w", "ۅ": "w", "ۆ": "w", "ۇ": "w", "ۈ": "w", "ۉ": "w", "ۊ": "w", "ۋ": "w",
    "ی": "y", "ۍ": "y", "ێ": "y", "ې": "y", "ۑ": "y", "ے": "y", "ۓ": "y",
    
    # Numbers - keep as numbers
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    
    # Punctuation - use standard Latin punctuation
    "،": ",", "؛": ";", "؟": "?", "۔": ".", "ـ": "_"
}


def transliterate_arabic_to_latin(text: str) -> str:
    return "".join(arabic_to_latin.get(ch, ch) for ch in text)


def format_timestamp(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    delta = timedelta(seconds=seconds)
    total_ms = int(delta.total_seconds() * 1000)
    hours, rem_ms = divmod(total_ms, 3600 * 1000)
    minutes, rem_ms = divmod(rem_ms, 60 * 1000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def align_english_to_arabic_segments(ar_segments, en_segments):
    aligned_english_texts = []
    j = 0
    for ar in ar_segments:
        ar_start = ar.get("start", 0.0)
        ar_end = ar.get("end", ar_start)
        while j < len(en_segments) and en_segments[j].get("end", 0.0) <= ar_start:
            j += 1
        k = j
        collected = []
        while k < len(en_segments) and en_segments[k].get("start", 0.0) < ar_end + 0.01:
            txt = en_segments[k].get("text", "").strip()
            if txt:
                collected.append(txt)
            k += 1
        if not collected and j < len(en_segments):
            collected.append(en_segments[j].get("text", "").strip())
        aligned_english_texts.append(" ".join(t for t in collected if t))
    return aligned_english_texts


def create_bilingual_srt(arabic_result, english_result, srt_output_path):
    ar_segments = arabic_result.get("segments", [])
    en_segments = english_result.get("segments", [])
    en_texts_aligned = align_english_to_arabic_segments(ar_segments, en_segments)

    with open(srt_output_path, "w", encoding="utf-8") as f:
        for i, (ar_seg, en_text) in enumerate(zip(ar_segments, en_texts_aligned)):
            # Get precise timing with word-level accuracy if available
            start_time = format_timestamp(ar_seg.get("start", 0.0))
            end_time = format_timestamp(ar_seg.get("end", 0.0))
            
            # Get text content
            arabic_text = ar_seg.get("text", "").strip()
            english_text = (en_text or "").strip()
            
            # Use simple Latin transliteration for readability
            latin_transliteration = transliterate_arabic_to_latin(arabic_text)
            
            # Write subtitle block
            f.write(f"{i + 1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(english_text + "\n")
            f.write(latin_transliteration + "\n\n")
            
            # If we have word-level timestamps, create additional precise segments
            if ar_seg.get("words") and len(ar_seg["words"]) > 0:
                words = ar_seg["words"]
                for j, word in enumerate(words):
                    if word.get("text", "").strip():
                        word_start = format_timestamp(word.get("start", ar_seg.get("start", 0.0)))
                        word_end = format_timestamp(word.get("end", ar_seg.get("end", 0.0)))
                        word_text = word.get("text", "").strip()
                        word_translit = transliterate_arabic_to_latin(word_text)
                        
                        f.write(f"{i + 1}.{j + 1}\n")
                        f.write(f"{word_start} --> {word_end}\n")
                        f.write(word_text + "\n")
                        f.write(word_translit + "\n\n")


def _parse_timecode(tc: str) -> float:
    hh, mm, rest = tc.split(":", 2)
    ss, ms = rest.split(",", 1)
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def load_srt_segments_with_transliteration(srt_path: str):
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


def create_bilingual_srt_from_existing(ar_srt_segments, english_result, srt_output_path):
    en_segments = english_result.get("segments", [])
    ar_for_align = [{"start": seg["start"], "end": seg["end"]} for seg in ar_srt_segments]
    en_texts_aligned = align_english_to_arabic_segments(ar_for_align, en_segments)
    with open(srt_output_path, "w", encoding="utf-8") as f:
        for i, (ar_seg, en_text) in enumerate(zip(ar_srt_segments, en_texts_aligned)):
            start_time = format_timestamp(ar_seg.get("start", 0.0))
            end_time = format_timestamp(ar_seg.get("end", 0.0))
            transliteration = ar_seg.get("transliteration", "").strip()
            if not transliteration:
                transliteration = transliterate_arabic_to_latin(ar_seg.get("arabic", ""))
            f.write(f"{i + 1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write((en_text or "").strip() + "\n")
            f.write(transliteration + "\n\n")


def align_english_to_arabic_segments(ar_segments, en_segments):
    """Align English segments to Arabic segments by timing overlap."""
    en_texts = []
    
    for ar_seg in ar_segments:
        ar_start = ar_seg["start"]
        ar_end = ar_seg["end"]
        
        # Find English segment with best overlap
        best_overlap = 0
        best_en_text = ""
        
        for en_seg in en_segments:
            en_start = en_seg["start"]
            en_end = en_seg["end"]
            
            # Calculate overlap
            overlap_start = max(ar_start, en_start)
            overlap_end = min(ar_end, en_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_en_text = en_seg.get("text", "").strip()
        
        en_texts.append(best_en_text)
    
    return en_texts


def escape_for_ffmpeg_subtitles(path: str) -> str:
    return (
        path.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(";", "\\;")
        .replace("#", "\\#")
    )


