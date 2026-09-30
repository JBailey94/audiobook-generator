"""
Text normalization and sentence splitting for TTS processing.
Ported from ebook2audiobook's text processing pipeline.
"""

import re
import unicodedata

# SML (Speech Markup Language) tags
SML_TAG_PATTERN = re.compile(r"\[/?\s*(?:break|pause|voice)(?:\s*:\s*.*?)?\s*\]", re.DOTALL)
SML_ESCAPE_TAG = 0xE000

# Punctuation sets
PUNCTUATION_SPLIT_HARD = {".", "!", "?", "…", "‽", "—", "。", "！", "？", "।", "॥", "።", "፧", "།", "។", "៕"}
PUNCTUATION_SPLIT_SOFT = {",", ":", ";", "，", "、", "·", "ฯ", "፡", "፣", "፤", "፥", "፦", "״", "༎", "໌", "ໍ"}

# Punctuation that causes TTS hallucinations
PUNCTUATION_SWITCH = {
    "«": '"', "»": '"', "“": '"', "”": '"', "‘": "'", "’": "'",
    "„": '"', "–": ".", "_": " ", "‗": " ", "¯": " ", "ˍ": " ",
    "﹍": " ", "﹎": " ", "﹏": " ", "＿": " ", "...": "…",
    "‽": "?", "⁉": "?!", "‼": "!!", "⁈": "?!", "⁇": "??",
    "﹖": "?", "﹗": "!", "۔": ".", "॥": ".", "。": ".",
    "።": ".", "།": ".", "፡": ":", "፤": ";", "।": ".",
    "•": ".", "›": "", "#": "-", "†": "-", "¶": "-",
    "—": ".", "(": ",", ")": ",",
}

# Characters to remove
CHARS_TO_REMOVE = {"\\", "|", "©", "®", "™", "*", "`", "\u00A0", "\xa0"}

# Language-specific max chars (from ebook2audiobook's conf_lang.py)
LANGUAGE_MAX_CHARS = {
    "eng": 250, "fra": 273, "deu": 253, "spa": 239, "por": 203,
    "ita": 213, "rus": 182, "pol": 224, "hun": 224, "tur": 226,
    "jpn": 71, "kor": 95, "zho": 82, "ara": 166, "hin": 142,
    "ben": 142, "fas": 182, "ind": 182, "jav": 182, "tam": 142,
    "tel": 142, "yor": 142, "tha": 142, "vie": 182, "nld": 253,
    "swe": 253, "dan": 253, "nob": 253, "fin": 253, "ces": 224,
    "ell": 239, "heb": 182, "ukr": 182, "ron": 213, "bul": 182,
}


def escape_sml(text: str) -> tuple[str, list[str]]:
    """Escape SML tags to protect them during text processing."""
    sml_blocks: list[str] = []

    def _replace(m: re.Match) -> str:
        sml_blocks.append(m.group(0))
        return chr(SML_ESCAPE_TAG + len(sml_blocks) - 1)

    return SML_TAG_PATTERN.sub(_replace, text), sml_blocks


def restore_sml(text: str, sml_blocks: list[str]) -> str:
    """Restore escaped SML tags after text processing."""
    for i, block in enumerate(sml_blocks):
        text = text.replace(chr(SML_ESCAPE_TAG + i), block)
    return text


def normalize_text(text: str, lang: str = "eng") -> str:
    """Normalize text for TTS processing."""
    # Remove emojis
    text = remove_emojis(text)

    # Replace punctuation that causes hallucinations
    for old, new in PUNCTUATION_SWITCH.items():
        text = text.replace(old, new)

    # Remove unwanted characters
    for ch in CHARS_TO_REMOVE:
        text = text.replace(ch, " ")

    # Replace multiple newlines with [pause]
    text = re.sub(r"(?:\r\n|\r|\n){2,}", " [pause] ", text)
    # Replace single newlines with spaces
    text = re.sub(r"\r\n|\r|\n", " ", text)

    # Normalize quotes
    text = re.sub(r'\s*"\s*', '"', text)
    text = re.sub(r'(?<=[\w])"(?=[\w]|$)', ", ", text)
    text = text.replace('"', "")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Reduce consecutive hard punctuation
    hard_pattern = "|".join(re.escape(c) for c in PUNCTUATION_SPLIT_HARD)
    text = re.sub(rf"(\s*({hard_pattern})\s*)+", r"\2 ", text).strip()

    # Reduce consecutive soft punctuation
    soft_pattern = "|".join(re.escape(c) for c in PUNCTUATION_SPLIT_SOFT)
    text = re.sub(rf"(\s*({soft_pattern})\s*)+", r"\2 ", text).strip()

    # Add space between letters and numbers
    text = re.sub(r"(?<=\w)(?=\d)|(?<=\d)(?=\w)", " ", text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    return text.strip()


def remove_emojis(text: str) -> str:
    """Remove emoji characters from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def _clean_len(s: str) -> int:
    """Length of string excluding escaped SML tags."""
    return len("".join(c for c in s if ord(c) < SML_ESCAPE_TAG))


def _force_split_segment(segment: str, max_chars: int) -> list[str]:
    """Force-split a segment at punctuation or spaces."""
    results = []
    rest = segment
    hard_set = PUNCTUATION_SPLIT_HARD
    soft_set = PUNCTUATION_SPLIT_SOFT

    while rest:
        if _clean_len(rest) <= max_chars:
            results.append(rest.strip())
            break

        cut = rest[: max_chars + 1]
        best_idx = -1

        # Try hard punctuation
        for i in range(len(cut) - 1, -1, -1):
            if cut[i] in hard_set:
                best_idx = i + 1
                break

        # Try soft punctuation
        if best_idx == -1:
            for i in range(len(cut) - 1, -1, -1):
                if cut[i] in soft_set:
                    best_idx = i + 1
                    break

        # Fall back to last space
        if best_idx == -1:
            idx = cut.rfind(" ")
            if idx > 0:
                best_idx = idx

        # Last resort: hard cut
        if best_idx <= 0:
            best_idx = max_chars

        # Don't cut inside escaped SML
        while best_idx < len(rest) and ord(rest[best_idx]) >= SML_ESCAPE_TAG:
            best_idx += 1

        left = rest[:best_idx].strip()
        right = rest[best_idx:].strip()

        if not left or right == rest:
            results.append(rest.strip())
            break

        results.append(left)
        rest = right

    return results


def split_into_sentences(text: str, lang: str = "eng", max_chars: int | None = None) -> list[str]:
    """Split text into TTS-ready sentence chunks."""
    if max_chars is None:
        max_chars = LANGUAGE_MAX_CHARS.get(lang, 250) // 2

    # Escape SML tags
    text, sml_blocks = escape_sml(text)

    # Tokenize into text and SML segments
    segments = []
    idx = 0
    tlen = len(text)
    current_text = []

    while idx < tlen:
        c = text[idx]
        if ord(c) >= SML_ESCAPE_TAG:
            if current_text:
                segments.append(("text", "".join(current_text)))
                current_text = []
            start = idx
            while idx < tlen and ord(text[idx]) >= SML_ESCAPE_TAG:
                idx += 1
            segments.append(("sml", text[start:idx]))
        else:
            current_text.append(c)
            idx += 1

    if current_text:
        segments.append(("text", "".join(current_text)))

    # Build sentences with SML-aware splitting
    final_list = []
    buffer = []
    current_len = 0

    for seg_type, seg_content in segments:
        if seg_type == "sml":
            buffer.append(seg_content)
            continue

        seg_clean_len = _clean_len(seg_content)
        potential_len = current_len + seg_clean_len

        if potential_len <= max_chars:
            buffer.append(seg_content)
            current_len = potential_len
            continue

        # Doesn't fit - try to cut at rightmost SML in buffer
        combined = "".join(buffer)
        cut_idx = -1
        j = len(combined) - 1
        while j >= 0:
            if ord(combined[j]) >= SML_ESCAPE_TAG:
                cut_idx = j + 1
                break
            j -= 1

        cut_done = False
        if 0 < cut_idx <= len(combined):
            part1 = combined[:cut_idx]
            part2 = combined[cut_idx:]
            if seg_clean_len <= max_chars and _clean_len(part2) + seg_clean_len <= max_chars:
                if _clean_len(part1) > 0:
                    final_list.append(part1.strip())
                buffer = [part2, seg_content] if part2 else [seg_content]
                current_len = _clean_len(part2) + seg_clean_len
                cut_done = True

        if not cut_done:
            # Flush buffer, force-split if needed
            if _clean_len(combined) > 0:
                if current_len <= max_chars:
                    final_list.append(combined.strip())
                else:
                    final_list.extend(_force_split_segment(combined.strip(), max_chars))

                if seg_clean_len > max_chars:
                    parts = _force_split_segment(seg_content, max_chars)
                    for p in parts[:-1]:
                        if p:
                            final_list.append(p)
                    buffer = [parts[-1]]
                    current_len = _clean_len(parts[-1])
                else:
                    buffer = [seg_content]
                    current_len = seg_clean_len
            else:
                # Buffer is pure SML - carry forward
                pending = combined
                if seg_clean_len > max_chars:
                    parts = _force_split_segment(seg_content, max_chars)
                    if parts and pending.strip():
                        parts[0] = pending + parts[0]
                    for p in parts[:-1]:
                        if p:
                            final_list.append(p)
                    buffer = [parts[-1]]
                    current_len = _clean_len(parts[-1])
                else:
                    buffer = [pending, seg_content] if pending.strip() else [seg_content]
                    current_len = seg_clean_len

    # Final flush
    if buffer:
        combined = "".join(buffer).strip()
        if combined:
            if _clean_len(combined) > 0:
                if _clean_len(combined) <= max_chars:
                    final_list.append(combined)
                else:
                    final_list.extend(_force_split_segment(combined, max_chars))
            elif final_list:
                final_list[-1] = final_list[-1].rstrip() + " " + combined

    # Strip leading noise and empty entries
    final_list = [s.lstrip(" \t\n\r.,;:!?") for s in final_list if s.strip()]
    final_list = [s for s in final_list if s]

    # Merge orphan-short sentences
    merge_threshold = max_chars // 2
    merge_ceiling = max_chars + max_chars // 2
    merged_list = []
    i = 0
    n = len(final_list)

    while i < n:
        cur = final_list[i].strip()
        if not cur:
            i += 1
            continue

        cur_len = _clean_len(cur)
        if cur_len <= merge_threshold:
            # Try to attach to previous
            if merged_list:
                prev = merged_list[-1]
                if _clean_len(prev) + 1 + cur_len <= merge_ceiling:
                    merged_list[-1] = prev.rstrip() + " " + cur.lstrip()
                    i += 1
                    continue
            # Try to glue to next
            if i + 1 < n:
                nxt = final_list[i + 1].strip()
                if cur_len + 1 + _clean_len(nxt) <= merge_ceiling:
                    merged_list.append(cur.rstrip() + " " + nxt.lstrip())
                    i += 2
                    continue

        merged_list.append(cur)
        i += 1

    final_list = merged_list

    # Restore SML tags
    if final_list:
        final_list = [restore_sml(s, sml_blocks) for s in final_list]

    return final_list


def process_file(file_path: str, lang: str = "eng", max_chars: int | None = None) -> list[str]:
    """Extract text from a file and split into TTS-ready chunks."""
    from file_processing import extract_text

    text = extract_text(file_path)
    text = normalize_text(text, lang=lang)
    return split_into_sentences(text, lang=lang, max_chars=max_chars)


def process_sections(
    file_path: str,
    lang: str = "eng",
    max_chars: int | None = None,
    target_chars: int | None = None,
    min_chars: int | None = None,
) -> list[tuple[str, list[str]]]:
    """
    Extract a book as ``(section title, chunks)`` pairs, one entry per chapter.

    Sections that contain no speakable text after normalization are dropped.
    """
    from file_processing import (
        DEFAULT_SECTION_CHARS,
        MIN_SECTION_CHARS,
        extract_sections,
    )

    sections = extract_sections(
        file_path,
        target_chars=target_chars or DEFAULT_SECTION_CHARS,
        min_chars=MIN_SECTION_CHARS if min_chars is None else min_chars,
    )

    processed: list[tuple[str, list[str]]] = []
    for section in sections:
        normalized = normalize_text(section.text, lang=lang)
        if not normalized:
            continue
        chunks = split_into_sentences(normalized, lang=lang, max_chars=max_chars)
        if chunks:
            processed.append((section.title, chunks))

    if not processed:
        raise ValueError(f"No speakable text found in {file_path}")
    return processed

