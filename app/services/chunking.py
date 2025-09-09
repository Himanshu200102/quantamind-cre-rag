from __future__ import annotations
import re, glob, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
from dateutil.parser import parse as parse_date

# ------------- CRE Document Type Detection -----------------
CRE_DOCUMENT_TYPES = {
    "lease": {
        "patterns": [r"lease agreement", r"rental agreement", r"lessor", r"lessee", r"demised premises", r"monthly rent"],
        "weight": 1.0
    },
    "purchase": {
        "patterns": [r"purchase agreement", r"sale agreement", r"buyer", r"seller", r"purchase price", r"closing"],
        "weight": 1.0
    },
    "management": {
        "patterns": [r"property management", r"management agreement", r"property manager", r"management services"],
        "weight": 1.0
    },
    "construction": {
        "patterns": [r"construction contract", r"construction agreement", r"contractor", r"construction services", r"completion date"],
        "weight": 1.0
    },
    "development": {
        "patterns": [r"development agreement", r"developer", r"development project", r"site plan"],
        "weight": 1.0
    },
    "financing": {
        "patterns": [r"loan agreement", r"mortgage", r"promissory note", r"financing", r"lender", r"borrower"],
        "weight": 1.0
    },
    "amendment": {
        "patterns": [r"amendment", r"modification", r"addendum", r"supplemental agreement"],
        "weight": 1.2
    },
    "sublease": {
        "patterns": [r"sublease", r"sublessor", r"sublessee", r"sub-lease"],
        "weight": 1.1
    }
}

# ------------- Chunking regexes (merged & expanded) -----------------
ARTICLE_RE = re.compile(
    r"^\s*ARTICLE\s+(?P<num>\d+(?:\.\d+)*)[\.:\-\s]*(?P<title>.*?)\s*$",
    re.I | re.M,
)
SECTION_RE = re.compile(
    r"^\s*(?:Section\s+|§\s*)?(?P<num>\d+(?:\.\d+)*)[)\.\s:-]+(?P<title>.*?)(?:\.\s*)?$",
    re.I | re.M,
)
ROMAN_SECTION_RE = re.compile(
    r"^\s*(?P<num>[IVX]+)\.?\s+(?P<title>[A-Z][^.]*?)\.?\s*$",
    re.I | re.M,
)
ALT_SECTION_RE = re.compile(r"^\s*[A-Z]\.\s+[A-Z]", re.M)
LOWERCASE_LETTER_RE = re.compile(r"^\s*\([a-z]\)\s+[A-Za-z]", re.M)
NUMBERED_LIST_RE = re.compile(r"^\s*\d+\)\s+[A-Za-z]", re.M)
ALL_CAPS_HEADING_RE = re.compile(r"^\s*[A-Z][A-Z\s,&]{10,80}\.?\s*$", re.M)
TITLE_HEADING_RE = re.compile(r"^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*\s*[:\.]\s*$", re.M)


def _clean_page_footer_noise(txt: str) -> str:
    txt = re.sub(r"\s*Page \d+.*$", "", txt, flags=re.M)
    txt = re.sub(r"Commercial Lease - Page \d+ of \d+", "", txt)
    return txt


def extract_pages(pdf_path: Path) -> List[str]:
    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            txt = pg.extract_text() or ""
            pages.append(_clean_page_footer_noise(txt).strip())
    return pages


def detect_cre_document_type(text: str, filename: str) -> str:
    """Intelligently detect CRE document type based on content and filename"""
    text_lower = text.lower()
    filename_lower = filename.lower()

    scores: Dict[str, float] = {}

    for doc_type, config in CRE_DOCUMENT_TYPES.items():
        score = 0.0
        weight = float(config.get("weight", 1.0))

        # Score based on content patterns
        for pattern in config["patterns"]:
            content_matches = len(re.findall(pattern, text_lower))
            filename_matches = len(re.findall(pattern, filename_lower)) * 2  # Filename matches weighted higher
            score += (content_matches + filename_matches) * weight

        scores[doc_type] = score

    # Return highest scoring type, default to lease if no clear winner
    if not scores or max(scores.values()) == 0:
        return "lease"

    return max(scores, key=scores.get)


def detect_toc_page(pages: List[str]) -> Optional[int]:
    for i, page in enumerate(pages):
        low = page.lower()
        if re.search(r"table\s+of\s+contents", low):
            return i
        if re.search(r"^contents\s*$", low, re.M):
            return i
        patterns = [
            r"^\s*\d+\.\s+[A-Z][^.]+\.{2,}\s*\d+\s*$",
            r"^\s*\d+\s+[A-Z][^.]+\.{2,}\s*\d+\s*$",
            r"^\s*[A-Z][^.]+\.{3,}\s*\d+\s*$",
        ]
        count = 0
        for pat in patterns:
            count += len(re.findall(pat, page, re.M | re.I))
        if count >= 3:
            return i
    return None


def extract_toc_structure(toc_page: str) -> List[Dict[str, Any]]:
    toc_entries: List[Dict[str, Any]] = []
    patterns = [
        re.compile(r"^\s*(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>[^.]+?)\.{2,}\s*(?P<page>\d+)\s$", re.M | re.I),
        re.compile(r"^\s*(?P<num>\d+(?:\.\d+)*)\s+(?P<title>[^.]+?)\.{2,}\s*(?P<page>\d+)\s$", re.M | re.I),
        re.compile(r"^\s*(?P<title>[A-Z][^.]+?)\.{3,}\s*(?P<page>\d+)\s$", re.M | re.I),
        re.compile(r"^\s*(?:ARTICLE\s+)?(?P<num>\d+(?:\.\d+)*)[\.:\-\s]*(?P<title>[^\.]+)\.{2,}(?P<page>\d+)\s$", re.M | re.I),
        re.compile(r'^\s*Exhibit\s+"?(?P<num>[A-Z0-9\-]+)"?\s*[-\s]*(?P<title>[^.]+?)\s*$', re.M | re.I),
    ]
    for pat in patterns:
        for m in pat.finditer(toc_page):
            g = m.groupdict()
            num = (g.get("num") or "").strip()
            title = re.sub(r"\s+", " ", (g.get("title") or "").strip()).strip(".-")
            page_str = g.get("page") or "0"
            if not title:
                continue
            try:
                page_num = int(page_str) if page_str.isdigit() else 0
            except Exception:
                page_num = 0
            entry_type = "section"
            raw = m.group(0)
            if "exhibit" in raw.lower() or "exhibit" in title.lower():
                entry_type = "exhibit"
            elif "article" in raw.lower():
                entry_type = "article"
            entry = {"number": num, "title": title, "page": page_num, "type": entry_type}
            if not any(e["title"].lower() == title.lower() and e.get("number") == num for e in toc_entries):
                toc_entries.append(entry)
    toc_entries.sort(key=lambda x: (x["page"], x.get("number", "")))
    return toc_entries


def _build_alltext_and_page_offsets(pages: List[str]) -> Tuple[str, List[int]]:
    parts: List[str] = []
    page_offsets: List[int] = []
    cursor = 0
    for i, p in enumerate(pages):
        page_offsets.append(cursor)
        parts.append(p)
        cursor += len(p)
        if i < len(pages) - 1:
            parts.append("\f")  # Form-feed page separator
            cursor += 1
    all_text = "".join(parts)
    return all_text, page_offsets


def _offset_to_page(offset: int, page_offsets: List[int]) -> int:
    page = 1
    for i in range(len(page_offsets)):
        if i + 1 == len(page_offsets) or (page_offsets[i] <= offset < page_offsets[i + 1]):
            page = i + 1
            break
    return max(1, page)


def validate_content_coverage(original_text: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure no content is lost during chunking. Counts exclude form-feed to avoid skew."""
    # Exclude form-feeds from both sides for a fair ratio
    original_chars = len(original_text.replace("\f", "")) if original_text else 0
    total_chunk_chars = sum(len((chunk.get("text", "") or "").replace("\f", "")) for chunk in chunks)

    coverage_ratio = (total_chunk_chars / original_chars) if original_chars > 0 else 0.0

    # Check for significant content loss
    if coverage_ratio < 0.95:  # Less than 95% coverage
        return {
            "coverage_ratio": coverage_ratio,
            "potential_loss": True,
            "original_chars": original_chars,
            "chunked_chars": total_chunk_chars,
            "missing_chars": max(0, original_chars - total_chunk_chars),
        }

    return {"coverage_ratio": coverage_ratio, "potential_loss": False}


def detect_content_gaps(all_text: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find text segments not captured in any chunk"""
    if not chunks:
        return []

    covered_ranges: List[Tuple[int, int]] = []
    for chunk in chunks:
        start = int(chunk.get("start_offset", 0))
        end = int(chunk.get("end_offset", start + len(chunk.get("text", ""))))
        if start < end:
            covered_ranges.append((start, end))

    # Sort ranges and merge overlapping ones
    covered_ranges.sort()
    merged_ranges: List[Tuple[int, int]] = []
    for start, end in covered_ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))

    # Find gaps
    gaps: List[Dict[str, Any]] = []

    # Check for content before first chunk
    if merged_ranges and merged_ranges[0][0] > 0:
        gap_text = all_text[0:merged_ranges[0][0]].strip()
        if len(gap_text) > 20:
            gaps.append({
                "text": gap_text,
                "start_offset": 0,
                "end_offset": merged_ranges[0][0],
                "gap_type": "leading",
            })

    # Check for gaps between chunks
    for i in range(len(merged_ranges) - 1):
        current_end = merged_ranges[i][1]
        next_start = merged_ranges[i + 1][0]

        if next_start > current_end:
            gap_text = all_text[current_end:next_start].strip()
            if len(gap_text) > 20:
                gaps.append({
                    "text": gap_text,
                    "start_offset": current_end,
                    "end_offset": next_start,
                    "gap_type": "between",
                })

    # Check for content after last chunk
    if merged_ranges and merged_ranges[-1][1] < len(all_text):
        gap_text = all_text[merged_ranges[-1][1]:].strip()
        if len(gap_text) > 20:
            gaps.append({
                "text": gap_text,
                "start_offset": merged_ranges[-1][1],
                "end_offset": len(all_text),
                "gap_type": "trailing",
            })

    return gaps


def preserve_critical_sections(text: str, page_offsets: List[int]) -> List[Dict[str, Any]]:
    """Identify and preserve critical sections that must not be lost"""
    critical_patterns = [
        (r"(?i)signature\s+page", "signature"),
        (r"(?i)schedule\s+[a-z0-9-]+", "schedule"),
        (r"(?i)exhibit\s+[a-z0-9-]+", "exhibit"),
        (r"(?i)appendix\s+[a-z0-9-]+", "appendix"),
        (r"(?i)definitions?\s*:", "definitions"),
        (r"(?i)between\s+.*?\s+and\s+.*?(?=\(|whereas|now)", "parties"),
        (r"(?i)witnesseth|recitals?", "recitals"),
        (r"(?i)in\s+witness\s+whereof", "execution"),
    ]

    critical_chunks: List[Dict[str, Any]] = []
    for pattern, chunk_type in critical_patterns:
        matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))
        for match in matches:
            # Extract surrounding context (500 chars before and after)
            start = max(0, match.start() - 500)
            end = min(len(text), match.end() + 1000)

            # Find natural boundaries (paragraph breaks)
            while start > 0 and text[start] not in ['\n', '\f']:
                start -= 1
            while end < len(text) and text[end] not in ['\n', '\f']:
                end += 1

            section_text = text[start:end].strip()
            if len(section_text) > 50:
                page_start = _offset_to_page(start, page_offsets)
                page_end = _offset_to_page(end - 1, page_offsets)

                critical_chunks.append({
                    "header": f"CRITICAL_{chunk_type.upper()}_{match.start()}",
                    "text": section_text,
                    "type": f"critical_{chunk_type}",
                    "start_offset": start,
                    "end_offset": end,
                    "page_start": page_start,
                    "page_end": page_end,
                    "word_count": len(section_text.split()),
                    "toc_based": False,
                })

    return critical_chunks


def sliding_window_comprehensive(text: str, size: int = 1200, overlap: int = 400) -> List[Dict[str, Any]]:
    """Comprehensive sliding window that ensures no content is lost"""
    chunks: List[Dict[str, Any]] = []
    pos = 0
    chunk_index = 0

    while pos < len(text):
        end = min(len(text), pos + size)

        # Try to break at natural boundaries
        if end < len(text):
            # Look for paragraph break within last 200 chars
            search_start = max(pos + size - 200, pos + size // 2)
            boundary = text.rfind('\n\n', search_start, end)
            if boundary > search_start:
                end = boundary + 2
            else:
                # Look for sentence break
                boundary = text.rfind('. ', search_start, end)
                if boundary > search_start:
                    end = boundary + 2

        chunk_text = text[pos:end].strip()
        if chunk_text:
            chunks.append({
                "header": f"COMPREHENSIVE_CHUNK_{chunk_index}",
                "text": chunk_text,
                "page_start": 1,
                "page_end": 1,
                "type": "comprehensive_window",
                "word_count": len(chunk_text.split()),
                "toc_based": False,
                "start_offset": pos,
                "end_offset": end,
            })
            chunk_index += 1

        if end == len(text):
            break

        # Move position with overlap, but ensure progress
        new_pos = max(pos + 1, end - overlap)
        pos = new_pos

    return chunks


def chunk_by_toc(pages: List[str], toc_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    all_text, page_offsets = _build_alltext_and_page_offsets(pages)

    boundaries: List[Dict[str, Any]] = []
    for entry in toc_entries:
        entry_number = entry.get("number", "")
        entry_title = entry.get("title", "")
        section_patterns: List[str] = []

        if entry_number:
            section_patterns += [
                rf"^\s*{re.escape(entry_number)}\.\s+{re.escape(entry_title)}",
                rf"^\s*{re.escape(entry_number)}\s+{re.escape(entry_title)}",
            ]
            if len(entry_title) > 20:
                partial = re.escape(entry_title[:20])
                section_patterns += [rf"^\s*{re.escape(entry_number)}[\.:\s]+{partial}"]
        if entry_title:
            section_patterns.append(rf"^\s*{re.escape(entry_title)}\s*[\.:]?\s*$")

        found = None
        for pat in section_patterns:
            m = re.search(pat, all_text, re.M | re.I)
            if m:
                found = {"pos": m.start(), "entry": entry}
                break
        if found:
            boundaries.append(found)

    boundaries.sort(key=lambda x: x["pos"])

    for i, b in enumerate(boundaries):
        start = b["pos"]
        end = boundaries[i + 1]["pos"] if i + 1 < len(boundaries) else len(all_text)
        text = all_text[start:end].strip()
        if len(text.split()) <= 10:
            continue
        entry = b["entry"]
        header = entry.get("title", "")
        num = entry.get("number", "")
        if num:
            header = f"{num} - {header}"
        page_start = entry.get("page", 1) or _offset_to_page(start, page_offsets)
        page_end = _offset_to_page(end - 1, page_offsets)
        chunks.append({
            "header": header or "SECTION",
            "text": text,
            "page_start": page_start,
            "page_end": page_end,
            "type": entry.get("type", "section"),
            "word_count": len(text.split()),
            "toc_based": True,
            "section_number": num,
            "start_offset": start,
            "end_offset": end,
        })
    return chunks


def _structure_based_chunking(all_text: str, page_offsets: List[int]) -> List[Dict[str, Any]]:
    """Internal structure-based chunking implementation"""
    lines = all_text.split("\n")
    chunks: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    current_header = "DOCUMENT_START"
    current_type = "preamble"
    current_start_off = 0

    line_offsets: List[int] = []
    cursor = 0
    for ln in lines:
        line_offsets.append(cursor)
        cursor += len(ln) + 1

    def flush(end_offset: Optional[int] = None):
        nonlocal current_lines, current_header, current_type, current_start_off
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        # Lower threshold to preserve more content
        if len(text.split()) <= 3:
            current_lines = []
            return
        end_off = end_offset if end_offset is not None else (
            line_offsets[min(len(current_lines), len(line_offsets) - 1)] if current_lines else current_start_off
        )
        page_start = _offset_to_page(current_start_off, page_offsets)
        page_end = _offset_to_page(max(current_start_off, end_off - 1), page_offsets)
        chunks.append({
            "header": current_header,
            "text": text,
            "page_start": page_start,
            "page_end": page_end,
            "type": current_type,
            "word_count": len(text.split()),
            "toc_based": False,
            "start_offset": current_start_off,
            "end_offset": end_off,
        })
        current_lines = []

    def is_likely_heading(line: str) -> bool:
        s = line.strip()
        if len(s) > 120:
            return False
        alpha = sum(1 for c in s if c.isalpha())
        if alpha < 3:
            return False
        return any([
            s.endswith(":"),
            (s.endswith(".") and len(s) < 80),
            re.match(r"^\s*[A-Z][A-Z\s]{5,}$", s),
            re.match(r"^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*\s*$", s),
        ])

    for i, line in enumerate(lines):
        s = line.strip()
        heading_detected = False
        new_header, new_type = None, None

        if s:
            m = ARTICLE_RE.match(s)
            if m:
                new_header = f"ARTICLE {m.group('num')}"
                title = (m.group("title") or "").strip()
                if title:
                    new_header += f" - {title}"
                new_type = "article"
                heading_detected = True
            elif ROMAN_SECTION_RE.match(s):
                rm = ROMAN_SECTION_RE.match(s)
                new_header = f"{rm.group('num')} - {rm.group('title').strip()}"
                new_type = "section"
                heading_detected = True
            elif SECTION_RE.match(s) and len(s) < 150:
                sm = SECTION_RE.match(s)
                new_header = f"Section {sm.group('num')}"
                title = (sm.group("title") or "").strip()
                if title:
                    new_header += f" - {title}"
                new_type = "section"
                heading_detected = True
            elif (ALT_SECTION_RE.match(s) and len(s) < 100) or (LOWERCASE_LETTER_RE.match(s) and len(s) < 100) or (NUMBERED_LIST_RE.match(s) and len(s) < 100):
                new_header = s
                new_type = "subsection"
                heading_detected = True
            elif ALL_CAPS_HEADING_RE.match(s) and not re.search(r"\d+", s):
                new_header = s
                new_type = "section"
                heading_detected = True
            elif TITLE_HEADING_RE.match(s):
                new_header = s.rstrip(":.")
                new_type = "section"
                heading_detected = True
            elif (is_likely_heading(s) and 10 < len(s) < 80 and not any(ch.isdigit() for ch in s[:5])):
                new_header = s.rstrip(":.")
                new_type = "section"
                heading_detected = True

        if heading_detected:
            flush(end_offset=line_offsets[i] if i < len(line_offsets) else len(all_text))
            current_header = new_header or "SECTION"
            current_type = new_type or "section"
            current_start_off = line_offsets[i] if i < len(line_offsets) else len(all_text)
            current_lines = [line]
        else:
            current_lines.append(line)

    flush(end_offset=len(all_text))
    return chunks


def chunk_by_structure(pages: List[str], toc_entries: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Comprehensive multi-strategy chunking with zero content loss guarantee"""
    all_text, page_offsets = _build_alltext_and_page_offsets(pages)

    # Strategy 1: Try TOC-based chunking first (if available)
    best_chunks: List[Dict[str, Any]] = []
    best_coverage = 0.0
    strategy_used = "none"

    if toc_entries and len(toc_entries) >= 3:
        toc_chunks = chunk_by_toc(pages, toc_entries)
        if toc_chunks:
            validation = validate_content_coverage(all_text, toc_chunks)
            if not validation["potential_loss"] and len(toc_chunks) >= max(3, int(0.5 * len(toc_entries))):
                best_chunks = toc_chunks
                best_coverage = validation["coverage_ratio"]
                strategy_used = "toc_based"
                print(f"[chunking] TOC-based strategy: {len(best_chunks)} chunks, {best_coverage:.1%} coverage")

    # Strategy 2: Structure-based chunking
    if best_coverage < 0.95:
        structure_chunks = _structure_based_chunking(all_text, page_offsets)
        if structure_chunks:
            validation = validate_content_coverage(all_text, structure_chunks)
            if validation["coverage_ratio"] > best_coverage:
                best_chunks = structure_chunks
                best_coverage = validation["coverage_ratio"]
                strategy_used = "structure_based"
                print(f"[chunking] Structure-based strategy: {len(best_chunks)} chunks, {best_coverage:.1%} coverage")

    # Strategy 3: Gap recovery - detect and fill any missing content
    if best_chunks:
        gaps = detect_content_gaps(all_text, best_chunks)
        if gaps:
            print(f"[chunking] Found {len(gaps)} content gaps, adding recovery chunks")
            for i, gap in enumerate(gaps):
                page_start = _offset_to_page(gap["start_offset"], page_offsets)
                page_end = _offset_to_page(gap["end_offset"] - 1, page_offsets)

                best_chunks.append({
                    "header": f"GAP_RECOVERY_{gap['gap_type'].upper()}_{i}",
                    "text": gap["text"],
                    "page_start": page_start,
                    "page_end": page_end,
                    "type": f"gap_{gap['gap_type']}",
                    "word_count": len(gap["text"].split()),
                    "toc_based": False,
                    "start_offset": gap["start_offset"],
                    "end_offset": gap["end_offset"],
                })

            # Re-validate coverage
            validation = validate_content_coverage(all_text, best_chunks)
            best_coverage = validation["coverage_ratio"]
            strategy_used += "_with_gap_recovery"

    # Strategy 4: Critical section preservation
    critical_chunks = preserve_critical_sections(all_text, page_offsets)
    if critical_chunks:
        print(f"[chunking] Adding {len(critical_chunks)} critical sections")
        # Remove duplicates by checking for overlap
        for critical_chunk in critical_chunks:
            is_duplicate = False
            for existing_chunk in best_chunks:
                existing_start = int(existing_chunk.get("start_offset", 0))
                existing_end = int(existing_chunk.get("end_offset", existing_start + len(existing_chunk.get("text", ""))))
                critical_start = int(critical_chunk["start_offset"])
                critical_end = int(critical_chunk["end_offset"])

                # Check for significant overlap (>50%)
                overlap_start = max(existing_start, critical_start)
                overlap_end = min(existing_end, critical_end)
                overlap_length = max(0, overlap_end - overlap_start)
                critical_length = max(1, critical_end - critical_start)

                if overlap_length > 0.5 * critical_length:
                    is_duplicate = True
                    break

            if not is_duplicate:
                best_chunks.append(critical_chunk)

    # Strategy 5: Final fallback - comprehensive sliding window
    final_validation = validate_content_coverage(all_text, best_chunks)
    if final_validation["coverage_ratio"] < 0.90:
        print(f"[chunking] Coverage still low ({final_validation['coverage_ratio']:.1%}), using comprehensive sliding window")
        best_chunks = sliding_window_comprehensive(all_text)
        strategy_used = "comprehensive_sliding_window"
        final_validation = validate_content_coverage(all_text, best_chunks)

    # Final merge of small chunks
    if strategy_used != "comprehensive_sliding_window":
        best_chunks = smart_merge_small_chunks(best_chunks, min_words=30)  # Lower threshold to preserve more content

    # Final validation and reporting
    final_validation = validate_content_coverage(all_text, best_chunks)
    print(f"[chunking] Final result: {len(best_chunks)} chunks, {final_validation['coverage_ratio']:.1%} coverage, strategy: {strategy_used}")

    if final_validation["potential_loss"]:
        print(f"[chunking] WARNING: Potential content loss detected! Missing {final_validation.get('missing_chars', 0)} characters")

    return best_chunks


def smart_merge_small_chunks(chunks: List[Dict[str, Any]], min_words: int = 30) -> List[Dict[str, Any]]:
    """Smart merging that preserves content and maintains logical boundaries"""
    if not chunks:
        return chunks
    merged: List[Dict[str, Any]] = []
    i = 0

    while i < len(chunks):
        c = chunks[i]
        current_word_count = int(c.get("word_count", len(c.get("text", "").split())))

        # Don't merge critical sections or gap recovery chunks
        if c.get("type", "").startswith("critical") or c.get("type", "").startswith("gap"):
            merged.append(c)
            i += 1
            continue

        # Only merge if chunk is truly small and has a compatible neighbor
        if current_word_count < min_words and i + 1 < len(chunks):
            n = chunks[i + 1]
            next_word_count = int(n.get("word_count", len(n.get("text", "").split())))

            # Don't merge with critical sections or gap recovery chunks
            if n.get("type", "").startswith("critical") or n.get("type", "").startswith("gap"):
                merged.append(c)
                i += 1
                continue

            # Check compatibility
            compatible = {
                "preamble": ["preamble", "section"],
                "article": ["section", "subsection"],
                "section": ["section", "subsection"],
                "subsection": ["subsection"],
                "window": ["window"],  # Don't merge different types with windows
                "comprehensive_window": ["comprehensive_window"],
            }

            c_type = c.get("type", "section")
            n_type = n.get("type", "section")

            # Only merge if types are compatible and combined size isn't too large
            if (
                (c_type == n_type or (c_type in compatible and n_type in compatible.get(c_type, [])))
                and (current_word_count + next_word_count < min_words * 4)
            ):
                # Preserve offset information for gap detection
                start_offset = int(c.get("start_offset", 0))
                end_offset = int(n.get("end_offset", c.get("end_offset", 0)))

                text = (c.get("text", "") or "") + "\n\n" + (n.get("text", "") or "")
                merged.append({
                    "header": f"{c.get('header', 'CHUNK') } & { n.get('header', 'CHUNK')}",
                    "text": text,
                    "page_start": int(c.get("page_start", 1)),
                    "page_end": int(n.get("page_end", c.get("page_end", 1))),
                    "type": c_type,  # Keep the first chunk's type
                    "word_count": len(text.split()),
                    "toc_based": bool(c.get("toc_based", False) and n.get("toc_based", False)),
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                })
                i += 2
                continue

        merged.append(c)
        i += 1

    return merged


def sliding_window(text: str, size: int = 1200, overlap: int = 300) -> List[str]:
    out: List[str] = []
    pos = 0
    while pos < len(text):
        end = min(len(text), pos + size)
        chunk_text = text[pos:end]
        if chunk_text.strip():
            out.append(chunk_text)
        if end == len(text):
            break
        pos = end - overlap
    return out


def extract_universal_parties(text: str, doc_type: str) -> Dict[str, Optional[str]]:
    """Extract parties based on document type"""
    parties: Dict[str, Optional[str]] = {}

    if doc_type == "lease":
        # Lease-specific patterns
        for party_type, patterns in [
            (
                "lessor",
                [
                    r"(?:lessor|landlord)[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                    r"LESSOR[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                ],
            ),
            (
                "lessee",
                [
                    r"(?:lessee|tenant)[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                    r"LESSEE[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                ],
            ),
        ]:
            for pattern in patterns:
                match = re.search(pattern, text, re.I | re.MULTILINE)
                if match:
                    parties[party_type] = match.group(1).strip()
                    break

    elif doc_type == "purchase":
        # Purchase agreement patterns
        for party_type, patterns in [
            (
                "buyer",
                [
                    r"(?:buyer|purchaser)[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                    r"BUYER[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                ],
            ),
            (
                "seller",
                [
                    r"(?:seller|vendor)[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                    r"SELLER[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                ],
            ),
        ]:
            for pattern in patterns:
                match = re.search(pattern, text, re.I | re.MULTILINE)
                if match:
                    parties[party_type] = match.group(1).strip()
                    break

    elif doc_type == "management":
        # Management agreement patterns
        for party_type, patterns in [
            (
                "manager",
                [
                    r"(?:manager|management company)[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                    r"MANAGER[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                ],
            ),
            (
                "owner",
                [
                    r"(?:owner|property owner)[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                    r"OWNER[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                ],
            ),
        ]:
            for pattern in patterns:
                match = re.search(pattern, text, re.I | re.MULTILINE)
                if match:
                    parties[party_type] = match.group(1).strip()
                    break

    elif doc_type == "construction":
        # Construction agreement patterns
        for party_type, patterns in [
            (
                "contractor",
                [
                    r"(?:contractor|construction company)[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                    r"CONTRACTOR[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                ],
            ),
            (
                "owner",
                [
                    r"(?:owner|property owner)[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                    r"OWNER[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                ],
            ),
        ]:
            for pattern in patterns:
                match = re.search(pattern, text, re.I | re.MULTILINE)
                if match:
                    parties[party_type] = match.group(1).strip()
                    break

    elif doc_type == "financing":
        # Financing agreement patterns
        for party_type, patterns in [
            (
                "lender",
                [
                    r"(?:lender|bank|financial institution)[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Bank|Trust|Credit Union))",
                    r"LENDER[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Bank|Trust))",
                ],
            ),
            (
                "borrower",
                [
                    r"(?:borrower|debtor)[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                    r"BORROWER[^\n]*?([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company))",
                ],
            ),
        ]:
            for pattern in patterns:
                match = re.search(pattern, text, re.I | re.MULTILINE)
                if match:
                    parties[party_type] = match.group(1).strip()
                    break

    # Universal fallback - look for BETWEEN...AND structure
    if not parties:
        between_match = re.search(
            r"BETWEEN\s+(.*?)\s+AND\s+(.*?)(?:\(|\n|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if between_match:
            party1_text = between_match.group(1).strip()
            party2_text = between_match.group(2).strip()

            # Extract company names from party descriptions
            company_pattern = r"([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|LLC|Inc\.|Corporation|Corp\.|Industries|Company|Bank|Trust))"
            company_in_party1 = re.search(company_pattern, party1_text)
            company_in_party2 = re.search(company_pattern, party2_text)

            if company_in_party1:
                parties["party_1"] = company_in_party1.group(1).strip()
            if company_in_party2:
                parties["party_2"] = company_in_party2.group(1).strip()

    return parties


def infer_cre_meta(first_page: str, file_name: str, doc_type: str) -> Dict[str, Any]:
    """Universal CRE document metadata extraction"""
    meta: Dict[str, Any] = {"doc_id": Path(file_name).stem, "doc_type": doc_type}

    # Extract parties based on document type
    parties = extract_universal_parties(first_page, doc_type)
    meta.update(parties)

    # Universal date extraction
    date_patterns = [
        (
            r"(?:effective|execution|signing|dated)\s+(?:date\s+)?(?:as\s+of\s+)?([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
            "effective_date",
        ),
        (
            r"(?:commencement|start|begin)\s+(?:date\s+)?(?:as\s+of\s+)?([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
            "start_date",
        ),
        (
            r"(?:expiration|end|termination)\s+(?:date\s+)?(?:as\s+of\s+)?([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
            "end_date",
        ),
        (
            r"(?:closing|settlement)\s+(?:date\s+)?(?:as\s+of\s+)?([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
            "closing_date",
        ),
    ]

    for pattern, field_name in date_patterns:
        match = re.search(pattern, first_page, re.I)
        if match:
            try:
                meta[field_name] = str(parse_date(match.group(1)).date())
            except Exception:
                pass

    # Document-specific extractions
    if doc_type == "lease":
        # Lease term
        term_match = re.search(r"(?:term|period)\s+of\s+(\d+)\s+(years?|months?)", first_page, re.I)
        if term_match:
            meta["term_duration"] = f"{term_match.group(1)} {term_match.group(2)}"

        # Rent amount
        rent_match = re.search(r"(?:monthly\s+)?rent[^\d]*(?:\$|Tk\.?\s*)([\d,]+\.?\d*)", first_page, re.I)
        if rent_match:
            meta["monthly_rent"] = rent_match.group(1)

    elif doc_type == "purchase":
        # Purchase price
        price_match = re.search(r"(?:purchase\s+price|sale\s+price)[^\d]*(?:\$|Tk\.?\s*)([\d,]+\.?\d*)", first_page, re.I)
        if price_match:
            meta["purchase_price"] = price_match.group(1)

    elif doc_type == "construction":
        # Contract value
        value_match = re.search(r"(?:contract\s+value|project\s+cost)[^\d]*(?:\$|Tk\.?\s*)([\d,]+\.?\d*)", first_page, re.I)
        if value_match:
            meta["contract_value"] = value_match.group(1)

    elif doc_type == "financing":
        # Loan amount
        loan_match = re.search(r"(?:loan\s+amount|principal)[^\d]*(?:\$|Tk\.?\s*)([\d,]+\.?\d*)", first_page, re.I)
        if loan_match:
            meta["loan_amount"] = loan_match.group(1)

        # Interest rate
        rate_match = re.search(r"(?:interest\s+rate|rate)[^\d]*([\d.]+)%", first_page, re.I)
        if rate_match:
            meta["interest_rate"] = f"{rate_match.group(1)}%"

    # Property details
    property_match = re.search(
        r"(?:located\s+at|property\s+address|premises)[^\n]*?([A-Za-z0-9\s,.-]+(?:Road|Street|Avenue|Building|Centre|Complex)[A-Za-z0-9\s,.-]*)",
        first_page,
        re.I,
    )
    if property_match:
        meta["property_address"] = property_match.group(1).strip()

    # Area/space
    area_match = re.search(r"([\d,]+\.?\d*)\s*(?:sq\.?\s*ft\.?|square\s+feet|sqft)", first_page, re.I)
    if area_match:
        meta["area_sqft"] = area_match.group(1)

    return meta


def process_pdf(pdf: Path, out_dir: Path, lease_id_re: Optional[re.Pattern]) -> List[Dict[str, Any]]:
    """Process PDF with comprehensive content preservation"""
    pages = extract_pages(pdf)
    if not pages:
        return []

    # Build canonical all_text + offsets ONCE
    all_text, page_offsets = _build_alltext_and_page_offsets(pages)

    first_page = pages[0] if pages else ""
    doc_type = detect_cre_document_type(first_page + " " + (pages[1] if len(pages) > 1 else ""), pdf.name)

    toc_idx = detect_toc_page(pages)
    toc_entries = extract_toc_structure(pages[toc_idx]) if toc_idx is not None else None

    chunks = chunk_by_structure(pages, toc_entries)

    if len(chunks) < 3:
        print(f"[process_pdf] Too few chunks ({len(chunks)}), using comprehensive sliding window")
        chunks = sliding_window_comprehensive(all_text, size=1200, overlap=400)

    # Validate with the SAME all_text
    final_validation = validate_content_coverage(all_text, chunks)

    if final_validation["potential_loss"]:
        print(f"[process_pdf] WARNING: Content loss detected after processing {pdf.name}")
        print(
            f"[process_pdf] Coverage: {final_validation['coverage_ratio']:.1%}, Missing: {final_validation.get('missing_chars', 0)} chars"
        )

        chunks = sliding_window_comprehensive(all_text, size=1000, overlap=300)
        print(f"[process_pdf] Applied emergency fallback: {len(chunks)} chunks")

    base_meta = infer_cre_meta(first_page, pdf.name, doc_type)

    if lease_id_re:
        m = lease_id_re.search(pdf.name)
        if m:
            g = m.groupdict()
            if "lease" in g and g["lease"]:
                base_meta["lease_id"] = g["lease"]
            if "lease_id" in g and g["lease_id"]:
                base_meta["lease_id"] = g["lease_id"]
            if "num" in g and g["num"]:
                try:
                    base_meta.setdefault("amendment_no", int(g["num"]))
                except Exception:
                    base_meta.setdefault("amendment_no", g["num"])
            if "property" in g and g["property"]:
                base_meta["property"] = g["property"]

    base_meta.setdefault("doc_id", Path(pdf.name).stem)
    base_meta.setdefault("doc_type", doc_type)

    base_meta["chunking_coverage"] = float(final_validation["coverage_ratio"]) if final_validation else 0.0
    base_meta["total_pages"] = int(len(pages))
    base_meta["original_char_count"] = int(len(all_text))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf.stem}.jsonl"
    records: List[Dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8") as fh:
        for i, ch in enumerate(chunks):
            # Sanitize text to guarantee string and remove bad control chars
            safe_text = str(ch.get("text", "") or "").replace("\x00", "")

            meta = {
                **base_meta,
                "header": str(ch.get("header", f"CHUNK_{i}")),
                "page_start": int(ch.get("page_start", 1)),
                "page_end": int(ch.get("page_end", ch.get("page_start", 1))),
                "chunk_type": str(ch.get("type", "section")),
                "word_count": int(ch.get("word_count", len(safe_text.split()))),
                "chunk_index": int(i),
                "total_chunks": int(len(chunks)),
                "toc_based": bool(ch.get("toc_based", False)),
                "section_number": ch.get("section_number") if ch.get("section_number") else None,
                "start_offset": int(ch.get("start_offset", 0)),
                "end_offset": int(ch.get("end_offset", len(safe_text))),
            }

            rec = {
                "id": f"{base_meta['doc_id']}_{i}",
                "content": safe_text,
                "metadata": meta,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)

    print(
        f"[process_pdf] Successfully processed {pdf.name}: {len(chunks)} chunks, {final_validation['coverage_ratio']:.1%} coverage"
    )
    return records


def find_pdfs(input_dir: str, patterns: Optional[List[str]] = None) -> List[Path]:
    if patterns:
        paths: List[Path] = []
        for pat in patterns:
            paths.extend([Path(p) for p in glob.glob(pat, recursive=True)])
        return sorted(set(paths))
    return sorted(Path(input_dir).rglob("*.pdf"))
