from __future__ import annotations

import os
import re
import time
import shutil
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

SEC_BASE = "https://data.sec.gov"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
TICKERS_JSON = "https://www.sec.gov/files/company_tickers.json"

USER_AGENT = os.getenv("SEC_USER_AGENT", "QuantaMind Research (contact: youremail@example.com)")
REQUEST_PAUSE_SEC = float(os.getenv("SEC_REQUEST_PAUSE_SEC", "0.2"))

def _requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return s

def _zfill_cik(raw_cik: str) -> str:
    import re as _re
    digits = _re.sub(r"\D+", "", str(raw_cik or ""))
    if not digits:
        raise ValueError("CIK is empty after stripping.")
    return digits.zfill(10)


def resolve_cik(ticker: Optional[str], cik: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
    if cik:
        return _zfill_cik(cik), None, None
    if not ticker:
        raise ValueError("Either 'cik' or 'ticker' must be provided.")
    sess = _requests_session()
    resp = sess.get(TICKERS_JSON, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    t_upper = ticker.strip().upper()
    for _, entry in data.items():
        if entry.get("ticker", "").upper() == t_upper:
            cik10 = _zfill_cik(entry.get("cik_str", ""))
            company = entry.get("title")
            return cik10, company, t_upper
    raise ValueError(f"Ticker '{ticker}' not found in SEC company_tickers.json")

def _fetch_json(url: str, sess: requests.Session) -> Dict[str, Any]:
    time.sleep(REQUEST_PAUSE_SEC)
    r = sess.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def get_company_submissions(cik10: str, sess: requests.Session) -> Dict[str, Any]:
    url = f"{SEC_BASE}/submissions/CIK{cik10}.json"
    return _fetch_json(url, sess)

def _in_date_range(date_str: str, since: Optional[str], until: Optional[str]) -> bool:
    if not date_str:
        return False
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    if since:
        ds = datetime.strptime(since, "%Y-%m-%d").date()
        if d < ds:
            return False
    if until:
        du = datetime.strptime(until, "%Y-%m-%d").date()
        if d > du:
            return False
    return True

def list_recent_filings(
    cik10: str,
    forms: List[str],
    since: Optional[str],
    until: Optional[str],
    limit: int,
    sess: requests.Session,
) -> List[Dict[str, Any]]:
    sub = get_company_submissions(cik10, sess)
    recent = sub.get("filings", {}).get("recent", {})
    acc = recent.get("accessionNumber", [])
    form = recent.get("form", [])
    filing_date = recent.get("filingDate", [])
    report_date = recent.get("reportDate", [])
    primary = recent.get("primaryDocument", [])

    rows = []
    for i in range(min(len(acc), len(form), len(filing_date), len(primary))):
        rows.append(
            {
                "accessionNumber": acc[i],
                "form": form[i],
                "filingDate": filing_date[i],
                "reportDate": report_date[i] if i < len(report_date) else None,
                "primaryDocument": primary[i],
            }
        )

    forms_set = {f.upper() for f in forms} if forms else set()
    out = []
    for r in rows:
        if forms_set and r["form"].upper() not in forms_set:
            continue
        if since or until:
            if not _in_date_range(r["filingDate"], since, until):
                continue
        out.append(r)

    out.sort(key=lambda x: (x["filingDate"], x["accessionNumber"]), reverse=True)
    return out[:limit]

def _filing_dir_parts(cik10: str, accession_no: str) -> Tuple[str, str]:
    cik_no_zeros = str(int(cik10))
    acc_nodash = accession_no.replace("-", "")
    return cik_no_zeros, acc_nodash

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _download_binary(url: str, dest: Path, sess: requests.Session) -> None:
    time.sleep(REQUEST_PAUSE_SEC)
    r = sess.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        shutil.copyfileobj(r.raw, fh)

def _save_text(url: str, dest: Path, sess: requests.Session) -> None:
    time.sleep(REQUEST_PAUSE_SEC)
    r = sess.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)

def _looks_like_ex10(name: str) -> bool:
    n = name.lower()
    if not re.search(r"\.(htm|html|txt|pdf)$", n):
        return False
    return bool(re.search(r"(ex[-_ ]?10|ex10)", n))

def download_filing_and_exhibits(
    cik10: str,
    filing: Dict[str, Any],
    base_dir: Path,
    sess: requests.Session,
) -> Dict[str, Any]:
    accession_no = filing["accessionNumber"]
    form = filing["form"]
    filing_date = filing.get("filingDate")
    report_date = filing.get("reportDate")
    primary_doc = filing["primaryDocument"]

    cik_no_zeros, acc_nodash = _filing_dir_parts(cik10, accession_no)
    filing_root_url = f"{ARCHIVES}/{cik_no_zeros}/{acc_nodash}"
    primary_url = f"{filing_root_url}/{primary_doc}"
    index_json_url = f"{filing_root_url}/index.json"

    target_dir = Path(base_dir) / "raw" / cik10 / acc_nodash
    _ensure_dir(target_dir)

    primary_path = target_dir / primary_doc
    try:
        _save_text(primary_url, primary_path, sess)
    except Exception as e:
        logger.warning(f"Primary doc download failed: {primary_url} - {e}")
        primary_path = None

    exhibits_saved: List[str] = []
    try:
        idx = _fetch_json(index_json_url, sess)
        files = idx.get("directory", {}).get("item", []) or []
        for f in files:
            name = f.get("name") or ""
            if _looks_like_ex10(name):
                url = f"{filing_root_url}/{name}"
                dest = target_dir / name
                try:
                    _download_binary(url, dest, sess)
                    exhibits_saved.append(str(dest.as_posix()))
                except Exception as ex:
                    logger.warning(f"Exhibit download failed: {url} - {ex}")
    except Exception as e:
        logger.info(f"No index.json or failed to fetch for {accession_no}: {e}")

    return {
        "accession_no": accession_no,
        "form": form,
        "filing_date": filing_date,
        "report_date": report_date,
        "primary_doc": str(primary_path.as_posix()) if primary_path else None,
        "exhibits": exhibits_saved,
    }

@dataclass
class DownloadPlan:
    cik10: str
    company: Optional[str]
    ticker: Optional[str]
    filings: List[Dict[str, Any]]

def plan_downloads(
    ticker: Optional[str],
    cik: Optional[str],
    forms: List[str],
    since: Optional[str],
    until: Optional[str],
    limit: int,
) -> DownloadPlan:
    sess = _requests_session()
    cik10, company, norm_ticker = resolve_cik(ticker, cik)
    filings = list_recent_filings(cik10, forms, since, until, limit, sess)
    return DownloadPlan(cik10=cik10, company=company, ticker=norm_ticker, filings=filings)

def execute_downloads(plan: DownloadPlan, out_dir: str) -> Dict[str, Any]:
    sess = _requests_session()
    base_dir = Path(out_dir)
    saved: List[Dict[str, Any]] = []
    for f in plan.filings:
        try:
            saved_item = download_filing_and_exhibits(plan.cik10, f, base_dir, sess)
            saved.append(saved_item)
        except Exception as e:
            logger.error(f"Failed to save filing {f.get('accessionNumber')}: {e}")
    return {
        "cik": plan.cik10,
        "company": plan.company,
        "ticker": plan.ticker,
        "forms": list({x["form"] for x in plan.filings}),
        "saved_count": len(saved),
        "saved": saved,
        "base_dir": str((base_dir / "raw" / plan.cik10).as_posix()),
        "note": "Files stored under base_dir by accession number (no dashes). Exhibits include EX-10.* where present.",
    }
