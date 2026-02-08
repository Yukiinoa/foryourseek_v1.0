from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

from ..adapters.papers.enrich_meta import enrich_paper_from_html
from ..adapters.papers.rss_generic import parse_rss
from ..core.db import Database, PaperRow
from ..core.doi_meta import (
    fetch_crossref_abstract,
    search_crossref_by_title,
    search_openalex_by_journal,
    search_openalex_by_source_id,
    search_openalex_by_title,
    search_openalex_by_title_source_id,
)
from ..core.http import HttpClient
from ..core.llm import LLMClient
from ..core.logging import log_error, log_event
from ..core.publisher_api import fetch_publisher_abstract
from ..core.schema import normalize_paper_fields
from ..core.text_cleaner import fix_text_encoding, truncate_text
from ..core.utils import make_paper_id, make_paper_ids


def _match_keywords(hay: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    h = hay.lower()
    return any(k.lower() in h for k in keywords if k)


def _match_any(hay: str, keywords: List[str]) -> bool:
    if not keywords:
        return False
    h = hay.lower()
    return any(k.lower() in h for k in keywords if k)


def _prefer_abs_url(url: str) -> str:
    if not url:
        return url
    low = url.lower()
    if "tandfonline.com" not in low:
        return url
    if "/doi/abs/" in low:
        return url
    if "/doi/full/" in low:
        return url.replace("/doi/full/", "/doi/abs/")
    if "/doi/pdf/" in low:
        return url.replace("/doi/pdf/", "/doi/abs/")
    return url


def _prefilter_score(
    title: str,
    abstract: str,
    include_keywords: List[str],
    exclude_keywords: List[str],
    *,
    title_weight: int,
    abstract_weight: int,
    allow_if_no_keywords: bool,
) -> tuple[int, List[str], List[str], bool]:
    hay_title = (title or "").lower()
    hay_abs = (abstract or "").lower()
    hay = f"{hay_title}\n{hay_abs}"

    if _match_any(hay, exclude_keywords):
        return -1, [], [], True

    if not include_keywords:
        return (1 if allow_if_no_keywords else 0), [], [], False

    t_hits = [k for k in include_keywords if k and k.lower() in hay_title]
    a_hits = [k for k in include_keywords if k and k.lower() in hay_abs]
    score = len(set(t_hits)) * int(title_weight) + len(set(a_hits)) * int(abstract_weight)
    # keep order for display
    t_hits = list(dict.fromkeys(t_hits))
    a_hits = list(dict.fromkeys(a_hits))
    return score, t_hits, a_hits, False


def run_paper_agent(
    *,
    run_id: str,
    cfg: Dict[str, Any],
    db: Database,
    http: HttpClient,
    llm: LLMClient,
) -> Dict[str, Any]:
    profile = cfg["profile"]
    papers_cfg = cfg["papers"]
    extract_cfg = cfg.get("extract", {})
    min_len = int(extract_cfg.get("min_len", 120))
    enable_fallback = extract_cfg.get("fallback", "trafilatura") != "none"
    include_kw = profile.get("include_keywords", [])
    exclude_kw = profile.get("exclude_keywords", [])

    journals = papers_cfg.get("journals", [])
    max_per_day = int(papers_cfg.get("max_per_day", 30))
    max_candidates = int(papers_cfg.get("max_candidates_per_run", max_per_day))
    max_llm = int(papers_cfg.get("max_llm_per_run", max_per_day))
    min_per_journal = int(papers_cfg.get("min_candidates_per_journal", 0))
    pre_cfg = papers_cfg.get("prefilter", {}) or {}
    pre_title_w = int(pre_cfg.get("title_weight", 3))
    pre_abs_w = int(pre_cfg.get("abstract_weight", 1))
    pre_min_score = int(pre_cfg.get("min_score_for_llm", 2))
    pre_allow_no_kw = bool(pre_cfg.get("allow_if_no_keywords", True))

    found = 0
    new_count = 0
    strong_count = 0
    budget_hit_count = 0
    llm_input_tokens_sum = 0
    llm_output_tokens_sum = 0
    llm_total_tokens_sum = 0
    llm_usage_by_provider: Dict[str, Dict[str, Any]] = {}
    new_items_for_email: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    rule_skipped_total = 0
    prefilter_candidates_total = 0
    prefilter_eligible_total = 0
    prefilter_skipped_total = 0
    debug_cfg = papers_cfg.get("debug", {}) or {}
    export_audit = bool(debug_cfg.get("export_audit", False))
    audit_limit = int(debug_cfg.get("audit_limit", 0) or 0)
    source_counts: Dict[str, int] = {}
    journal_stats: Dict[str, Dict[str, Any]] = {}

    def _bump_usage(
        provider: str, model: str, in_tokens: int, out_tokens: int, total_tokens: int
    ) -> None:
        if not provider:
            return
        entry = llm_usage_by_provider.setdefault(
            provider,
            {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "models": {},
            },
        )
        entry["calls"] += 1
        entry["input_tokens"] += int(in_tokens or 0)
        entry["output_tokens"] += int(out_tokens or 0)
        entry["total_tokens"] += int(total_tokens or 0)
        if model:
            models = entry.setdefault("models", {})
            models[model] = int(models.get(model, 0)) + 1

    journal_buckets: List[Dict[str, Any]] = []
    for j in journals:
        name = j.get("name") or "(unknown)"
        rss_url = j.get("rss") or ""
        if not rss_url:
            continue

        entries: List[Dict[str, Any]] = []
        rss_count = 0
        fallback_method = ""
        fallback_count = 0
        rss_source = "rss"
        rss_error = ""
        rss_tb = None
        try:
            try:
                rss_text = http.get_text(rss_url)
            except Exception:
                rss_text = http.get_text_smart(rss_url)
            entries = parse_rss(rss_text, source_name=name)
            rss_count = len(entries)
            found += rss_count
        except Exception as ex:
            rss_tb = traceback.format_exc()
            rss_error = repr(ex)
            fb_cfg = j.get("rss_fallback") or {}
            method = (fb_cfg.get("method") or "").lower()
            if method == "openalex":
                fb_limit = int(fb_cfg.get("max_results", max_candidates))
                fb_journal = fb_cfg.get("journal_name") or name
                fb_source_id = fb_cfg.get("openalex_source_id") or fb_cfg.get("source_id") or ""
                if fb_source_id:
                    entries = search_openalex_by_source_id(fb_source_id, http=http, limit=fb_limit)
                else:
                    entries = search_openalex_by_journal(fb_journal, http=http, limit=fb_limit)
                if entries:
                    fallback_method = "openalex"
                    fallback_count = len(entries)
                    rss_source = "openalex"
                    found += fallback_count
                    log_event(
                        "paper_rss_fallback",
                        run_id=run_id,
                        journal=name,
                        method=fallback_method,
                        entries=fallback_count,
                        rss_error=rss_error,
                    )
            if not entries:
                db.record_error(
                    run_id,
                    module="paper",
                    source=name,
                    error_type=type(ex).__name__,
                    message=rss_tb or repr(ex),
                )
                log_error(
                    "paper_agent_source_failed",
                    run_id=run_id,
                    source=name,
                    error=repr(ex),
                )
                continue

        items: List[Dict[str, Any]] = []
        seen_skipped = 0
        seen_hit_types = {"doi": 0, "url": 0, "title": 0}
        seen_hit_multi = 0
        seen_hit_samples: List[Dict[str, Any]] = []
        for e in entries:
            if len(items) >= max_candidates:
                break

            title = e.get("title", "")
            url = e.get("url", "")
            doi = e.get("doi", "")
            published_at = e.get("published_at", "")
            summary = e.get("summary", "")

            item_id = make_paper_id(doi=doi, url=url, title=title)
            candidate_ids = make_paper_ids(doi=doi, url=url, title=title)
            seen_ids = db.find_seen_ids(candidate_ids)
            if seen_ids:
                seen_skipped += 1
                hit_types = set()
                for sid in seen_ids:
                    if sid.startswith("doi:"):
                        hit_types.add("doi")
                    elif sid.startswith("url:"):
                        hit_types.add("url")
                    elif sid.startswith("title:"):
                        hit_types.add("title")
                for ht in hit_types:
                    seen_hit_types[ht] += 1
                if len(hit_types) > 1:
                    seen_hit_multi += 1
                if len(seen_hit_samples) < 3:
                    seen_hit_samples.append(
                        {
                            "title": title,
                            "hit_types": sorted(hit_types),
                            "seen_ids": seen_ids[:3],
                        }
                    )
                continue

            items.append(
                {
                    "item_id": item_id,
                    "title": title,
                    "url": url,
                    "doi": doi,
                    "published_at": published_at,
                    "summary": summary,
                    "journal": name,
                    "journal_cfg": j,
                }
            )

        journal_buckets.append({"name": name, "items": items})
        journal_stats[name] = {
            "rss_url": rss_url,
            "rss_entries": rss_count,
            "rss_fallback_method": fallback_method,
            "rss_fallback_entries": fallback_count,
            "rss_source": rss_source,
            "rss_error": rss_error,
            "seen_skipped": seen_skipped,
            "seen_hit_types": seen_hit_types,
            "seen_hit_multi": seen_hit_multi,
            "seen_hit_samples": seen_hit_samples,
            "unseen_candidates": len(items),
            "sample_titles": [it.get("title", "") for it in items[:3]],
            "sample_dois": [it.get("doi", "") for it in items[:3]],
        }

    # Candidate selection: per-journal minimum then round-robin fill
    selected: List[Dict[str, Any]] = []
    journal_order = [b["name"] for b in journal_buckets]
    bucket_map = {b["name"]: list(b["items"]) for b in journal_buckets}

    for _ in range(min_per_journal):
        for name in journal_order:
            if len(selected) >= max_candidates:
                break
            items = bucket_map.get(name) or []
            if items:
                selected.append(items.pop(0))
                bucket_map[name] = items
        if len(selected) >= max_candidates:
            break

    while len(selected) < max_candidates:
        progressed = False
        for name in journal_order:
            items = bucket_map.get(name) or []
            if items:
                selected.append(items.pop(0))
                bucket_map[name] = items
                progressed = True
                if len(selected) >= max_candidates:
                    break
        if not progressed:
            break

    selected_counts: Dict[str, int] = {}
    selected_samples: Dict[str, List[str]] = {}
    for c0 in selected:
        jname = c0.get("journal") or "(unknown)"
        selected_counts[jname] = selected_counts.get(jname, 0) + 1
        if jname not in selected_samples:
            selected_samples[jname] = []
        if len(selected_samples[jname]) < 3:
            selected_samples[jname].append(c0.get("title", ""))
    for name, stats in journal_stats.items():
        stats["selected"] = selected_counts.get(name, 0)
        stats["selected_sample_titles"] = selected_samples.get(name, [])

    for c0 in selected:
        if len(candidates) >= max_candidates:
            break

        title = c0["title"]
        url = c0["url"]
        doi = c0["doi"]
        published_at = c0["published_at"]
        summary = c0["summary"]
        name = c0["journal"]
        journal_cfg = c0["journal_cfg"]

        # enrichment
        authors = ""
        abstract = summary
        abstract_source = "rss"
        extract_method = "rss"
        extract_warnings: List[str] = []
        detail_error: str | None = None
        enrich_cfg = journal_cfg.get("enrich") or {}

        # Configurable publisher APIs fallback (for any provider) when still only RSS or abstract is short
        if abstract_source == "rss" or not abstract or len(abstract) < min_len:
            try:
                prov_abs, prov_name = fetch_publisher_abstract(
                    cfg=cfg,
                    doi=doi,
                    url=url,
                    title=title,
                    journal=name,
                    http=http,
                )
                if prov_abs:
                    abstract = prov_abs
                    abstract_source = prov_name or "publisher_api"
                    extract_method = abstract_source
                    extract_warnings = []
            except Exception:
                pass

        # DOI fallback (Crossref) when abstract is still only RSS or too short
        if doi and (abstract_source == "rss" or not abstract or len(abstract) < min_len):
            try:
                doi_abs = fetch_crossref_abstract(doi, http=http)
                if doi_abs:
                    abstract = doi_abs
                    abstract_source = "crossref"
                    extract_method = "crossref"
                    extract_warnings = []
            except Exception:
                pass

        # Title search fallback (Crossref) when DOI is missing, abstract is short,
        # or the abstract is still only from RSS.
        if (
            abstract_source == "rss" or not doi or not abstract or len(abstract) < min_len
        ) and title:
            try:
                found_doi, found_abs = search_crossref_by_title(title, name, http=http)
                if found_doi and not doi:
                    doi = found_doi
                if found_abs:
                    abstract = found_abs
                    abstract_source = "crossref_title"
                    extract_method = "crossref_title"
                    extract_warnings = []
            except Exception:
                pass

        # OpenAlex fallback when still only RSS or abstract is short
        if (abstract_source == "rss" or not abstract or len(abstract) < min_len) and title:
            try:
                openalex_source_id = (journal_cfg.get("openalex_source_id") or "").strip()
                if openalex_source_id:
                    found_doi, found_abs = search_openalex_by_title_source_id(
                        title, openalex_source_id, http=http
                    )
                else:
                    found_doi, found_abs = search_openalex_by_title(title, name, http=http)
                if found_doi and not doi:
                    doi = found_doi
                if found_abs:
                    abstract = found_abs
                    abstract_source = "openalex_title"
                    extract_method = "openalex_title"
                    extract_warnings = []
            except Exception:
                pass

        # HTML enrich (meta tags / JSON-LD). Try last to avoid 403-heavy sites.
        if enrich_cfg.get("enabled") and enrich_cfg.get("method") == "meta_tags" and url:
            need_html = (
                abstract_source == "rss" or not abstract or len(abstract) < min_len or not doi
            )
            if need_html:
                try:
                    html_url = _prefer_abs_url(url)
                    html = http.get_text_smart(html_url)
                    enriched = enrich_paper_from_html(
                        html,
                        abstract_selector=enrich_cfg.get("abstract_selector", "") or "",
                        min_len=min_len,
                        enable_fallback=enable_fallback,
                    )
                    doi = enriched.get("doi") or doi
                    authors = enriched.get("authors") or authors
                    if (
                        abstract_source == "rss" or not abstract or len(abstract) < min_len
                    ) and enriched.get("abstract"):
                        abstract = enriched.get("abstract") or abstract
                        abstract_source = enriched.get("abstract_source") or abstract_source
                        extract_method = (
                            enriched.get("extract_method") or abstract_source or extract_method
                        )
                        extract_warnings = list(enriched.get("extract_warnings") or [])
                except Exception as ex:
                    detail_error = repr(ex)

        abstract = fix_text_encoding(abstract)

        abstract_ok = bool(abstract and len(abstract) >= min_len)
        source_ok = abstract_source != "rss"
        if detail_error and not (abstract_ok or source_ok):
            log_error(
                "paper_enrich_failed",
                run_id=run_id,
                source=name,
                url=url,
                error=detail_error,
            )

        source_counts[abstract_source] = source_counts.get(abstract_source, 0) + 1
        log_event(
            "paper_abstract_source",
            run_id=run_id,
            source=abstract_source,
            extract_method=extract_method,
            journal=name,
            doi=doi,
            url=url,
        )

        pre_score, pre_title_hits, pre_abs_hits, pre_excluded = _prefilter_score(
            title,
            abstract,
            include_kw,
            exclude_kw,
            title_weight=pre_title_w,
            abstract_weight=pre_abs_w,
            allow_if_no_keywords=pre_allow_no_kw,
        )

        candidates.append(
            {
                "item_id": c0["item_id"],
                "title": title,
                "url": url,
                "doi": doi,
                "published_at": published_at,
                "summary": summary,
                "journal": name,
                "authors": authors,
                "abstract": abstract,
                "abstract_source": abstract_source,
                "extract_method": extract_method,
                "extract_warnings": extract_warnings,
                "prefilter_score": pre_score,
                "prefilter_excluded": pre_excluded,
                "prefilter_title_hits": pre_title_hits,
                "prefilter_abs_hits": pre_abs_hits,
            }
        )
        prefilter_candidates_total += 1

    # Sort by prefilter score and only send top-N to LLM
    candidates.sort(key=lambda x: x.get("prefilter_score", 0), reverse=True)
    llm_used = 0
    for c in candidates:
        if new_count >= max_candidates:
            break

        title = c["title"]
        url = c["url"]
        doi = c["doi"]
        published_at = c["published_at"]
        name = c["journal"]
        authors = c["authors"]
        abstract = c["abstract"]
        abstract_source = c["abstract_source"]
        extract_method = c["extract_method"]
        extract_warnings = c["extract_warnings"]
        pre_score = int(c.get("prefilter_score", 0))
        pre_excluded = bool(c.get("prefilter_excluded", False))
        pre_title_hits = c.get("prefilter_title_hits") or []
        pre_abs_hits = c.get("prefilter_abs_hits") or []
        eligible_for_llm = (pre_score >= pre_min_score) or (not include_kw and pre_allow_no_kw)
        llm_provider = ""
        llm_model = ""

        if pre_excluded:
            rule_skipped_total += 1
            rec = "skip"
            score = 0
            summary_bullets = []
            reasons = ["命中排除关键词（规则跳过）"]
            llm_status = "skipped"
            llm_input_chars = 0
            llm_input_tokens = 0
            llm_output_tokens = 0
            llm_total_tokens = 0
            budget_hit = 0
            prefilter_skip_reason = "rule_exclude"
        elif llm_used < max_llm and eligible_for_llm:
            prefilter_eligible_total += 1
            llm_used += 1
            llm_out = llm.analyze_paper(
                profile=profile,
                thresholds={
                    "strong_threshold": int(papers_cfg.get("strong_threshold", 8)),
                    "normal_threshold": int(papers_cfg.get("normal_threshold", 5)),
                },
                title=title,
                abstract=abstract,
                journal=name,
                published_at=published_at,
                url=url,
            )
            rec = llm_out.recommendation
            score = llm_out.relevance_score
            summary_bullets = llm_out.summary_bullets
            reasons = llm_out.reasons
            llm_status = llm_out.llm_status
            llm_input_chars = llm_out.llm_input_chars
            llm_input_tokens = llm_out.llm_input_tokens
            llm_output_tokens = llm_out.llm_output_tokens
            llm_total_tokens = llm_out.llm_total_tokens
            llm_provider = getattr(llm_out, "llm_provider", "") or ""
            llm_model = getattr(llm_out, "llm_model", "") or ""
            budget_hit = llm_out.budget_hit
            if budget_hit:
                budget_hit_count += 1
            llm_input_tokens_sum += llm_input_tokens
            llm_output_tokens_sum += llm_output_tokens
            llm_total_tokens_sum += llm_total_tokens
            _bump_usage(
                llm_provider,
                llm_model,
                llm_input_tokens,
                llm_output_tokens,
                llm_total_tokens,
            )
            prefilter_skip_reason = ""
        else:
            rec = "skip"
            score = 0
            summary_bullets = []
            if not eligible_for_llm:
                prefilter_skipped_total += 1
                if include_kw and pre_score <= 0:
                    reasons = ["粗筛：未命中关注关键词"]
                else:
                    reasons = [f"粗筛得分 {pre_score} < 阈值 {pre_min_score}"]
                prefilter_skip_reason = "score_below_threshold"
            else:
                prefilter_eligible_total += 1
                prefilter_skipped_total += 1
                reasons = [f"粗筛排序未入选（LLM 名额 {max_llm} 已用完）"]
                prefilter_skip_reason = "llm_quota_full"
            if pre_title_hits or pre_abs_hits:
                reasons.append(f"命中：title {pre_title_hits[:3]} / abstract {pre_abs_hits[:3]}")
            llm_status = "prefilter_skip"
            llm_input_chars = 0
            llm_input_tokens = 0
            llm_output_tokens = 0
            llm_total_tokens = 0
            llm_provider = ""
            llm_model = ""
            budget_hit = 0

        # store & mark seen (alias dedup if DOI available)
        item_id = make_paper_id(doi=doi, url=url, title=title)
        seen_ids = make_paper_ids(doi=doi, url=url, title=title)
        if not seen_ids:
            seen_ids = [item_id]
        db.mark_seen_many(seen_ids, item_type="paper", source=name)
        db.upsert_paper(
            PaperRow(
                id=item_id,
                title=title,
                url=url,
                doi=doi,
                published_at=published_at,
                journal=name,
                authors=authors,
                abstract=abstract,
                abstract_source=abstract_source,
                extract_method=extract_method,
                extract_warnings=extract_warnings,
                summary_bullets=summary_bullets,
                relevance_score=score,
                recommendation=rec,
                reasons=reasons,
                llm_status=llm_status,
                llm_input_chars=llm_input_chars,
                llm_input_tokens=llm_input_tokens,
                llm_output_tokens=llm_output_tokens,
                llm_total_tokens=llm_total_tokens,
                budget_hit=budget_hit,
            )
        )

        new_count += 1
        if rec == "strong":
            strong_count += 1

        item = {
            "id": item_id,
            "title": title,
            "url": url,
            "doi": doi,
            "published_at": published_at,
            "journal": name,
            "abstract_source": abstract_source,
            "extract_method": extract_method,
            "extract_warnings": extract_warnings,
            "abstract_preview": (
                abstract if rec in ("strong", "normal") else truncate_text(abstract, 600)
            ),
            "summary_bullets": summary_bullets,
            "relevance_score": score,
            "recommendation": rec,
            "reasons": reasons,
            "rule_skip": pre_excluded,
            "rule_skip_reason": "exclude" if pre_excluded else "",
            "prefilter_score": pre_score,
            "prefilter_excluded": pre_excluded,
            "prefilter_title_hits": pre_title_hits,
            "prefilter_abs_hits": pre_abs_hits,
            "prefilter_skip_reason": prefilter_skip_reason,
            "llm_status": llm_status,
            "llm_input_chars": llm_input_chars,
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
            "llm_total_tokens": llm_total_tokens,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "budget_hit": budget_hit,
        }
        item.update(normalize_paper_fields(item))
        new_items_for_email.append(item)

    log_event(
        "paper_agent_done",
        run_id=run_id,
        papers_found=found,
        papers_new=new_count,
        papers_strong=strong_count,
        llm_input_tokens=llm_input_tokens_sum,
        llm_output_tokens=llm_output_tokens_sum,
        llm_total_tokens=llm_total_tokens_sum,
        llm_usage_by_provider=llm_usage_by_provider,
    )
    log_event(
        "paper_rule_skip_stats",
        run_id=run_id,
        rule_skipped=rule_skipped_total,
        prefilter_candidates=prefilter_candidates_total,
        prefilter_eligible=prefilter_eligible_total,
        prefilter_skipped=prefilter_skipped_total,
        prefilter_min_score=pre_min_score,
        prefilter_keywords_count=len(include_kw),
    )
    log_event("paper_journal_capture_stats", run_id=run_id, stats=journal_stats)
    log_event("paper_abstract_source_stats", run_id=run_id, stats=source_counts)
    if export_audit:
        Path("logs").mkdir(parents=True, exist_ok=True)
        out_path = Path("logs") / f"paper_audit_{run_id}.jsonl"
        count = 0
        with out_path.open("w", encoding="utf-8") as f:
            for item in new_items_for_email:
                if audit_limit > 0 and count >= audit_limit:
                    break
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
        log_event(
            "paper_audit_exported",
            run_id=run_id,
            path=str(out_path),
            count=count,
        )
    return {
        "papers_found": found,
        "papers_new": new_count,
        "papers_strong": strong_count,
        "budget_hit": budget_hit_count,
        "llm_input_tokens": llm_input_tokens_sum,
        "llm_output_tokens": llm_output_tokens_sum,
        "llm_total_tokens": llm_total_tokens_sum,
        "llm_usage_by_provider": llm_usage_by_provider,
        "papers_new_items": new_items_for_email,
    }
