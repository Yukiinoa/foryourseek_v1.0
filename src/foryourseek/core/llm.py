from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from jsonschema import ValidationError
from jsonschema import validate as js_validate

from .budget import Budget
from .llm_providers.registry import get_provider
from .logging import log_event
from .text_cleaner import clean_text_for_llm, head_sentences, truncate_text
from .utils import safe_int

LLM_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["summary_bullets", "relevance_score", "recommendation", "reasons"],
    "properties": {
        "summary_bullets": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "string"},
        },
        "relevance_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "recommendation": {"type": "string", "enum": ["strong", "normal", "skip"]},
        "reasons": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "string"},
        },
        "keywords": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


@dataclass
class LLMResult:
    summary_bullets: List[str]
    relevance_score: int
    recommendation: str
    reasons: List[str]
    keywords: List[str]
    confidence: float
    llm_status: str  # ok|skipped|failed
    raw_json: str = ""
    llm_input_chars: int = 0
    budget_hit: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0
    llm_provider: str = ""
    llm_model: str = ""


def _decide_recommendation(score: int, strong_th: int, normal_th: int) -> str:
    if score >= strong_th:
        return "strong"
    if score >= normal_th:
        return "normal"
    return "skip"


def _split_sentences(text: str) -> List[str]:
    t = clean_text_for_llm(text)
    if not t:
        return []
    parts = re.split(r"(?<=[.!?。！？])\\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _preview(text: str, limit: int = 240) -> str:
    if not text:
        return ""
    t = text.replace("\n", "\\n")
    if len(t) <= limit:
        return t
    return t[:limit] + "…"


def _extract_first_json_object(text: str) -> str:
    """Extract the first JSON object substring from a blob of text.

    This is a best-effort parser for providers that ignore "JSON only" constraints
    and wrap the JSON in markdown or reasoning text.
    """
    if not text:
        return ""

    # Prefer fenced ```json ... ``` blocks when present.
    m = re.search(r"```json\\s*(.*?)\\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        cand = (m.group(1) or "").strip()
        if cand:
            return cand

    # Generic fenced code block fallback.
    m = re.search(r"```\\s*(.*?)\\s*```", text, flags=re.DOTALL)
    if m:
        cand = (m.group(1) or "").strip()
        # Often the code block is JSON without language tag.
        if cand.startswith("{") and cand.endswith("}"):
            return cand

    start = text.find("{")
    if start < 0:
        return ""

    # Bracket-balance scan with string/escape awareness.
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
            continue

    return ""


def _pick_sentence(text: str, keywords: List[str]) -> str:
    if not text:
        return ""
    for sent in _split_sentences(text):
        low = sent.lower()
        for kw in keywords:
            if kw and kw.lower() in low:
                return sent.strip()
    return ""


def _build_llm_result(
    obj: Dict[str, Any],
    content: str,
    usage: Dict[str, int],
    llm_input_chars: int,
    strong_th: int,
    normal_th: int,
    provider_name: str,
    model_name: str,
) -> LLMResult:
    in_tokens = safe_int(usage.get("input") if usage else 0, 0)
    out_tokens = safe_int(usage.get("output") if usage else 0, 0)
    total_tokens = safe_int(usage.get("total") if usage else 0, 0)
    if in_tokens == 0 and llm_input_chars > 0:
        in_tokens = max(1, llm_input_chars // 4)
    if out_tokens == 0 and content:
        out_tokens = max(1, len(content) // 4)
    if total_tokens == 0:
        total_tokens = in_tokens + out_tokens

    score = safe_int(obj.get("relevance_score"), 0)
    rec = obj.get("recommendation") or _decide_recommendation(score, strong_th, normal_th)
    return LLMResult(
        summary_bullets=list(obj.get("summary_bullets") or [])[:3],
        relevance_score=score,
        recommendation=rec,
        reasons=list(obj.get("reasons") or [])[:3],
        keywords=list(obj.get("keywords") or [])[:10],
        confidence=float(obj.get("confidence") or 0.5),
        llm_status="ok",
        raw_json=content,
        llm_input_chars=llm_input_chars,
        budget_hit=0,
        llm_input_tokens=in_tokens,
        llm_output_tokens=out_tokens,
        llm_total_tokens=total_tokens,
        llm_provider=provider_name or "",
        llm_model=model_name or "",
    )


def _fallback_bullets(text: str, *, output_language: str) -> List[str]:
    if not text:
        return [
            (
                "(No usable abstract; rule-based downgrade.)"
                if (output_language or "zh").lower() == "en"
                else "（无可用摘要，规则降级）"
            )
        ]
    sentences = _split_sentences(text)
    bullets: List[str] = []
    for sent in sentences:
        if len(sent) > 220:
            bullets.extend(textwrap.wrap(sent, width=180))
        else:
            bullets.append(sent)
        if len(bullets) >= 3:
            break
    bullets = [b.strip() for b in bullets if b.strip()]
    if not bullets:
        bullets = [text[:200].strip()] if text else []
    return bullets[:3]


def _keyword_fallback(
    title: str,
    text: str,
    include_keywords: List[str],
    exclude_keywords: List[str],
    strong_th: int,
    normal_th: int,
    *,
    agent: str,
    output_language: str = "zh",
    llm_input_chars: int = 0,
    llm_status: str = "failed",
    budget_hit: int = 0,
    llm_input_tokens: int = 0,
    llm_output_tokens: int = 0,
    llm_total_tokens: int = 0,
    llm_provider: str = "",
    llm_model: str = "",
) -> LLMResult:
    agent_norm = (agent or "papers").lower()
    if agent_norm not in ("papers", "jobs"):
        agent_norm = "papers"

    def _compact(s: str, max_len: int = 260) -> str:
        s = re.sub(r"\s+", " ", (s or "")).strip()
        if len(s) <= max_len:
            return s
        return s[: max_len - 1].rstrip() + "…"

    hay = f"{title}\n{text}".lower()
    if any(k.lower() in hay for k in exclude_keywords or []):
        score = 0
        if agent_norm == "jobs":
            reasons = [
                (
                    "[Topic Connection] Excluded by negative keywords (rule-based)."
                    if output_language == "en"
                    else "[Topic Connection] 命中排除关键词（规则跳过）"
                ),
                ("[Funding] Not mentioned." if output_language == "en" else "[Funding] 未提及"),
                (
                    "[Key Requirement] Not mentioned."
                    if output_language == "en"
                    else "[Key Requirement] 未提及"
                ),
            ]
        else:
            reasons = [
                (
                    "[Score breakdown] total=0/10 (excluded by negative keywords; LLM unavailable)"
                    if output_language == "en"
                    else "[Score breakdown] 总分 0/10（命中排除关键词；LLM 不可用）"
                ),
                (
                    "[Critical value] Skipped (excluded)."
                    if output_language == "en"
                    else "[Critical value] 已跳过（被排除）"
                ),
                (
                    "[Limitations] Skipped (excluded)."
                    if output_language == "en"
                    else "[Limitations] 已跳过（被排除）"
                ),
            ]
    else:
        hits = [k for k in (include_keywords or []) if k.lower() in hay]
        score = min(10, 3 + 2 * len(hits))
        if agent_norm == "jobs":
            funding_keys = [
                "salary",
                "stipend",
                "funded",
                "funding",
                "scholarship",
                "fellowship",
                "€",
                "$",
                "£",
            ]
            req_keys = [
                "phd",
                "doctor",
                "master",
                "msc",
                "bachelor",
                "citizen",
                "citizenship",
                "visa",
                "english",
                "ielts",
                "toefl",
                "experience",
                "years",
                "degree",
            ]
            funding_sentence = _compact(_pick_sentence(text, funding_keys), 260)
            req_sentence = _compact(_pick_sentence(text, req_keys), 260)
            if output_language == "en":
                topic_reason = (
                    f"[Topic Connection] Keyword overlap (rule-based; LLM unavailable): {hits[:5]}"
                    if hits
                    else "[Topic Connection] No clear keyword overlap; included for manual review (LLM unavailable)."
                )
                funding_reason = (
                    f"[Funding] {funding_sentence}"
                    if funding_sentence
                    else "[Funding] Not mentioned."
                )
                req_reason = (
                    f"[Key Requirement] {req_sentence}"
                    if req_sentence
                    else "[Key Requirement] Not mentioned."
                )
            else:
                topic_reason = (
                    f"[Topic Connection] 关键词命中（规则判断，LLM 不可用）：{hits[:5]}"
                    if hits
                    else "[Topic Connection] 无明确关键词命中，仅供人工扫读（LLM 不可用）"
                )
                funding_reason = (
                    f"[Funding] {funding_sentence}" if funding_sentence else "[Funding] 未提及"
                )
                req_reason = (
                    f"[Key Requirement] {req_sentence}"
                    if req_sentence
                    else "[Key Requirement] 未提及"
                )
            reasons = [topic_reason, funding_reason, req_reason]
        else:
            # Papers: prioritize methods / data sources / metrics-results in fallback reasons.
            method_keys = [
                "we propose",
                "we present",
                "we develop",
                "model",
                "framework",
                "algorithm",
                "architecture",
                "deep learning",
                "machine learning",
                "neural",
                "cnn",
                "transformer",
                "mamba",
                "segmentation",
                "classification",
                "regression",
                "assimilation",
                "bayesian",
                "kalman",
                "ensemble",
                "monte carlo",
            ]
            data_keys = [
                "satellite",
                "sentinel",
                "landsat",
                "modis",
                "airs",
                "era5",
                "reanalysis",
                "cmip",
                "cmip6",
                "gcm",
                "dataset",
                "in-situ",
                "gauge",
                "field campaign",
            ]
            result_keys = [
                "rmse",
                "mae",
                "mse",
                "r^2",
                "r2",
                "correlation",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "miou",
                "mIoU",
                "outperforms",
                "%",
                " r >",
                " r=",
                " p <",
            ]

            method_sentence = _compact(_pick_sentence(text, method_keys), 260)
            data_sentence = _compact(_pick_sentence(text, data_keys), 260)
            result_sentence = _compact(_pick_sentence(text, result_keys), 260)
            critical_sentence = result_sentence or method_sentence or data_sentence

            missing: List[str] = []
            if not method_sentence:
                missing.append("methods")
            if not data_sentence:
                missing.append("data sources")
            if not result_sentence:
                missing.append("metrics/results")

            if output_language == "en":
                breakdown = (
                    f"[Score breakdown] Rule-based score; keyword overlap={hits[:5]} (LLM unavailable), total={score}/10"
                    if hits
                    else f"[Score breakdown] Rule-based score; no keyword overlap found (LLM unavailable), total={score}/10"
                )
                critical = (
                    f"[Critical value] {critical_sentence}"
                    if critical_sentence
                    else "[Critical value] Methods/data/results are not clearly mentioned in the provided excerpt."
                )
                limitations = (
                    f"[Limitations] Missing in excerpt: {', '.join(missing)}."
                    if missing
                    else "[Limitations] None obvious from the excerpt."
                )
            else:
                breakdown = (
                    f"[Score breakdown] 规则评分；关键词命中={hits[:5]}（LLM 不可用），总分={score}/10"
                    if hits
                    else f"[Score breakdown] 规则评分；未发现关键词命中（LLM 不可用），总分={score}/10"
                )
                critical = (
                    f"[Critical value] {critical_sentence}"
                    if critical_sentence
                    else "[Critical value] 摘要片段未清晰提及方法/数据/结果。"
                )
                limitations = (
                    f"[Limitations] 摘要片段缺失：{', '.join(missing)}。"
                    if missing
                    else "[Limitations] 从摘要片段无法判断明显局限。"
                )
            reasons = [breakdown, critical, limitations]
    rec = _decide_recommendation(score, strong_th, normal_th)
    fallback_summary = head_sentences(text, max_chars=480)
    bullets = _fallback_bullets(fallback_summary, output_language=output_language)
    return LLMResult(
        summary_bullets=bullets[:3],
        relevance_score=score,
        recommendation=rec,
        reasons=reasons[:3],
        keywords=[],
        confidence=0.2,
        llm_status=llm_status,
        raw_json="",
        llm_input_chars=llm_input_chars,
        budget_hit=budget_hit,
        llm_input_tokens=llm_input_tokens,
        llm_output_tokens=llm_output_tokens,
        llm_total_tokens=llm_total_tokens,
        llm_provider=llm_provider or "",
        llm_model=llm_model or "",
    )


class LLMClient:
    def __init__(self, cfg: Dict[str, Any], budget: Budget, *, run_id: str = "") -> None:
        self.cfg = cfg
        self.budget = budget
        self.provider = (cfg.get("provider", "openai") or "openai").lower()
        self.model = cfg.get("model", "gpt-4o-mini")
        self.temperature = cfg.get("temperature", 0)
        self.timeout_sec = cfg.get("timeout_sec", 45)
        self.max_input_chars = cfg.get("max_input_chars_per_item", 8000)
        self.on_budget_exceeded = cfg.get("on_budget_exceeded", "degrade")
        self.output_language = (cfg.get("output_language", "zh") or "zh").lower()
        self.fallback_providers = list(cfg.get("fallback_providers") or [])
        self.provider_options = cfg.get("provider_options", {}) or {}
        self.per_agent = cfg.get("per_agent", {}) or {}
        self.response_schema_cfg = cfg.get("response_schema", True)
        self.run_id = run_id

    def analyze_paper(
        self,
        *,
        profile: Dict[str, Any],
        thresholds: Dict[str, int],
        title: str,
        abstract: str,
        journal: str,
        published_at: str,
        url: str,
    ) -> LLMResult:
        agent_cfg = self._agent_cfg("papers")
        output_language = (agent_cfg.get("output_language") or self.output_language).lower()
        max_chars = int(agent_cfg.get("max_input_chars_per_item", self.max_input_chars))

        text = clean_text_for_llm(abstract or "")
        text = truncate_text(text, max_chars)
        input_len = len(text)

        include_kw = profile.get("include_keywords", [])
        exclude_kw = profile.get("exclude_keywords", [])
        strong_th = thresholds.get("strong_threshold", 8)
        normal_th = thresholds.get("normal_threshold", 5)

        if not self.budget.can_call():
            return _keyword_fallback(
                title,
                text,
                include_kw,
                exclude_kw,
                strong_th,
                normal_th,
                agent="papers",
                output_language=output_language,
                llm_input_chars=input_len,
                llm_status="budget",
                budget_hit=1,
                llm_provider="",
                llm_model="",
            )
        prompt = _build_paper_prompt(
            profile,
            thresholds,
            title,
            text,
            journal,
            published_at,
            url,
            output_language=output_language,
        )
        return self._call_llm(
            prompt,
            strong_th=strong_th,
            normal_th=normal_th,
            title=title,
            fallback_text=text,
            include_kw=include_kw,
            exclude_kw=exclude_kw,
            llm_input_chars=input_len,
            agent="papers",
            output_language=output_language,
        )

    def analyze_job(
        self,
        *,
        profile: Dict[str, Any],
        thresholds: Dict[str, int],
        title: str,
        org: str,
        location: str,
        deadline: str,
        description: str,
        url: str,
    ) -> LLMResult:
        agent_cfg = self._agent_cfg("jobs")
        output_language = (agent_cfg.get("output_language") or self.output_language).lower()
        max_chars = int(agent_cfg.get("max_input_chars_per_item", self.max_input_chars))

        text = clean_text_for_llm(description or "")
        text = truncate_text(text, max_chars)
        input_len = len(text)

        include_kw = profile.get("include_keywords", [])  # for jobs, still helpful
        exclude_kw = profile.get("exclude_keywords", [])
        strong_th = thresholds.get("strong_threshold", 8)
        normal_th = thresholds.get("normal_threshold", 5)

        if not self.budget.can_call():
            result = _keyword_fallback(
                title,
                text,
                include_kw,
                exclude_kw,
                strong_th,
                normal_th,
                agent="jobs",
                output_language=output_language,
                llm_input_chars=input_len,
                llm_status="budget",
                budget_hit=1,
                llm_provider="",
                llm_model="",
            )
            result.reasons = _normalize_job_reasons(result.reasons, output_language=output_language)
            return result
        prompt = _build_job_prompt(
            profile,
            thresholds,
            title,
            org,
            location,
            deadline,
            text,
            url,
            output_language=output_language,
        )
        result = self._call_llm(
            prompt,
            strong_th=strong_th,
            normal_th=normal_th,
            title=title,
            fallback_text=text,
            include_kw=include_kw,
            exclude_kw=exclude_kw,
            llm_input_chars=input_len,
            agent="jobs",
            output_language=output_language,
        )
        result.reasons = _normalize_job_reasons(result.reasons, output_language=output_language)
        return result

    def _agent_cfg(self, agent: str) -> Dict[str, Any]:
        base = {k: v for k, v in self.cfg.items() if k != "per_agent"}
        override = self.per_agent.get(agent, {}) or {}
        merged = dict(base)
        merged.update(override)
        return merged

    def _provider_chain(self, agent_cfg: Dict[str, Any]) -> List[str]:
        primary = (agent_cfg.get("provider") or self.provider or "openai").lower()
        fallback = agent_cfg.get("fallback_providers")
        if fallback is None:
            fallback = self.fallback_providers
        chain = [primary] + [p for p in (fallback or []) if p and p.lower() != primary]
        return [c.lower() for c in chain if c]

    def _response_schema(self, agent_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cfg = agent_cfg.get("response_schema", self.response_schema_cfg)
        if cfg is True:
            return LLM_JSON_SCHEMA
        if isinstance(cfg, dict):
            return cfg
        return None

    def _merged_provider_options(self, provider: str, agent_cfg: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        base_opts = (
            (self.provider_options.get(provider) or {})
            if isinstance(self.provider_options, dict)
            else {}
        )
        if isinstance(base_opts, dict):
            merged.update(base_opts)
        agent_opts = (
            (agent_cfg.get("provider_options") or {}).get(provider)
            if isinstance(agent_cfg.get("provider_options"), dict)
            else None
        )
        if isinstance(agent_opts, dict):
            merged.update(agent_opts)
        return merged

    def _call_llm(
        self,
        user_prompt: str,
        *,
        strong_th: int,
        normal_th: int,
        title: str,
        fallback_text: str,
        include_kw: List[str],
        exclude_kw: List[str],
        llm_input_chars: int,
        agent: str,
        output_language: str,
    ) -> LLMResult:
        agent_cfg = self._agent_cfg(agent)
        model = agent_cfg.get("model", self.model)
        temperature = float(agent_cfg.get("temperature", self.temperature))
        timeout_sec = int(agent_cfg.get("timeout_sec", self.timeout_sec))
        schema = self._response_schema(agent_cfg)
        provider_chain = self._provider_chain(agent_cfg)

        self.budget.consume_call(1)

        any_ready = False
        for provider_name in provider_chain:
            provider = get_provider(provider_name)
            if not provider:
                continue
            opts = self._merged_provider_options(provider_name, agent_cfg)
            model_for_provider = opts.get("model", model)
            if not provider.ready(opts):
                continue
            any_ready = True
            try:
                content, usage = provider.call_json(
                    model=str(model_for_provider),
                    temperature=temperature,
                    timeout_sec=timeout_sec,
                    user_prompt=user_prompt,
                    response_schema=schema,
                    cfg=opts,
                )
                content_json = content
                try:
                    obj = json.loads(content_json)
                except json.JSONDecodeError as exc:
                    extracted = _extract_first_json_object(content_json)
                    if extracted:
                        try:
                            obj = json.loads(extracted)
                            content_json = extracted
                            log_event(
                                "llm_provider_json_extracted",
                                run_id=self.run_id,
                                agent=agent,
                                provider=provider_name,
                                model=str(model_for_provider),
                                content_len=len(content or ""),
                                extracted_len=len(extracted),
                                extracted_preview=_preview(extracted),
                            )
                        except json.JSONDecodeError:
                            log_event(
                                "llm_provider_non_json",
                                run_id=self.run_id,
                                agent=agent,
                                provider=provider_name,
                                model=str(model_for_provider),
                                content_len=len(content or ""),
                                content_preview=_preview(content),
                                error=str(exc)[:200],
                            )
                            raise
                    else:
                        log_event(
                            "llm_provider_non_json",
                            run_id=self.run_id,
                            agent=agent,
                            provider=provider_name,
                            model=str(model_for_provider),
                            content_len=len(content or ""),
                            content_preview=_preview(content),
                            error=str(exc)[:200],
                        )
                        raise
                try:
                    js_validate(obj, LLM_JSON_SCHEMA)
                except ValidationError as exc:
                    log_event(
                        "llm_provider_schema_error",
                        run_id=self.run_id,
                        agent=agent,
                        provider=provider_name,
                        model=str(model_for_provider),
                        error=str(exc)[:200],
                        content_preview=_preview(content),
                    )
                    raise
                return _build_llm_result(
                    obj,
                    content_json,
                    usage,
                    llm_input_chars,
                    strong_th,
                    normal_th,
                    provider_name,
                    str(model_for_provider),
                )
            except Exception as exc:
                log_event(
                    "llm_provider_failed",
                    run_id=self.run_id,
                    agent=agent,
                    provider=provider_name,
                    model=str(model_for_provider),
                    error_type=type(exc).__name__,
                    error=str(exc)[:300],
                )
                # retry with stricter instruction + lower temperature
                try:
                    content, usage = provider.call_json(
                        model=str(model_for_provider),
                        temperature=0.0,
                        timeout_sec=timeout_sec,
                        user_prompt="You must output JSON only, with no extra text.\n\n"
                        + user_prompt,
                        response_schema=schema,
                        cfg=opts,
                    )
                    content_json = content
                    try:
                        obj = json.loads(content_json)
                    except json.JSONDecodeError as exc:
                        extracted = _extract_first_json_object(content_json)
                        if extracted:
                            try:
                                obj = json.loads(extracted)
                                content_json = extracted
                                log_event(
                                    "llm_provider_json_extracted",
                                    run_id=self.run_id,
                                    agent=agent,
                                    provider=provider_name,
                                    model=str(model_for_provider),
                                    content_len=len(content or ""),
                                    extracted_len=len(extracted),
                                    extracted_preview=_preview(extracted),
                                )
                            except json.JSONDecodeError:
                                log_event(
                                    "llm_provider_non_json",
                                    run_id=self.run_id,
                                    agent=agent,
                                    provider=provider_name,
                                    model=str(model_for_provider),
                                    content_len=len(content or ""),
                                    content_preview=_preview(content),
                                    error=str(exc)[:200],
                                )
                                raise
                        else:
                            log_event(
                                "llm_provider_non_json",
                                run_id=self.run_id,
                                agent=agent,
                                provider=provider_name,
                                model=str(model_for_provider),
                                content_len=len(content or ""),
                                content_preview=_preview(content),
                                error=str(exc)[:200],
                            )
                            raise
                    try:
                        js_validate(obj, LLM_JSON_SCHEMA)
                    except ValidationError as exc:
                        log_event(
                            "llm_provider_schema_error",
                            run_id=self.run_id,
                            agent=agent,
                            provider=provider_name,
                            model=str(model_for_provider),
                            error=str(exc)[:200],
                            content_preview=_preview(content),
                        )
                        raise
                    return _build_llm_result(
                        obj,
                        content_json,
                        usage,
                        llm_input_chars,
                        strong_th,
                        normal_th,
                        provider_name,
                        str(model_for_provider),
                    )
                except Exception as exc:
                    log_event(
                        "llm_provider_failed",
                        run_id=self.run_id,
                        agent=agent,
                        provider=provider_name,
                        model=str(model_for_provider),
                        error_type=type(exc).__name__,
                        error=str(exc)[:300],
                    )
                    continue

        status = "no_provider" if not any_ready else "failed"
        return _keyword_fallback(
            title,
            fallback_text,
            include_kw,
            exclude_kw,
            strong_th,
            normal_th,
            agent=agent,
            output_language=output_language,
            llm_input_chars=llm_input_chars,
            llm_status=status,
            budget_hit=0,
            llm_provider="",
            llm_model="",
        )


def _build_paper_prompt(
    profile: Dict[str, Any],
    thresholds: Dict[str, int],
    title: str,
    abstract: str,
    journal: str,
    published_at: str,
    url: str,
    *,
    output_language: str = "zh",
) -> str:
    strong_th = thresholds.get("strong_threshold", 8)
    normal_th = thresholds.get("normal_threshold", 5)
    lang = "English" if output_language == "en" else "Chinese"
    return f"""You are a senior academic journal editor and reviewer. Use an explainable evaluation and output JSON only (no extra text).

Field: {profile.get('field')}
Focus: {profile.get('focus')}
Include keywords: {profile.get('include_keywords')}
Exclude keywords: {profile.get('exclude_keywords')}

Paper:
Title: {title}
Abstract: {abstract}
Journal: {journal}
Published: {published_at}
URL: {url}

Scoring criteria (total 0–10, integer):
Evaluate relevance based strictly on the user's Field, Focus, and Keywords.
A Topic Match (0-3): Alignment with user's specific research interests.
B Technical Depth (0-3): Does the abstract reveal specific methods/architectures/data (score high) vs. generic high-level descriptions (score low)?
C Utility & Novelty (0-2): Does this offer actionable tools, datasets, or insights for the user's work? Is it a significant advancement?
D Evidence Strength (0-2): Are quantitative results or specific performance metrics provided? (Penalize vague claims like "better performance").
Total = A+B+C+D

Output requirements:
1) summary_bullets: 1–3 bullets in {lang} (objective, no hype)
2) relevance_score: total score (0–10 integer)
3) recommendation:
   - strong: score >= {strong_th}
   - normal: score >= {normal_th}
   - skip: otherwise
4) reasons: exactly 3 items, each must start with the prefixes below (in {lang}):
   - [Score breakdown] A=Topic Match (x/3), B=Technical Depth (x/3), C=Utility & Novelty (x/2), D=Evidence Strength (x/2), total=?
   - [Critical value] What specific technical insight, method, or resource does this offer to the user?
   - [Limitations] Note missing metrics, vague methods, or potential applicability issues.
5) confidence: 0–1 (based on evidence sufficiency and match)
"""


def _build_job_prompt(
    profile: Dict[str, Any],
    thresholds: Dict[str, int],
    title: str,
    org: str,
    location: str,
    deadline: str,
    description: str,
    url: str,
    *,
    output_language: str = "zh",
) -> str:
    strong_th = thresholds.get("strong_threshold", 8)
    normal_th = thresholds.get("normal_threshold", 5)
    lang = "English" if output_language == "en" else "Chinese"
    return f"""You are a helpful academic assistant scouting for PhD opportunities. Output JSON only.

User Profile:
Field: {profile.get('field')}
Focus: {profile.get('focus')}
Keywords: {profile.get('include_keywords')}

Job Details:
Title: {title}
Org: {org}
Loc: {location}
Deadline: {deadline}
URL: {url}
Description excerpt:
{description[:4000]}

Evaluation Logic:
**Assume this IS a PhD position (pre-filtered).**
Your sole task is to evaluate the **Research Topic Alignment** with the user's profile.

Scoring (0-10) - Research Fit:
- **High (8-10)**: The research topic matches the user's specific 'Focus' or 'Keywords' very closely.
- **Medium (5-7)**: The research is in the same general 'Field' but not a precise match for the user's 'Focus'.
- **Low (0-4)**: The research topic is unrelated to the user's interests (e.g. wrong discipline).

Output requirements:
1) summary_bullets: 1–2 bullets in {lang} summarizing the research topic.
2) relevance_score: Integer 0-10 based on research fit.
3) recommendation:
   - strong: score >= {strong_th}
   - normal: score >= {normal_th}
   - skip: otherwise
4) reasons: exactly 3 items in {lang}:
   - [Topic Connection] Specifically analyze how this research connects to the user's 'Focus'.
   - [Funding] Extract salary, stipend, or "fully funded" info (or state "Not mentioned").
   - [Key Requirement] Extract critical constraints (e.g. citizenship, specific degree) or major benefits.
5) confidence: 0–1
"""


def _normalize_job_reasons(reasons: List[str], *, output_language: str) -> List[str]:
    tags = ["[Topic Connection]", "[Funding]", "[Key Requirement]"]
    if not isinstance(reasons, list):
        reasons = []
    out: List[str] = []
    for idx, tag in enumerate(tags):
        raw = reasons[idx].strip() if idx < len(reasons) and isinstance(reasons[idx], str) else ""
        if raw.startswith("["):
            out.append(raw)
            continue
        if raw:
            out.append(f"{tag} {raw}")
        else:
            fallback = "Not mentioned." if (output_language or "zh").lower() == "en" else "未提及"
            out.append(f"{tag} {fallback}")
    return out
