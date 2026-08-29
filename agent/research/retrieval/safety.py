"""Layered, fail-closed handling for untrusted Research retrieval content."""
from __future__ import annotations

import base64
import binascii
import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import unquote

from agent.config import FORBIDDEN_PAYLOAD_KEYS, TEST_METRICS_SENTINEL


class SafetyRejectionReason(str, Enum):
    HIDDEN_TEST_LEAKAGE = "hidden_test_leakage"
    PROMPT_INJECTION = "prompt_injection"
    INVALID_EXTERNAL_PAYLOAD = "invalid_external_payload"
    PAYLOAD_SIZE_LIMIT = "payload_size_limit"
    PAYLOAD_DEPTH_LIMIT = "payload_depth_limit"
    CYCLIC_PAYLOAD = "cyclic_payload"


class ResearchSafetyError(ValueError):
    """Typed rejection raised before untrusted retrieval data can be used."""

    def __init__(self, reason: SafetyRejectionReason, origin: str, detail: str):
        self.reason = reason
        self.origin = origin
        self.detail = detail
        super().__init__(f"{origin}: {detail} [{reason.value}]")


def _decode_bytes(value: bytes) -> str:
    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")
    for encoding in encodings:
        try:
            decoded = value.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding.startswith("utf-16") and decoded:
            printable = sum(character.isprintable() or character.isspace() for character in decoded)
            if printable / len(decoded) < 0.9:
                continue
        return decoded
    return value.decode("latin-1")


_BASE64_ALPHABET = re.compile(r"[A-Za-z0-9+/_-]+={0,2}")


def _looks_like_base64_token(text: str) -> bool:
    compact = "".join(text.split())
    if len(compact) < 8 or _BASE64_ALPHABET.fullmatch(compact) is None:
        return False
    core = compact.rstrip("=")
    return not (any(character in core for character in "+/") and any(
        character in core for character in "-_"
    ))


def _readable_encoded_text(value: bytes) -> str | None:
    try:
        decoded = value.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    if len(decoded) < 8:
        return None
    printable = sum(character.isprintable() or character.isspace() for character in decoded)
    if printable / len(decoded) < 0.95:
        return None
    # This prevents ordinary all-alphanumeric IDs from being mistaken for
    # encoded prose while still accepting common encoded phrases/keys.
    looks_like_nested_encoding = (
        _looks_like_base64_token(decoded)
    ) or (
        len(decoded) >= 16
        and len(decoded) % 2 == 0
        and re.fullmatch(r"[0-9a-fA-F]+", decoded) is not None
    )
    if not looks_like_nested_encoding and not any(
        character.isspace() or character in "_-:@/+= "
        for character in decoded
    ):
        return None
    return decoded


def _decode_whole_base64(
    text: str,
    *,
    origin: str,
    declared: bool,
    chain_established: bool = False,
) -> str | None:
    if not declared and any(character.isspace() for character in text):
        return None
    compact = "".join(text.split()) if declared else text
    if not compact or _BASE64_ALPHABET.fullmatch(compact) is None:
        if declared:
            raise ResearchSafetyError(
                SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
                origin,
                "declared nested base64 text is malformed or ambiguous",
            )
        return None

    unpadded = compact.rstrip("=")
    existing_padding = compact[len(unpadded):]
    has_standard_symbols = any(character in unpadded for character in "+/")
    has_url_symbols = any(character in unpadded for character in "-_")
    strongly_encoded = (
        declared
        or chain_established
        or bool(existing_padding)
    )
    if has_standard_symbols and has_url_symbols:
        if strongly_encoded:
            raise ResearchSafetyError(
                SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
                origin,
                "base64 text mixes standard and URL-safe alphabets",
            )
        return None

    remainder = len(unpadded) % 4
    if remainder == 1:
        if strongly_encoded:
            raise ResearchSafetyError(
                SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
                origin,
                "base64 text has invalid length remainder 1",
            )
        return None

    required_padding = {0: "", 2: "==", 3: "="}[remainder]
    if existing_padding and existing_padding != required_padding:
        raise ResearchSafetyError(
            SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
            origin,
            "base64 text has inconsistent padding",
        )
    padded = unpadded + required_padding
    try:
        raw = base64.b64decode(
            padded,
            altchars=b"-_" if has_url_symbols else None,
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        if declared or strongly_encoded:
            raise ResearchSafetyError(
                SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
                origin,
                "base64 text is invalid or ambiguous",
            ) from exc
        return None

    if declared:
        try:
            decoded = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ResearchSafetyError(
                SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
                origin,
                "declared nested base64 is not UTF-8 text",
            ) from exc
        printable = sum(character.isprintable() or character.isspace() for character in decoded)
        if decoded and printable / len(decoded) >= 0.95:
            return decoded
        raise ResearchSafetyError(
            SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
            origin,
            "declared nested base64 does not decode to readable text",
        )
    return _readable_encoded_text(raw)


def _decode_unlabeled_text(
    text: str,
    *,
    origin: str,
    chain_established: bool,
) -> str:
    compact = "".join(text.split())
    if len(compact) >= 8:
        decoded = _decode_whole_base64(
            text,
            origin=origin,
            declared=False,
            chain_established=chain_established,
        )
        if decoded is not None:
            return decoded
    if len(compact) >= 16 and len(compact) % 2 == 0 and re.fullmatch(r"[0-9a-fA-F]+", compact):
        decoded = _readable_encoded_text(bytes.fromhex(compact))
        if decoded is not None:
            return decoded
    return text


def _normalize_unicode(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )
    return " ".join(normalized.split())


def _decode_declared_base64(text: str, *, origin: str) -> str:
    stripped = text.strip()
    prefix = next(
        (
            candidate
            for candidate in ("base64:", "data:text/plain;base64,")
            if stripped.casefold().startswith(candidate)
        ),
        None,
    )
    if prefix is None:
        return text
    payload = stripped[len(prefix):]
    decoded = _decode_whole_base64(payload, origin=origin, declared=True)
    if decoded is None:  # pragma: no cover - declared decoding fails closed above
        raise ResearchSafetyError(
            SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
            origin,
            "declared nested base64 text is invalid",
        )
    return decoded


def _normalization_round(
    text: str,
    *,
    origin: str,
    chain_established: bool,
) -> tuple[str, tuple[int, ...], bool]:
    """Apply one ordered round and return all intermediate decoded sizes."""
    sizes: list[int] = []

    def changed(previous: str, next_text: str) -> str:
        if next_text != previous:
            sizes.append(len(next_text))
        return next_text

    decoded_this_round = False
    current = changed(text, _normalize_unicode(text))
    url_decoded = unquote(current)
    decoded_this_round = decoded_this_round or url_decoded != current
    current = changed(current, url_decoded)
    html_decoded = html.unescape(current)
    decoded_this_round = decoded_this_round or html_decoded != current
    current = changed(current, html_decoded)
    if re.search(r"\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}", current):
        escaped = re.sub(
            r"\\u([0-9a-fA-F]{4})|\\x([0-9a-fA-F]{2})",
            lambda match: chr(int(match.group(1) or match.group(2), 16)),
            current,
        )
        decoded_this_round = decoded_this_round or escaped != current
        current = changed(current, escaped)
    declared = _decode_declared_base64(current, origin=origin)
    decoded_this_round = decoded_this_round or declared != current
    decoded = declared if declared != current else _decode_unlabeled_text(
        current,
        origin=origin,
        chain_established=chain_established or decoded_this_round,
    )
    decoded_this_round = decoded_this_round or decoded != current
    current = changed(current, decoded)
    current = changed(current, _normalize_unicode(current))
    return current, tuple(sizes), chain_established or decoded_this_round


def _iter_normalized_text(
    value: str | bytes,
    *,
    origin: str,
    max_decode_rounds: int,
    max_decoded_chars: int,
) -> Iterator[str]:
    if isinstance(value, bytes):
        current = _decode_bytes(value)
    elif isinstance(value, str):
        current = value
    else:
        raise ResearchSafetyError(
            SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
            origin,
            "external content must be text or bytes",
        )

    cumulative_chars = len(current)
    if cumulative_chars > max_decoded_chars:
        raise ResearchSafetyError(
            SafetyRejectionReason.PAYLOAD_SIZE_LIMIT,
            origin,
            f"cumulative decoded text exceeds {max_decoded_chars} characters",
        )

    chain_established = False
    for _ in range(max_decode_rounds):
        normalized, stage_sizes, chain_established = _normalization_round(
            current,
            origin=origin,
            chain_established=chain_established,
        )
        cumulative_chars += sum(stage_sizes)
        if cumulative_chars > max_decoded_chars:
            raise ResearchSafetyError(
                SafetyRejectionReason.PAYLOAD_SIZE_LIMIT,
                origin,
                f"cumulative decoded text exceeds {max_decoded_chars} characters",
            )
        yield normalized
        if normalized == current:
            return
        current = normalized

    probe, _, _ = _normalization_round(
        current,
        origin=origin,
        chain_established=chain_established,
    )
    if probe != current:
        raise ResearchSafetyError(
            SafetyRejectionReason.PAYLOAD_DEPTH_LIMIT,
            origin,
            f"external text exceeds maximum decoding depth {max_decode_rounds}",
        )


def normalize_external_text(
    value: str | bytes,
    *,
    origin: str,
    max_decode_rounds: int = 4,
    max_decoded_chars: int = 512_000,
) -> str:
    """Return the bounded fixed point of iterative external-text decoding."""
    normalized = ""
    for normalized in _iter_normalized_text(
        value,
        origin=origin,
        max_decode_rounds=max_decode_rounds,
        max_decoded_chars=max_decoded_chars,
    ):
        pass
    return normalized


_CONFUSABLE_ASCII = str.maketrans({
    # Common Cyrillic/Greek lookalikes sufficient for deterministic leakage
    # detection; this is intentionally not a general transliterator.
    "а": "a", "е": "e", "і": "i", "ј": "j", "о": "o", "р": "p",
    "с": "c", "ѕ": "s", "т": "t", "х": "x", "у": "y", "һ": "h",
    "Α": "a", "α": "a", "Ε": "e", "ε": "e", "Ι": "i", "ι": "i",
    "Κ": "k", "κ": "k", "Μ": "m", "μ": "m", "Ν": "n", "ν": "v",
    "Ο": "o", "ο": "o", "Ρ": "p", "ρ": "p", "Τ": "t", "τ": "t",
    "Χ": "x", "χ": "x",
})


def _detection_view(text: str) -> str:
    return text.casefold().translate(_CONFUSABLE_ASCII)


_LEAKAGE_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    re.escape(TEST_METRICS_SENTINEL),
    r"\bhidden\W*test\b",
    r"\bprivate\W*(?:test|leaderboard)\b",
    r"\btest\W*(?:primary|gauc|ndcg\W*5|metrics?)\b",
    r"\b(?:primary|gauc|ndcg\W*5)\W*test\b",
    r"\btest\W*(?:split|set)\W*(?:score|metric|result)s?\b",
))

_PROMPT_INJECTION_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\b(?:ignore|disregard|forget)\b.{0,50}\b(?:previous|prior|system|developer)\b.{0,30}\binstructions?\b",
    r"\boverride\b.{0,35}\b(?:instructions?|prompt|policy)\b",
    r"\b(?:reveal|print|expose|leak|exfiltrate)\b.{0,50}\b(?:system prompt|developer message|secret|api key)\b",
    r"\b(?:act as|you are now)\b.{0,40}\b(?:assistant|system|developer|language model)\b",
    r"<\|\s*(?:system|assistant|user)\s*\|>",
    r"\[/?INST\]",
    r"#{2,}\s*(?:instruction|developer)\s*:\s*(?:ignore|override|disregard)\b",
))


_VALIDATED_EVIDENCE_TOKEN = object()


@dataclass(frozen=True, init=False)
class InertEvidenceText:
    """Scanner-validated external text; direct construction is rejected."""

    origin: str
    text: str
    content_id: str

    def __init__(
        self,
        origin: str,
        text: str,
        content_id: str,
        *,
        _validation_token: object | None = None,
    ) -> None:
        if _validation_token is not _VALIDATED_EVIDENCE_TOKEN:
            raise ResearchSafetyError(
                SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
                origin if isinstance(origin, str) else "external Research evidence",
                "validated external evidence must be created by prepare_external_text()",
            )
        expected_content_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_id != expected_content_id:
            raise ResearchSafetyError(
                SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
                origin,
                "validated external evidence content_id does not match its text",
            )
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "content_id", content_id)

    def to_prompt_block(self) -> str:
        """Keep evidence explicitly delimited and JSON-escaped in later prompts."""
        payload = json.dumps(
            {"content_id": self.content_id, "origin": self.origin, "text": self.text},
            ensure_ascii=False,
            sort_keys=True,
        )
        return (
            "BEGIN_EXTERNAL_RESEARCH_EVIDENCE_DATA\n"
            f"{payload}\n"
            "END_EXTERNAL_RESEARCH_EVIDENCE_DATA"
        )


@dataclass
class _ScanState:
    nodes: int = 0
    total_chars: int = 0


class ResearchSafetyScanner:
    """Normalize, classify, and reject unsafe external Research content."""

    def __init__(
        self,
        *,
        max_text_chars: int = 128_000,
        max_total_chars: int = 256_000,
        max_depth: int = 24,
        max_nodes: int = 10_000,
        max_decode_rounds: int = 4,
        max_decoded_chars: int = 512_000,
    ) -> None:
        limits = (
            max_text_chars,
            max_total_chars,
            max_depth,
            max_nodes,
            max_decode_rounds,
            max_decoded_chars,
        )
        if any(not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 for limit in limits):
            raise ValueError("Research safety limits must be positive integers")
        self.max_text_chars = max_text_chars
        self.max_total_chars = max_total_chars
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_decode_rounds = max_decode_rounds
        self.max_decoded_chars = max_decoded_chars

    def scan_text(self, text: str | bytes, *, origin: str = "external Research evidence") -> str:
        if not isinstance(text, (str, bytes)):
            raise ResearchSafetyError(
                SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
                origin,
                "external content must be text or bytes",
            )
        if len(text) > self.max_text_chars:
            raise ResearchSafetyError(
                SafetyRejectionReason.PAYLOAD_SIZE_LIMIT,
                origin,
                f"external text exceeds {self.max_text_chars} characters/bytes",
            )
        normalized = ""
        for normalized in _iter_normalized_text(
            text,
            origin=origin,
            max_decode_rounds=self.max_decode_rounds,
            max_decoded_chars=self.max_decoded_chars,
        ):
            if len(normalized) > self.max_text_chars:
                raise ResearchSafetyError(
                    SafetyRejectionReason.PAYLOAD_SIZE_LIMIT,
                    origin,
                    f"decoded external text exceeds {self.max_text_chars} characters",
                )
            self._assert_safe_normalized(normalized, origin=origin)
        return normalized

    @staticmethod
    def _assert_safe_normalized(normalized: str, *, origin: str) -> None:
        detection_text = _detection_view(normalized)
        compact = re.sub(r"[^a-z0-9]+", "", detection_text)
        forbidden_compact = {
            re.sub(r"[^a-z0-9]+", "", _detection_view(key))
            for key in FORBIDDEN_PAYLOAD_KEYS
        }
        forbidden_compact.add(
            re.sub(r"[^a-z0-9]+", "", _detection_view(TEST_METRICS_SENTINEL))
        )
        if any(key and key in compact for key in forbidden_compact):
            raise ResearchSafetyError(
                SafetyRejectionReason.HIDDEN_TEST_LEAKAGE,
                origin,
                "contains a forbidden hidden-test key",
            )
        compact_leakage_markers = (
            "hiddentest", "privatetest", "privateleaderboard", "testprimary",
            "testgauc", "testndcg5", "testmetrics", "primarytest", "gauctest",
            "ndcg5test",
        )
        if any(marker in compact for marker in compact_leakage_markers) or any(
            pattern.search(detection_text) for pattern in _LEAKAGE_PATTERNS
        ):
            raise ResearchSafetyError(
                SafetyRejectionReason.HIDDEN_TEST_LEAKAGE,
                origin,
                "contains hidden-test metrics or development results",
            )
        if any(pattern.search(detection_text) for pattern in _PROMPT_INJECTION_PATTERNS):
            raise ResearchSafetyError(
                SafetyRejectionReason.PROMPT_INJECTION,
                origin,
                "contains prompt-injection-like instructions",
            )

    def prepare_external_text(
        self,
        text: str | bytes,
        *,
        origin: str = "external Research evidence",
    ) -> InertEvidenceText:
        normalized_origin = self.scan_text(origin, origin="external evidence origin")
        normalized = self.scan_text(text, origin=normalized_origin)
        content_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return InertEvidenceText(
            origin=normalized_origin,
            text=normalized,
            content_id=content_id,
            _validation_token=_VALIDATED_EVIDENCE_TOKEN,
        )

    def scan_value(self, value: Any, *, origin: str = "external Research evidence") -> None:
        self._scan_value(value, origin=origin, depth=0, active=set(), state=_ScanState())

    def _scan_value(
        self,
        value: Any,
        *,
        origin: str,
        depth: int,
        active: set[int],
        state: _ScanState,
    ) -> None:
        if depth > self.max_depth:
            raise ResearchSafetyError(
                SafetyRejectionReason.PAYLOAD_DEPTH_LIMIT,
                origin,
                f"external payload exceeds maximum depth {self.max_depth}",
            )
        state.nodes += 1
        if state.nodes > self.max_nodes:
            raise ResearchSafetyError(
                SafetyRejectionReason.PAYLOAD_SIZE_LIMIT,
                origin,
                f"external payload exceeds maximum node count {self.max_nodes}",
            )

        if isinstance(value, Enum):
            self._scan_value(
                value.value, origin=origin, depth=depth + 1, active=active, state=state
            )
            return
        if isinstance(value, Path):
            self._scan_scalar_text(str(value), origin=origin, state=state)
            return
        if isinstance(value, (str, bytes)):
            self._scan_scalar_text(value, origin=origin, state=state)
            return
        if value is None or isinstance(value, (int, float, bool)):
            return

        track_cycle = is_dataclass(value) or isinstance(value, Mapping) or (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        )
        identity = id(value)
        if track_cycle and identity in active:
            raise ResearchSafetyError(
                SafetyRejectionReason.CYCLIC_PAYLOAD,
                origin,
                "external payload contains a reference cycle",
            )
        if track_cycle:
            active.add(identity)
        try:
            if is_dataclass(value):
                for item in fields(value):
                    self._scan_scalar_text(item.name, origin=f"{origin} field", state=state)
                    self._scan_value(
                        getattr(value, item.name),
                        origin=f"{origin}.{item.name}",
                        depth=depth + 1,
                        active=active,
                        state=state,
                    )
                return
            if isinstance(value, Mapping):
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise ResearchSafetyError(
                            SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
                            origin,
                            "contains a non-string key",
                        )
                    self._scan_scalar_text(key, origin=f"{origin} key", state=state)
                    self._scan_value(
                        child,
                        origin=f"{origin}.{key}",
                        depth=depth + 1,
                        active=active,
                        state=state,
                    )
                return
            if isinstance(value, Sequence):
                for index, child in enumerate(value):
                    self._scan_value(
                        child,
                        origin=f"{origin}[{index}]",
                        depth=depth + 1,
                        active=active,
                        state=state,
                    )
                return
        finally:
            if track_cycle:
                active.remove(identity)

        raise ResearchSafetyError(
            SafetyRejectionReason.INVALID_EXTERNAL_PAYLOAD,
            origin,
            f"contains unsupported value {type(value).__name__}",
        )

    def _scan_scalar_text(self, value: str | bytes, *, origin: str, state: _ScanState) -> None:
        normalized = self.scan_text(value, origin=origin)
        state.total_chars += len(normalized)
        if state.total_chars > self.max_total_chars:
            raise ResearchSafetyError(
                SafetyRejectionReason.PAYLOAD_SIZE_LIMIT,
                origin,
                f"external payload exceeds {self.max_total_chars} normalized characters",
            )


DEFAULT_RESEARCH_SAFETY_SCANNER = ResearchSafetyScanner()


def assert_safe_external_evidence(value: Any, *, origin: str = "external Research evidence") -> None:
    DEFAULT_RESEARCH_SAFETY_SCANNER.scan_value(value, origin=origin)
