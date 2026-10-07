"""
Turns untrusted email bodies into a self-contained document for a sandboxed iframe.

Layers, each sufficient on its own to stop script injection:
1. nh3 sanitization removes scripts, event handlers and dangerous URLs.
2. A CSP inside the frame only allows the one nonce'd resize script to run.
3. The iframe is sandboxed without allow-same-origin, so even script that did
   run would have an opaque origin with no access to the app.

Remote resources (tracking pixels, web fonts) are blocked by the CSP unless
the viewer opts in per message.
"""

import base64
import html
import re
import secrets
from typing import Dict, Optional, Tuple
from urllib.parse import unquote

import nh3

_EXTRA_TAGS = {"style", "font", "center", "big", "small", "u", "s", "strike", "tt"}
_PRESENTATION_ATTRIBUTES = {
    "style", "class", "align", "valign", "width", "height", "bgcolor", "color",
    "border", "cellpadding", "cellspacing", "dir", "face", "size", "background",
}
_ALLOWED_TAGS = set(nh3.ALLOWED_TAGS) | _EXTRA_TAGS
_ALLOWED_ATTRIBUTES: Dict[str, set] = {
    tag: set(attrs) for tag, attrs in nh3.ALLOWED_ATTRIBUTES.items()
}
_ALLOWED_ATTRIBUTES["*"] = _ALLOWED_ATTRIBUTES.get("*", set()) | _PRESENTATION_ATTRIBUTES
_ALLOWED_ATTRIBUTES.setdefault("img", set()).update({"src", "alt", "title"})
_ALLOWED_ATTRIBUTES.setdefault("a", set()).update({"href", "title"})
_URL_SCHEMES = set(nh3.ALLOWED_URL_SCHEMES) | {"data", "cid"}

_REMOTE_REF_RE = re.compile(
    r"""(?:\b(?:src|background|srcset)\s*=\s*["']?\s*|url\(\s*["']?\s*)(?:https?:)?//""",
    re.IGNORECASE,
)
_CID_RE = re.compile(r"""cid:([^"'\s)>]+)""", re.IGNORECASE)
_URL_IN_TEXT_RE = re.compile(r"https?://[^\s<>\"]+")

MAX_INLINE_IMAGE_BYTES = 10 * 1024 * 1024

_FRAME_STYLE = """
html { height: auto; }
body { margin: 0; padding: 16px; font-family: system-ui, -apple-system, sans-serif;
       color: #111827; overflow-wrap: break-word; }
img { max-width: 100%; height: auto; }
pre.plain-text { white-space: pre-wrap; font-family: inherit; margin: 0; }
"""

# Reports the document height so the parent can size the iframe without
# needing same-origin access.
_RESIZE_SCRIPT = """
(function () {
  var last = 0;
  function report() {
    var height = Math.ceil(document.documentElement.getBoundingClientRect().height);
    if (height !== last) {
      last = height;
      parent.postMessage({ type: "mbox-frame-height", height: height }, "*");
    }
  }
  new ResizeObserver(report).observe(document.documentElement);
  window.addEventListener("load", report);
})();
"""


def sanitize_html(content: str) -> str:
    return nh3.clean(
        content,
        tags=_ALLOWED_TAGS,
        clean_content_tags={"script"},
        attributes=_ALLOWED_ATTRIBUTES,
        url_schemes=_URL_SCHEMES,
        set_tag_attribute_values={"a": {"target": "_blank"}},
    )


def _linkify(escaped_text: str) -> str:
    def replace(match: "re.Match[str]") -> str:
        url = match.group(0)
        trailing = ""
        # Trailing punctuation and an escaped ">" (from "<http://...>") are not part of the URL
        while True:
            if url.endswith("&gt;"):
                url, trailing = url[:-4], "&gt;" + trailing
            elif url and url[-1] in ".,;:!?)]'":
                url, trailing = url[:-1], url[-1] + trailing
            else:
                break
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>{trailing}'

    return _URL_IN_TEXT_RE.sub(replace, escaped_text)


def plain_text_to_html(text: str) -> str:
    return f'<pre class="plain-text">{_linkify(html.escape(text))}</pre>'


def inline_cid_images(content: str, inline_images: Dict[str, Tuple[str, bytes]]) -> str:
    """Replace cid: references with data: URIs so inline images render offline."""
    if not inline_images:
        return content
    budget = [MAX_INLINE_IMAGE_BYTES]

    def replace(match: "re.Match[str]") -> str:
        cid = unquote(match.group(1)).strip("<>")
        image = inline_images.get(cid)
        if image is None or len(image[1]) > budget[0]:
            return match.group(0)
        budget[0] -= len(image[1])
        mime_type, data = image
        return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"

    return _CID_RE.sub(replace, content)


def has_remote_content(content: str) -> bool:
    return _REMOTE_REF_RE.search(content) is not None


def _content_security_policy(nonce: str, allow_remote: bool) -> str:
    remote = " https: http:" if allow_remote else ""
    return (
        "default-src 'none'; "
        f"script-src 'nonce-{nonce}'; "
        f"style-src 'unsafe-inline'{remote}; "
        f"img-src data:{remote}; "
        f"font-src data:{remote}; "
        f"media-src data:{remote}; "
        "form-action 'none'"
    )


def render_email_frame(
    body: Optional[str],
    body_type: str,
    inline_images: Optional[Dict[str, Tuple[str, bytes]]] = None,
    allow_remote: bool = False,
) -> Tuple[str, bool]:
    """
    Build the srcdoc document for one email.

    Returns (document_html, has_remote_content). The caller is responsible for
    attribute-escaping the document when placing it in ``srcdoc``.
    """
    if not body:
        content = '<p style="color:#6b7280">This message has no displayable content.</p>'
    elif body_type == "HTML":
        content = sanitize_html(inline_cid_images(body, inline_images or {}))
    else:
        content = plain_text_to_html(body)

    remote = has_remote_content(content)
    nonce = secrets.token_urlsafe(16)
    document = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f'<meta http-equiv="Content-Security-Policy" content="{_content_security_policy(nonce, allow_remote)}">'
        '<meta name="referrer" content="no-referrer">'
        f"<style>{_FRAME_STYLE}</style></head><body>"
        f"{content}"
        f'<script nonce="{nonce}">{_RESIZE_SCRIPT}</script>'
        "</body></html>"
    )
    return document, remote
