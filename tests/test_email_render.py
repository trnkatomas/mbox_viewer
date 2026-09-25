"""Tests for turning untrusted email bodies into sandboxed iframe documents."""

import re

from email_render import (
    has_remote_content,
    inline_cid_images,
    plain_text_to_html,
    render_email_frame,
    sanitize_html,
)
from email_service import parse_search_query


class TestSanitizeHtml:
    def test_removes_script_and_handlers(self):
        out = sanitize_html('<p onclick="x()">hi</p><script>alert(1)</script><img src=x onerror=alert(1)>')
        assert "script" not in out and "onclick" not in out and "onerror" not in out
        assert "hi" in out

    def test_removes_javascript_urls(self):
        assert "javascript" not in sanitize_html('<a href="javascript:alert(1)">x</a>')

    def test_keeps_email_styling(self):
        out = sanitize_html('<style>p{color:red}</style><table bgcolor="#fff"><tr><td style="padding:4px">x</td></tr></table>')
        assert "<style>p{color:red}</style>" in out
        assert 'style="padding:4px"' in out
        assert 'bgcolor="#fff"' in out

    def test_links_open_in_new_tab_without_opener(self):
        out = sanitize_html('<a href="https://example.com">x</a>')
        assert 'target="_blank"' in out and "noopener" in out

    def test_style_breakout_cannot_inject_script(self):
        out = sanitize_html("<style></style><script>alert(1)</script></style>")
        assert "<script" not in out


class TestFrameDocument:
    def test_only_nonced_script_is_allowed(self):
        doc, _ = render_email_frame("<p>hi</p>", "HTML")
        nonce = re.search(r"script-src 'nonce-([^']+)'", doc).group(1)
        assert f'<script nonce="{nonce}">' in doc
        assert doc.count("<script") == 1

    def test_nonce_differs_per_render(self):
        first, _ = render_email_frame("<p>hi</p>", "HTML")
        second, _ = render_email_frame("<p>hi</p>", "HTML")
        nonce = re.compile(r"nonce-([^']+)")
        assert nonce.search(first).group(1) != nonce.search(second).group(1)

    def test_remote_content_detected_and_blocked(self):
        doc, remote = render_email_frame('<img src="https://t.example/pixel.gif">', "HTML")
        assert remote
        assert "img-src data:;" in doc

    def test_plain_text_is_escaped_and_linkified(self):
        doc, _ = render_email_frame("a < b <https://example.com/x?a=1&b=2>.", "Plain Text")
        assert "a &lt; b" in doc
        assert 'href="https://example.com/x?a=1&amp;b=2"' in doc
        assert "&gt;.</pre>" in doc

    def test_empty_body(self):
        doc, remote = render_email_frame(None, "None")
        assert "no displayable content" in doc and not remote


class TestInlineImages:
    def test_cid_replaced_with_data_uri(self):
        out = inline_cid_images('<img src="cid:logo@x">', {"logo@x": ("image/png", b"\x89PNG")})
        assert 'src="data:image/png;base64,iVBORw=="' in out

    def test_unknown_cid_left_alone(self):
        assert inline_cid_images('<img src="cid:nope">', {"x": ("image/png", b"1")}) == '<img src="cid:nope">'


class TestRemoteDetection:
    def test_css_url(self):
        assert has_remote_content('<div style="background:url(https://x.example/a.png)">')

    def test_links_are_not_remote_content(self):
        assert not has_remote_content('<a href="https://example.com">x</a>')

    def test_plain_text_to_html_wraps(self):
        assert plain_text_to_html("x").startswith('<pre class="plain-text">')


class TestParseSearchQuery:
    def test_quoted_values_are_not_greedy(self):
        parsed = parse_search_query('subject:"team meeting" from:"Alice Smith" notes')
        assert parsed == {"subject": "team meeting", "from": "Alice Smith", "excerpt": "notes"}

    def test_rag_takes_rest_of_phrase(self):
        parsed = parse_search_query("from:alice rag:flight booking to prague")
        assert parsed["rag"] == "flight booking to prague"
        assert parsed["from"] == "alice"
        assert parsed["excerpt"] == ""

    def test_rag_stops_at_next_filter(self):
        parsed = parse_search_query("rag:flight booking from_date:2024-01-01")
        assert parsed["rag"] == "flight booking"
        assert parsed["from_date"] == "2024-01-01"

    def test_plain_keywords(self):
        assert parse_search_query("hello world") == {"excerpt": "hello world"}
