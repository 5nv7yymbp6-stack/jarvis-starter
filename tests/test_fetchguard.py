import asyncio
import unittest

from backtalk import fetchguard as F

REAL_403 = {"bytes": 0, "code": 403, "codeText": "Forbidden",
            "result": "The server returned HTTP 403 Forbidden.\n\nThe response body was not retrieved."}
URL = "https://bsd.sos.mo.gov/Search/BusinessEntity?FilingName=AY+Advanced+Agency"


def run(coro):
    return asyncio.run(coro)


class FetchGuardTest(unittest.TestCase):
    def test_the_missouri_403_is_caught_and_points_at_the_browser(self):
        out = run(F.post_tool({"tool_name": "WebFetch", "tool_input": {"url": URL},
                               "tool_response": REAL_403}, "t1", None))
        ctx = out["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"], "PostToolUse")
        self.assertIn("mcp__playwright__browser_navigate", ctx)
        self.assertIn(URL, ctx)
        self.assertIn("Do not tell Yousif to look it up himself", ctx)

    def test_a_403_in_plain_text_is_caught(self):
        self.assertEqual(F.blocked("The server returned HTTP 403 Forbidden."), "HTTP 403")
        self.assertEqual(F.blocked("Access denied"), "access denied")

    def test_a_good_fetch_is_left_alone(self):
        ok = {"code": 200, "result": "AY Advanced Technologies LLC, Active. " * 5}
        self.assertEqual(run(F.post_tool({"tool_name": "WebFetch", "tool_response": ok}, "t", None)), {})

    def test_a_long_page_that_mentions_blocked_is_not_a_failure(self):
        page = "How to get unblocked: this article explains why sites get blocked by Cloudflare. " * 10
        self.assertIsNone(F.blocked({"code": 200, "result": page}))

    def test_other_tools_are_ignored(self):
        self.assertEqual(run(F.post_tool({"tool_name": "Bash", "tool_response": REAL_403}, "t", None)), {})

    def test_a_failed_fetch_points_at_the_browser(self):
        out = run(F.post_tool_failure({"tool_name": "WebFetch", "tool_input": {"url": URL},
                                       "error": "Request failed with status code 403"}, "t", None))
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"], "PostToolUseFailure")
        self.assertIn("browser", out["hookSpecificOutput"]["additionalContext"])

    def test_an_interrupt_is_not_a_failure(self):
        self.assertEqual(run(F.post_tool_failure({"tool_name": "WebFetch", "error": "x",
                                                  "is_interrupt": True}, "t", None)), {})

    def test_garbage_input_never_raises(self):
        self.assertEqual(run(F.post_tool(None, None, None)), {})
        self.assertEqual(run(F.post_tool({"tool_name": "WebFetch", "tool_response": object()}, "t", None)), {})


if __name__ == "__main__":
    unittest.main()
