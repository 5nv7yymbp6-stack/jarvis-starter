"""Phase 8.0: characterization tests for backtalk/shellscan.py (no dedicated tests existed). They pin what the module
does TODAY, not what it ought to do. Where today's behavior is a known gap it is pinned in `KnownGapTests`, named as a
gap; fixing a gap changes how evidence.py and ledger.py behave and needs its own approval, at which point the matching
test is flipped on purpose."""
import re
import unittest

from backtalk import shellscan as SH

scan = SH._scan_bash


def redact(text):
    for pattern, repl in SH._PREVIEW_REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def unsafe_reason(cmd):
    segs, unsafe = scan(cmd)
    return segs, (unsafe[0] if unsafe else None)


class SplittingTests(unittest.TestCase):
    def test_a_plain_command_is_one_segment(self):
        self.assertEqual(scan("ls"), (["ls"], None))
        self.assertEqual(scan("FOO=1 ls"), (["FOO=1 ls"], None))

    def test_each_unquoted_separator_splits(self):
        self.assertEqual(scan("ls; pwd")[0], ["ls", "pwd"])
        self.assertEqual(scan("a && b || c")[0], ["a", "b", "c"])
        self.assertEqual(scan("a | b")[0], ["a", "b"])
        self.assertEqual(scan("a\nb")[0], ["a", "b"])

    def test_segments_are_stripped_and_empties_dropped(self):
        self.assertEqual(scan(" a ;  b ")[0], ["a", "b"])
        self.assertEqual(scan(""), ([], None))
        self.assertEqual(scan("   "), ([], None))
        self.assertEqual(scan(";;")[0], [])

    def test_quotes_protect_separators(self):
        self.assertEqual(scan("echo 'a;b'")[0], ["echo 'a;b'"])
        self.assertEqual(scan('echo "a;b"')[0], ['echo "a;b"'])

    def test_a_backslash_escapes_a_separator(self):
        self.assertEqual(scan(r"echo a\;b")[0], [r"echo a\;b"])

    def test_an_escaped_quote_inside_double_quotes_does_not_close_them(self):
        self.assertEqual(scan('echo "a\\";b"')[0], ['echo "a\\";b"'])

    def test_a_single_quote_has_no_escapes(self):
        self.assertEqual(scan("echo 'a\\'; ls")[0], ["echo 'a\\'", "ls"])

    def test_redirects_with_ampersand_are_not_backgrounding(self):
        for cmd in ("ls 2>&1", "cat <&3", "ls &>f"):
            self.assertEqual(scan(cmd), ([cmd], None), cmd)

    def test_array_assignment_is_one_segment(self):
        for cmd in ("a=(1 2)", "a+=(1 2)"):
            self.assertEqual(scan(cmd), ([cmd], None), cmd)

    def test_safe_glob_qualifiers_stay_in_the_segment(self):
        self.assertEqual(scan("ls *(.)"), (["ls *(.)"], None))


class UnparseableTests(unittest.TestCase):
    """segments None and no `unsafe` = the structure could not be understood; callers treat both as unverifiable."""

    def test_unbalanced_quotes_and_trailing_backslash(self):
        for cmd in ("echo 'x", 'echo "x', "echo x\\"):
            self.assertEqual(scan(cmd), (None, None), cmd)

    def test_unclosed_array_and_unclosed_qualifier(self):
        for cmd in ("a=(1 2", "ls *("):
            self.assertEqual(scan(cmd), (None, None), cmd)


class UnsafeConstructTests(unittest.TestCase):
    def test_background_forms(self):
        for cmd in ("ls &", "a |& b"):
            segs, why = unsafe_reason(cmd)
            self.assertIsNone(segs, cmd)
            self.assertEqual(why, "backgrounds a command", cmd)

    def test_grouping_and_process_substitution(self):
        for cmd in ("(ls)", "=(ls)", "x)", "a=(1;2)", "a=(1|2)"):
            segs, why = unsafe_reason(cmd)
            self.assertIsNone(segs, cmd)
            self.assertEqual(why, "shell grouping or process substitution", cmd)

    def test_unsafe_zsh_qualifiers_and_flags(self):
        for cmd in ("ls *(e:'rm x':)", "ls *(+fn)", "ls *()", "ls *(om[1])", "echo ${(e)x}", 'echo "${(e)x}"'):
            segs, why = unsafe_reason(cmd)
            self.assertIsNone(segs, cmd)
            self.assertEqual(why, "zsh glob qualifier or expansion I can't verify", cmd)

    def test_plain_parameter_expansion_is_fine(self):
        self.assertEqual(scan("echo ${x}"), (["echo ${x}"], None))

    def test_every_unsafe_tuple_is_a_reason_and_verb_pair(self):
        for t in (SH._UNSAFE_ZSH, SH._UNSAFE_BG, SH._UNSAFE_GROUP):
            self.assertEqual(len(t), 2)
            self.assertTrue(all(isinstance(x, str) and x for x in t))

    def test_safe_qualifier_set_excludes_eval_function_and_modifier_characters(self):
        for ch in "e+:[]()":
            self.assertNotIn(ch, SH._SAFE_QUAL)
        for ch in "./*0123456789":
            self.assertIn(ch, SH._SAFE_QUAL)


class KnownGapTests(unittest.TestCase):
    """Behavior pinned AS IT IS. Each is a place where the scanner vouches for something it cannot verify. Not fixed in
    Phase 8.0; a fix needs separate approval and then flips the assertion."""

    def test_gap_backticks_are_not_flagged(self):
        self.assertEqual(scan("echo `rm x`"), (["echo `rm x`"], None))

    def test_gap_command_substitution_that_looks_like_a_qualifier_passes(self):
        # `$(ls)` and `$(rm)`: the word before `(` is `$`, and the body's letters happen to be in the safe qualifier set
        self.assertEqual(scan("echo $(ls)"), (["echo $(ls)"], None))
        self.assertEqual(scan("echo $(rm)"), (["echo $(rm)"], None))

    def test_command_substitution_with_a_space_is_flagged_only_by_accident(self):
        segs, why = unsafe_reason("echo $(cat f)")
        self.assertIsNone(segs)
        self.assertEqual(why, "zsh glob qualifier or expansion I can't verify")

    def test_gap_brace_expansion_is_not_flagged(self):
        self.assertEqual(scan("echo {a,b}"), (["echo {a,b}"], None))

    def test_gap_heredoc_body_is_split_into_fake_commands(self):
        self.assertEqual(scan("cat <<EOF\nrm x\nEOF")[0], ["cat <<EOF", "rm x", "EOF"])


class RedactionTests(unittest.TestCase):
    CASES = {
        "curl https://u:pw@h.com/x": "curl https://***@h.com/x",
        "Authorization: Bearer abc123def": "Authorization: Bearer ***",
        "Bearer abc123": "Bearer ***",
        "X-Api-Key: abc": "X-Api-Key: ***",
        "Cookie: a=b": "Cookie: ***",
        "--password hunter2": "--password ***",
        "--token=abc": "--token=***",
        "curl -u me:pw h": "curl -u *** h",
        "curl -d 'a=b' h": "curl -d *** h",
        "FOO=bar ls": "FOO=*** ls",
        'export KEY="a b"': "export KEY=***",
        "API_KEY=abc sk-" + "b" * 20: "API_KEY=*** ***",
    }

    def test_each_family_is_redacted(self):
        for text, want in self.CASES.items():
            self.assertEqual(redact(text), want, text)

    def test_token_shapes_are_replaced_whole(self):
        for tok in ("sk-" + "a" * 20, "ghp_" + "A" * 25, "xoxb-1234567890-abc", "AKIA" + "A" * 16,
                    "eyJ" + "a" * 12 + "." + "b" * 12 + ".c", "a" * 40):
            self.assertEqual(redact(tok), "***", tok)
            self.assertEqual(redact(f"before {tok} after"), "before *** after", tok)

    def test_ordinary_text_is_untouched(self):
        for text in ("plain text", "ls -la /tmp", "a short sentence with no secrets"):
            self.assertEqual(redact(text), text)

    def test_redaction_is_idempotent(self):
        for text in list(self.CASES) + ["sk-" + "c" * 30 + " and FOO=bar"]:
            once = redact(text)
            self.assertEqual(redact(once), once, text)

    def test_the_secret_value_never_survives_in_the_covered_families(self):
        for text, secret in (("--password hunter2", "hunter2"), ("Bearer abc123", "abc123"), ("curl -u me:pw h", "me:pw"),
                             ("TOKEN=s3cr3tvalue", "s3cr3tvalue"), ("https://u:pw@h.com", "u:pw")):
            self.assertNotIn(secret, redact(text), text)

    def test_all_patterns_are_compiled_with_a_replacement(self):
        for pattern, repl in SH._PREVIEW_REDACTIONS:
            self.assertIsInstance(pattern, re.Pattern)
            self.assertIsInstance(repl, str)

    def test_specific_patterns_run_before_the_generic_catch_alls(self):
        # the last two entries are the NAME=value / long-blob catch-alls' neighbors: a url credential must be handled by
        # its own pattern (first), not left to the generic one
        self.assertIn("://", SH._PREVIEW_REDACTIONS[0][0].pattern)
        self.assertEqual(SH._PREVIEW_REDACTIONS[-1][1], "***")
        self.assertIn("32,", SH._PREVIEW_REDACTIONS[-1][0].pattern)


class RedactionKnownGapTests(unittest.TestCase):
    """Free-text and JSON password forms are NOT redacted. The module calls redaction a log aid, not a security control."""

    def test_gap_colon_form_is_not_redacted(self):
        self.assertEqual(redact("password: hunter2"), "password: hunter2")

    def test_gap_json_form_is_not_redacted(self):
        self.assertEqual(redact('{"password": "hunter2"}'), '{"password": "hunter2"}')

    def test_gap_sentence_form_is_not_redacted(self):
        self.assertEqual(redact("my password is hunter2"), "my password is hunter2")


class ScopeTests(unittest.TestCase):
    def test_module_has_no_side_effects_and_only_imports_re(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(SH))
        mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"re"})


if __name__ == "__main__":
    unittest.main()
