"""The vault indexer (vault-search/index.py): atomic save, one run at a time, retry after a failed embedding, and never
surfacing a superseded or archived Memory 2.0 item. A temp vault and a fake embedder: no Ollama, no real index."""
import fcntl
import importlib.util
import json
import os
import tempfile
import time
import unittest

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vault-search", "index.py")


def load():
    spec = importlib.util.spec_from_file_location("vault_index_under_test", PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@unittest.skipUnless(os.path.exists(PATH), "vault-search/index.py is not on this machine")
class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = os.path.join(self.tmp.name, "vault")
        self.idx_dir = os.path.join(self.tmp.name, "search")
        os.makedirs(self.vault)
        os.makedirs(self.idx_dir)
        self.m = load()
        self.m.VAULT_DIR, self.m.INDEX_DIR = self.vault, self.idx_dir
        self.m.INDEX_FILE = os.path.join(self.idx_dir, "index.json")
        self.m.LOCK_FILE = os.path.join(self.idx_dir, "index.lock")
        self.calls = []
        self.fail_on = set()

        def embed(text):
            self.calls.append(text)
            if any(k in text for k in self.fail_on):
                raise RuntimeError("ollama down")
            return [0.1, 0.2]
        self.m.embed = embed

    def tearDown(self):
        self.tmp.cleanup()

    def note(self, rel, text, mtime=None):
        p = os.path.join(self.vault, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        if mtime:
            os.utime(p, (mtime, mtime))
        return p

    @staticmethod
    def raw(path):
        with open(path, "rb") as fh:
            return fh.read()

    def index(self):
        with open(self.m.INDEX_FILE, encoding="utf-8") as fh:
            return json.load(fh)

    def files_in_index(self):
        return {e["file"] for e in self.index()["entries"]}


BODY = "this is a note about the cash buyer list in each market with enough words to be a real chunk of text"


class IndexingTests(Base):
    def test_a_new_note_is_indexed_and_an_unchanged_one_is_not_embedded_again(self):
        self.note("a.md", BODY, 1000)
        self.m.main()
        self.assertEqual(self.files_in_index(), {"a.md"})
        n = len(self.calls)
        self.m.main()
        self.assertEqual(len(self.calls), n)

    def test_a_changed_note_is_re_embedded_and_its_old_chunks_replaced(self):
        self.note("a.md", BODY, 1000)
        self.m.main()
        self.note("a.md", BODY + " and a brand new sentence here", 2000)
        self.m.main()
        texts = [e["text"] for e in self.index()["entries"] if e["file"] == "a.md"]
        self.assertEqual(len(texts), 1)
        self.assertIn("brand new sentence", texts[0])

    def test_a_deleted_note_leaves_the_index(self):
        p = self.note("a.md", BODY, 1000)
        self.m.main()
        os.unlink(p)
        self.m.main()
        self.assertEqual(self.files_in_index(), set())

    def test_transcripts_are_still_skipped(self):
        self.note("raw transcript.md", BODY, 1000)
        self.m.main()
        self.assertEqual(self.files_in_index(), set())


class RetryTests(Base):
    def test_a_failed_embedding_is_not_marked_indexed_and_is_retried_next_run(self):
        self.note("a.md", BODY + " zzfail", 1000)
        self.fail_on = {"zzfail"}
        self.m.main()
        self.assertEqual(self.files_in_index(), set())
        self.assertNotIn("a.md", self.index()["files"])                       # not marked done
        self.fail_on = set()
        self.m.main()
        self.assertEqual(self.files_in_index(), {"a.md"})                     # the next run indexes it, untouched file and all

    def test_a_failed_re_embedding_keeps_the_old_entries(self):
        self.note("a.md", BODY, 1000)
        self.m.main()
        old = [e["text"] for e in self.index()["entries"]]
        self.note("a.md", BODY + " zzfail", 2000)
        self.fail_on = {"zzfail"}
        self.m.main()
        self.assertEqual([e["text"] for e in self.index()["entries"]], old)  # still searchable, never emptied
        self.assertEqual(self.index()["files"]["a.md"], 1000)


class MemoryStatusTests(Base):
    def mem(self, rel, status):
        return self.note(rel, f'---\nmemory_schema: 1\nid: "x"\nstatus: "{status}"\n---\n\n{BODY}\n', 1000)

    def test_active_memories_are_searchable_superseded_and_archived_are_not(self):
        self.mem("11 - Memory/mem_a.md", "active")
        self.mem("11 - Memory/mem_b.md", "superseded")
        self.mem("11 - Memory/mem_c.md", "archived")
        self.m.main()
        self.assertEqual(self.files_in_index(), {"11 - Memory/mem_a.md"})

    def test_a_memory_that_becomes_superseded_leaves_the_index(self):
        self.mem("11 - Memory/mem_a.md", "active")
        self.m.main()
        self.assertEqual(self.files_in_index(), {"11 - Memory/mem_a.md"})
        p = self.mem("11 - Memory/mem_a.md", "superseded")
        os.utime(p, (2000, 2000))
        self.m.main()
        self.assertEqual(self.files_in_index(), set())

    def test_status_only_matters_inside_the_memory_folder(self):
        self.note("a.md", f'---\nstatus: superseded\n---\n{BODY}', 1000)         # an ordinary note's own status is its business
        self.m.main()
        self.assertEqual(self.files_in_index(), {"a.md"})

    def test_a_memory_file_with_no_status_is_searchable(self):
        self.note("11 - Memory/mem_z.md", BODY, 1000)
        self.m.main()
        self.assertEqual(self.files_in_index(), {"11 - Memory/mem_z.md"})


class SafetyTests(Base):
    def test_a_failed_save_leaves_the_old_index_intact_and_no_temp_file(self):
        self.note("a.md", BODY, 1000)
        self.m.main()
        before = self.raw(self.m.INDEX_FILE)
        real = os.replace
        os.replace = lambda *a: (_ for _ in ()).throw(OSError("disk"))
        try:
            with self.assertRaises(OSError):
                self.m.save_index({"files": {}, "entries": []})
        finally:
            os.replace = real
        self.assertEqual(self.raw(self.m.INDEX_FILE), before)
        self.assertEqual([n for n in os.listdir(self.idx_dir) if n.endswith(".tmp")], [])

    def test_a_reader_never_sees_a_partial_index_during_a_save(self):
        self.note("a.md", BODY, 1000)
        self.m.main()
        seen = []
        real = json_dump = self.m.json.dump

        def slow_dump(obj, f, **kw):
            real(obj, f, **kw)
            try:
                with open(self.m.INDEX_FILE, encoding="utf-8") as rf:
                    seen.append(json.load(rf))                                         # a reader, mid-save
            except ValueError:
                seen.append(None)
        self.m.json.dump = slow_dump
        try:
            self.m.save_index({"files": {"b.md": 1}, "entries": []})
        finally:
            self.m.json.dump = json_dump
        self.assertTrue(seen and seen[0] is not None)                                     # the old complete file was still there

    def test_only_one_run_at_a_time(self):
        with open(self.m.LOCK_FILE, "a+") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertFalse(self.m.run_locked())
        self.note("a.md", BODY, 1000)
        self.assertTrue(self.m.run_locked())
        self.assertEqual(self.files_in_index(), {"a.md"})


if __name__ == "__main__":
    unittest.main()
