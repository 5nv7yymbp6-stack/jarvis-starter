"""TEST-ONLY fixture arrangement for the Memory 2.0 store. NOT part of the product.

`MemoryStore.put` and `.apply` run every memory through the gate. Some tests need a store in a state the gate would never
create (history that already ended, an inferred item that is private, two items that conflict) so they can test how the
store READS or REPAIRS such states. `arrange` writes those files directly, bypassing every check. It exists here, under
tests/, and no production module may import it: tests/test_memory_trust_gaps.py enforces that with an AST scan.
"""
import os

from backtalk import memory_store as S


def arrange(store, item, evidence):
    """Write `item` into `store` exactly as given, with no gate and no screening. Returns the id."""
    text = S.serialize(item, evidence)
    with store._locked(write=True):
        S._write_atomic(store._path(item.id), text)
    return item.id
