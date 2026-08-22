"""Import flask_server with the optional RAG dependencies stubbed as PRESENT.

Why this test exists.

`RAGState` was used in three places and defined in none. Python evaluates
function parameter annotations at definition time, so

    def retrieve(state: RAGState) -> RAGState:

raised `NameError: name 'RAGState' is not defined` the instant the module was
imported -- but ONLY when langchain/pinecone/langgraph were installed, because
otherwise RAG_AVAILABLE is False and the whole block is skipped.

That is the worst shape a bug can have: the feature was dead in exactly the
environment it was built for, and perfectly healthy in every environment where it
did nothing. `python -m compileall` passes, `py_compile` passes, and importing it
on a machine without the deps passes. Nothing caught it.

So this test forces the RAG branch to execute with stub modules, and asserts the
import succeeds. It deliberately does NOT require the real dependencies -- it has
to run in CI, where installing langchain to check a NameError would be absurd.

    python test_flask_import.py
"""

from __future__ import annotations

import os
import sys
import types


def install_stubs() -> None:
    """Minimal stand-ins for the optional RAG packages.

    These only need to be importable and callable. The point is to make
    RAG_AVAILABLE true so the guarded block runs, not to exercise retrieval.
    """
    os.environ.setdefault("PINECONE_API_KEY", "stub-key-for-import-test")
    os.environ.setdefault("COHERE_API_KEY", "stub-key-for-import-test")
    os.environ.setdefault("GROQ_API_KEY", "stub-key-for-import-test")

    class _Stub:
        """Accepts any construction and any attribute access."""

        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return self

        def __getattr__(self, _name):
            return _Stub()

    class _Graph(_Stub):
        """StateGraph needs real method names, or the module's own try/except
        swallows the failure and we would not actually reach compile()."""

        def add_node(self, *a, **k):
            return self

        def add_edge(self, *a, **k):
            return self

        def set_entry_point(self, *a, **k):
            return self

        def compile(self, *a, **k):
            return _Stub()

    def mod(name: str, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    mod("langchain_cohere", CohereEmbeddings=_Stub)
    mod("langchain_pinecone", PineconeVectorStore=_Stub)
    mod("langchain_groq", ChatGroq=_Stub)
    mod("langchain_core")
    mod("langchain_core.documents", Document=_Stub)
    mod("pinecone", Pinecone=_Stub)
    mod("langgraph")
    mod("langgraph.graph", StateGraph=_Graph, END="__end__")


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    install_stubs()

    try:
        import flask_server
    except NameError as e:
        print(f"  FAIL  NameError in the RAG branch: {e}")
        print("        A name is used inside `if RAG_AVAILABLE:` that is never")
        print("        defined. This crashes the server wherever the RAG")
        print("        dependencies are actually installed.")
        return 1
    except ImportError as e:
        # Missing MANDATORY deps (flask, torch, ...) is an environment problem,
        # not the defect under test. Say so plainly rather than reporting a pass.
        print(f"  SKIP  a mandatory dependency is missing: {e}")
        print("        Install flask, flask-cors, python-dotenv, numpy, pillow,")
        print("        scikit-learn and torch to run this check.")
        return 0
    except SystemExit as e:
        print(f"  FAIL  the module called sys.exit({e.code}) during import")
        return 1

    checks = 0

    assert getattr(flask_server, "RAG_AVAILABLE", None) is not None, \
        "RAG_AVAILABLE is not defined at module level"
    checks += 1

    # The specific name that was missing.
    assert hasattr(flask_server, "RAGState"), \
        "RAGState is still not defined at module level"
    checks += 1

    # It has to be a usable mapping type with the three fields the graph threads
    # through, not merely *some* object bound to that name.
    ann = getattr(flask_server.RAGState, "__annotations__", {})
    for field in ("question", "documents", "answer"):
        assert field in ann, f"RAGState is missing the '{field}' field"
        checks += 1

    # And the nodes must exist, which is what forces the annotations to evaluate.
    for fn in ("retrieve", "generate"):
        assert callable(getattr(flask_server, fn, None)), \
            f"{fn}() was not defined -- the RAG branch did not run"
        checks += 1

    print(f"  PASS  flask_server imports with RAG deps present ({checks} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
