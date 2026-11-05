"""An empty module used as the "main" of a spawned worker process.

A worker started with the "spawn" method imports the main module of the parent
so that anything pickled by reference to ``__main__`` can be found again. A
plain script that has no ``if __name__ == "__main__"`` guard (a benchmark, a
``python -c`` snippet) would run again in every worker; pointing spawn at this
harmless module instead stops that. See :func:`volura.workers._safe_main`.
"""
