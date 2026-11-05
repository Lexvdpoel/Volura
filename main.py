#!/usr/bin/env python3
"""Start Volura."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()      # first, so a frozen build can spawn workers
    from volura.startup import main
    raise SystemExit(main())
