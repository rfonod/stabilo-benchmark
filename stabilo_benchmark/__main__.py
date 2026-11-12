#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Allow `python -m stabilo_benchmark <command>` to behave like the console script."""

import sys

from stabilo_benchmark.cli import main

if __name__ == '__main__':
    # sys.exit, not a bare call: the console-script wrapper setuptools generates exits with
    # main()'s return value, and `python -m` must agree with it. Discarding it made every
    # invocation succeed, so a scripted campaign would run on past a failed stage.
    sys.exit(main())
