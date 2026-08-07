# -*- coding: utf-8 -*-
"""Agent capability benchmark framework.

Evaluates local (ADLA/GGUF) and cloud models on MMLU-Pro, TAU2-Bench, and BFCL.
Two-phase evaluation: summary test (~500 fixed items) and full test.
Separates data collection (inference) from scoring (offline computation).
"""
