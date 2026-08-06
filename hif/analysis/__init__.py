"""Text-analysis instruments for hif.

This sub-package contains tools that analyze texts (prompts and generated
continuations) as objects — independent of the model under analysis.

Current modules
---------------
attention.py
    AttentionAnalyzer: a bidirectional encoder (DistilBERT by default) used
    purely as a text-analysis instrument.  It is NOT the model under analysis
    and does NOT probe the generation mechanism.  Its attention weights reveal
    the internal structure of the *input text* and the *generated text*, not
    the process by which the generation model produced the continuation.
"""
