# Sufficiency labelling guide (after Joren et al., 2025, Sec. 3.1)

1 = SUFFICIENT: the context alone lets you give ONE definitive answer.
    - The answer does not need to match your own knowledge; judge the context only.
    - Multi-hop reasoning is allowed only if every link is stated in the context.

0 = INSUFFICIENT if any of these hold:
    - The needed information is missing or incomplete.
    - Answering needs outside or specialist knowledge not in the context.
    - The context is inconclusive.
    - The context is CONTRADICTORY: passages support different answers and
      nothing tells you which one to choose (Joren Remark 3).

Never use your own world knowledge. Label in one sitting where possible.
