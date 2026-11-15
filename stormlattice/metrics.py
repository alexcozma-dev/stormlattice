"""
Scores for the evaluation table.

    Brier score, and Brier skill score against the block-by-hour climatology (the headline)
    CSI, reported with frequency bias
    birth hit rate, reported with false-alarm ratio
    bootstrap interval over test days, and a win count ("better on 11 of 14 days")

Every function takes arrays and returns numbers; scripts/evaluate.py does the looping and printing.

To write: all of the above. csi currently lives in scripts/evaluate.py and moves here.
"""
