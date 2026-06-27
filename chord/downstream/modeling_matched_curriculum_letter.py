"""Portable placeholder module for legacy MatchedCurriculumLETTER.

The exact T5 LETTER model is not imported by the cloud smoke backend. Keeping
this module path allows downstream adapters to resolve imports while the
repo-native train/eval path remains independent of old machine paths.
"""


class MatchedCurriculumLETTER:
    pass
