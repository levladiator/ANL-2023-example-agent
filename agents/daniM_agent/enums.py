from enum import Enum

class Fairness(Enum):
    FAIR = 1,
    UNFAIR = 0,

class Stance(Enum):
    GREEDY = 0,
    NEUTRAL = 1,
    GENEROUS = 2,

class NegotiationType(Enum):
    CONCEDER = 0,
    HARDLINER = 1,
    RANDOM = 2,
    UNKNOWN = 3,