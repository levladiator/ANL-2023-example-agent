from abc import ABC, abstractmethod
from geniusweb.issuevalue.Bid import Bid
from time import time


class NegotiationStrategy(ABC):

    @abstractmethod
    def score_bid(self, bid: Bid, agent) -> float:
        """
        Gets best bid based on current strategy
        """

    def score_bid_for_social_welfare(self, bid: Bid, agent) -> float:
        """
        Scores bid based on social welfare score, by adapting own utility score
        and predicted opponent utility score, and adding them together.

        Notes:
            - agent.eps = importance of passed time based on agent characteristics
            - agent.alpha = importance of agent.eps
        """
        progress = agent.progress.get(int(time() * 1000))

        our_utility = float(agent.profile.getUtility(bid))

        # Time pressure factor, high when progress is low or epsilon is high
        time_pressure = 1.0 - progress ** (1 / agent.eps)
        # Take into account our own utility
        # w1 = agent.alpha * time_pressure
        score = agent.alpha * time_pressure * our_utility

        if agent.opponent_model is not None:
            opponent_utility = agent.opponent_model.get_predicted_utility(bid)
            # w2 = 1 - w1
            opponent_score = (1.0 - agent.alpha * time_pressure) * opponent_utility
            # Take into account opponent's utility
            score += opponent_score

        return score
