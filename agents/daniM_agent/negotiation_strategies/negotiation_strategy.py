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
        progress = agent.progress.get(int(time() * 1000))

        our_utility = float(agent.profile.getUtility(bid))

        time_pressure = 1.0 - progress ** (1 / agent.eps)
        score = agent.alpha * time_pressure * our_utility

        if agent.opponent_model is not None:
            opponent_utility = agent.opponent_model.get_predicted_utility(bid)
            opponent_score = (1.0 - agent.alpha * time_pressure) * opponent_utility
            score += opponent_score

        return score