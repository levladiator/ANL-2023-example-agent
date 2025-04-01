from time import time

from agents.daniM_agent.negotiation_strategies.negotiation_strategy import NegotiationStrategy
from geniusweb.issuevalue.Bid import Bid

class ContinuedSmallerConcessionsStrategy(NegotiationStrategy):
    def __init__(self):
        self.target_utility_delta = 0.01

    def score_bid(self, bid: Bid, agent) -> float:
        if (len(agent.own_bids) > 2):
            own_last_bid_utility = float(agent.profile.getUtility(agent.own_bids[-1]))
            current_bid_utility = float(agent.profile.getUtility(bid))
            own_delta = current_bid_utility - own_last_bid_utility

            opponent_last_bid_utility = agent.opponent_model.get_predicted_utility(agent.own_bids[-1])
            opponent_current_bid_utility = agent.opponent_model.get_predicted_utility(bid)
            opponent_delta = opponent_current_bid_utility - opponent_last_bid_utility

            if own_delta >= 0 or opponent_delta <= 0:
                return 0

            progress = agent.progress.get(int(time() * 1000))

            our_utility = float(agent.profile.getUtility(bid))

            time_pressure = 1.0 - progress ** (1 / agent.eps)
            score = agent.alpha * time_pressure * our_utility

            if agent.opponent_model is not None:
                opponent_utility = agent.opponent_model.get_predicted_utility(bid)
                opponent_score = (1.0 - agent.alpha * time_pressure) * opponent_utility
                score += opponent_score

            return score
        return agent.profile.getUtility(bid)