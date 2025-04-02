from agents.daniM_agent.negotiation_strategies.negotiation_strategy import NegotiationStrategy
from geniusweb.issuevalue.Bid import Bid


class RespondToExtremeOfferStrategy(NegotiationStrategy):
    def __init__(self):
        self.delta = 0.03

    def score_bid(self, bid: Bid, agent) -> float:
        own_utility = float(agent.profile.getUtility(bid))

        if agent.opponent_model is None or len(agent.opponent_model.offers) < 1:
            return own_utility

        opponent_utility_last_bid = agent.opponent_model.get_predicted_utility(agent.opponent_model.offers[-1])

        if own_utility < opponent_utility_last_bid - self.delta:
            return 0

        return 1 - abs(own_utility - opponent_utility_last_bid)
