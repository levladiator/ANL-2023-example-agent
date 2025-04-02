from agents.daniM_agent.negotiation_strategies.negotiation_strategy import NegotiationStrategy
from geniusweb.issuevalue.Bid import Bid

class ContinuedConcessionsStrategy(NegotiationStrategy):
    def __init__(self):
        self.target_utility_delta = 0.01

    def score_bid(self, bid: Bid, agent) -> float:
        if len(agent.own_bids) > 2:
            return self.score_bid_for_social_welfare(bid, agent)
        return agent.profile.getUtility(bid)