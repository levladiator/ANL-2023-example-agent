from agents.daniM_agent.negotiation_strategies.negotiation_strategy import NegotiationStrategy
from geniusweb.issuevalue.Bid import Bid

class ContinuedConcessionsStrategy(NegotiationStrategy):
    def __init__(self):
        self.target_utility_delta = 0.01

    def score_bid(self, bid: Bid, agent) -> float:
        if len(agent.own_bids) > 2:
            own_last_bid_utility = float(agent.profile.getUtility(agent.own_bids[-1]))
            current_bid_utility = float(agent.profile.getUtility(bid))
            own_delta = current_bid_utility - own_last_bid_utility

            opponent_last_bid_utility = agent.opponent_model.get_predicted_utility(agent.own_bids[-1])
            opponent_current_bid_utility = agent.opponent_model.get_predicted_utility(bid)
            opponent_delta = opponent_current_bid_utility - opponent_last_bid_utility

            if own_delta >= 0 or opponent_delta <= 0:
                return 0

            return self.score_bid_for_social_welfare(bid, agent)
        else:
            return agent.profile.getUtility(bid)