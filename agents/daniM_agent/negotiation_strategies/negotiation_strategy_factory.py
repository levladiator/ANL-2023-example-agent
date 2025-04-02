from agents.daniM_agent.enums import Fairness, Stance, NegotiationType
from agents.daniM_agent.negotiation_strategies.negotiation_strategy import NegotiationStrategy
from agents.daniM_agent.negotiation_strategies.continued_concessions_strategy import ContinuedConcessionsStrategy
from agents.daniM_agent.negotiation_strategies.logrolling_strategy import LogrollingStrategy
from agents.daniM_agent.negotiation_strategies.respond_to_extreme_offer_strategy import RespondToExtremeOfferStrategy


class NegotiationStrategyFactory:
    @staticmethod
    def select_strategy(opponent_negotiation_model: NegotiationType, opponent_stance: Stance, opponent_fairness: Fairness) -> NegotiationStrategy:
        if opponent_negotiation_model == NegotiationType.CONCEDER:
            if opponent_stance == Stance.GREEDY or opponent_fairness == Fairness.UNFAIR:
                return LogrollingStrategy()
            elif opponent_stance == Stance.NEUTRAL or opponent_stance == Stance.GENEROUS:
                return ContinuedConcessionsStrategy()

        elif opponent_negotiation_model == NegotiationType.HARDLINER:
            if opponent_stance == Stance.GREEDY or opponent_fairness == Fairness.UNFAIR:
                return RespondToExtremeOfferStrategy()
            elif opponent_stance == Stance.NEUTRAL:
                return LogrollingStrategy()
            elif opponent_stance == Stance.GENEROUS:
                return ContinuedConcessionsStrategy()

        elif opponent_negotiation_model == NegotiationType.RANDOM:
            return ContinuedConcessionsStrategy()

        elif opponent_negotiation_model == NegotiationType.UNKNOWN:
            if opponent_stance == Stance.GREEDY or opponent_fairness == Fairness.UNFAIR:
                return RespondToExtremeOfferStrategy()
            elif opponent_stance == Stance.NEUTRAL:
                return LogrollingStrategy()
            elif opponent_stance == Stance.GENEROUS:
                return ContinuedConcessionsStrategy()
        return LogrollingStrategy()
