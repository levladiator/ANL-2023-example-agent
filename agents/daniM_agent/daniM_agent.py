import logging

from collections import defaultdict
from time import time
from typing import cast, Optional

import numpy as np
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from agents.daniM_agent.utils.opponent_model import OpponentModel
from agents.daniM_agent.enums import Fairness, Stance, NegotiationType

from agents.daniM_agent.negotiation_strategies.negotiation_strategy_factory import NegotiationStrategyFactory

from agents.daniM_agent.negotiation_strategies.negotiation_strategy import NegotiationStrategy
from agents.daniM_agent.negotiation_strategies.aggressive_early_offers_strategy import AggressiveEarlyOffersStrategy

EXPLORATION_TIME_LIMIT = 0.1

class DaniMAgent(DefaultParty):

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.all_bids: list[Bid] = []
        self.bids_times: dict[Bid, int] = defaultdict(int)
        self.last_received_bid: Bid = None
        self.last_bid_sent_index: int = 0
        self.own_bids: list[Bid] = []

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.batna: Optional[float] = None
        self.reservation_value: float = 0.8
        self.alpha: float = 0.95
        self.eps: float = 0.1

        self.opponent_model: OpponentModel = None
        self.opponent_utilities: list[float] = []
        self.opponent_negotiation_type: NegotiationType = NegotiationType.UNKNOWN
        self.opponent_fairness: Fairness = Fairness.FAIR
        self.opponent_stance: Stance = Stance.NEUTRAL
        self.classification_done = False

        self.negotiation_strategy: NegotiationStrategy = AggressiveEarlyOffersStrategy()

        self.logger.log(logging.INFO, "party is initialized")

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after it has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()

            self.domain = self.profile.getDomain()
            profile_connection.close()

            # set best alternative to no agreement
            batna_bid = self.profile.getReservationBid()
            self.batna = self.profile.getUtility(batna_bid) if batna_bid is not None else 0

            # compose a list of all possible bids
            all_bids_list = AllBidsList(self.domain)
            self.all_bids = []
            for i in range(all_bids_list.size()):
                self.all_bids.append(all_bids_list.get(i))
            self.all_bids.sort(key=lambda x: self.profile.getUtility(x), reverse=True)

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            {"SAOP"},
            {"geniusweb.profile.utilityspace.LinearAdditive"},
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Template agent for the ANL 2025 competition"

    def adjust_strategy_by_opponent_type(self):
        if self.opponent_negotiation_type == NegotiationType.CONCEDER:
            self.alpha = 0.95  # we want to be more selfish if the opponent is a conceder
            self.reservation_value = 0.75
            self.eps = 0.02
        elif self.opponent_negotiation_type == NegotiationType.HARDLINER:
            self.alpha = 0.80  # we have to concede more if the opponent is a hardliner
            self.reservation_value = 0.6
            self.eps = 0.08
        elif self.opponent_negotiation_type == NegotiationType.RANDOM:
            self.alpha = 0.87
            self.reservation_value = 0.65
            self.eps = 0.05

    def adjust_opponent_fairness(self, bid: Bid):
        our_utility: float = float(self.profile.getUtility(bid))
        opponent_utility: float = float(self.opponent_model.get_predicted_utility(bid))
        is_fair = abs(our_utility - opponent_utility) <= 0.4 or opponent_utility <= 0.5
        self.opponent_fairness = Fairness.FAIR if is_fair else Fairness.UNFAIR

    def adjust_opponent_stance(self):
        if len(self.opponent_model.offers) < 2:
            return

        last_bid_utility = self.opponent_model.get_predicted_utility(self.opponent_model.offers[-1])
        before_last_bid_utility = self.opponent_model.get_predicted_utility(self.opponent_model.offers[-2])
        delta_utility = last_bid_utility - before_last_bid_utility
        if delta_utility > 0:
            self.opponent_stance = Stance.GENEROUS
        elif delta_utility == 0:
            self.opponent_stance = Stance.NEUTRAL
        else:
            self.opponent_stance = Stance.GREEDY

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid
            if self.profile and bid:
                self.opponent_utilities.append(self.profile.getUtility(bid))

            progress = self.progress.get(int(time() * 1000))

            # Classification after N observations
            if (not self.classification_done) and (progress >= EXPLORATION_TIME_LIMIT):
                diffs = np.diff(self.opponent_utilities)
                positive_diffs = np.sum(diffs > 0)
                negative_diffs = np.sum(diffs < 0)
                if positive_diffs > int(0.4 * len(self.opponent_utilities)) and negative_diffs > int(0.4 * len(self.opponent_utilities)):
                    self.opponent_negotiation_type = NegotiationType.RANDOM
                else:
                    diff = self.opponent_utilities[-1] - self.opponent_utilities[0]
                    if diff > 0.15:
                        self.opponent_negotiation_type = NegotiationType.CONCEDER
                    else:
                        self.opponent_negotiation_type = NegotiationType.HARDLINER
                self.classification_done = True
                self.adjust_strategy_by_opponent_type()

                self.last_bid_sent_index = 0
                self.all_bids.sort(key=lambda x: float(self.profile.getUtility(x)) + self.opponent_model.get_predicted_utility(x), reverse=True)

            self.adjust_opponent_fairness(bid)
            self.adjust_opponent_stance()
            if progress >= EXPLORATION_TIME_LIMIT:
                self.negotiation_strategy = NegotiationStrategyFactory.select_strategy(self.opponent_negotiation_type, self.opponent_stance, self.opponent_fairness)

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            action = Offer(self.me, bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool:
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(int(time() * 1000))

        # very basic approach that accepts if the offer is valued above 0.7 and
        # 95% of the time towards the deadline has passed
        conditions = [
            self.profile.getUtility(bid) > self.reservation_value,
            progress >= 0.99 and self.profile.getUtility(bid) > self.batna,
        ]

        return any(conditions)

    def find_bid(self) -> Bid:
        best_bid_score = -1.0
        best_bid = None
        best_bid_index = 0

        window_size = max(self.progress.get(int(time() * 1000)) * 750, 5)

        right_boundary: int = min(int(self.last_bid_sent_index + window_size), len(self.all_bids))

        for i in range(self.last_bid_sent_index, right_boundary):
            bid = self.all_bids[i]
            bid_score = self.negotiation_strategy.score_bid(bid, self)
            own_utility = self.profile.getUtility(bid)
            if bid_score > best_bid_score and own_utility > self.reservation_value:
                best_bid_score, best_bid, best_bid_index = bid_score, bid, i

        if best_bid is None:
            return self.find_bid()

        self.last_bid_sent_index = best_bid_index
        self.own_bids.append(best_bid)

        return best_bid

    def score_bid(self, bid: Bid, alpha, eps) -> float:
        """Calculate heuristic score for a bid

        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self-interested and
                altruistic behaviour.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time.

        Returns:
            float: score
        """
        progress = self.progress.get(int(time() * 1000))

        our_utility = float(self.profile.getUtility(bid))

        time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_utility

        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
            score += opponent_score

        return score
