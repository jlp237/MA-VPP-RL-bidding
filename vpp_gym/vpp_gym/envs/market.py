import logging


def simulate_market(self, action_dict):
    """_summary_

    Args:
        action_dict (_type_): _description_
    """
    auction_bids = self.bids_df[self.market_start : self.market_end]
    logging.debug("log_step: " + str(self.logging_step) + " slot: " + "None" + " auction_bids = ")
    logging.debug(self.bids_df[self.market_start : self.market_end])

    logging.info("log_step: " + str(self.logging_step) + " slot: " + "None" + " Bid Submission time (D-1) = %s" % (self.bid_submission_time))
    logging.info("log_step: " + str(self.logging_step) + " slot: " + "None" + " Gate Closure time (D-1) = %s" % (self.gate_closure))
    logging.info("log_step: " + str(self.logging_step) + " slot: " + "None" + " Historic Data Window: from %s to %s " % (self.historic_data_start, self.historic_data_end))
    logging.info("log_step: " + str(self.logging_step) + " slot: " + "None" + " Forecast Data Window: from %s to %s " % (self.forecast_start, self.forecast_end))

    self.delivery_results["agents_bid_prices"] = [None] * 6
    self.delivery_results["agents_bid_sizes_round"] = [None] * 6
    self.delivery_results["slots_won"] = [None] * 6

    for slot in range(0, len(self.slot_date_list)):
        slot_date = self.slot_date_list[slot]
        logging.info("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " Current Slot Time: (D) = %s" % (slot_date))
        slot_bids = auction_bids[slot_date:slot_date].reset_index(drop=True).reset_index(drop=False)
        slot_bids_list = slot_bids.to_dict('records')
        # extract the bid size out of the agents action
        # ROUND TO FULL INTEGER
        '''agents_bid_size = round(action_dict["size"][slot])'''
        agents_bid_size = action_dict["size"][slot]
        self.delivery_results["agents_bid_sizes_round"][slot] = agents_bid_size
        # extract the bid price out of the agents action
        agents_bid_price = action_dict["price"][slot]
        self.delivery_results["agents_bid_prices"][slot] = agents_bid_price
        logging.info("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " agents_bid_size = %s" % (agents_bid_size))
        logging.info("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " agents_bid_price = %s" % (agents_bid_price))
        # get settlement price
        settlement_price_DE = [bid['settlement_price'] for bid in slot_bids_list if bid['country'] == "DE"][0]
        logging.info("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " settlement_price_DE : " + str(settlement_price_DE))
        self.delivery_results["slot_settlement_prices_DE"][slot] = settlement_price_DE

        # first check if agents bid size is 0 , which means: not participating in auciton
        if agents_bid_size == 0.0:
            logging.info("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " no participation in market of agent")
            # set slot won to None
            self.delivery_results["slots_won"][slot] = 0
        else:
            # second check if agents bid price is higher than the settlement price of Germany
            if agents_bid_price > settlement_price_DE:
                # if it is higher, the slot is LOST.
                self.delivery_results["slots_won"][slot] = -1
                # set settlement price for the current auctioned slot in slot_settlement_prices_DE list
            else:
                # If agents bid price is lower than settlement price (bid could be in awarded bids)
                # get CBMP of countries without LMP
                unique_country_bids = list({v['country']: v for v in slot_bids_list}.values())
                grouped_prices = [x['settlement_price'] for x in unique_country_bids]
                cbmp = max(set(grouped_prices), key=grouped_prices.count)
                logging.info("cbmp : " + str(cbmp))
                # check if settlement_price_DE is same as CBMP (no limit constraints where hit)
                if cbmp == settlement_price_DE:
                    price_filter = cbmp
                    logging.debug("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " DE has CBMP")
                else:
                    # if Germany has a price based on limit constraints
                    price_filter = settlement_price_DE
                    logging.debug("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " DE has LMP")

                # as the probability is high that the agents bid moved the last bid out of the list,
                # we have to check which bids moved out of the list and what is the new settlement price

                # sort the bid list based on the price
                slot_bids_list_sorted_by_price = sorted(slot_bids_list, key=lambda x: x['price'])
                # filter the bid list by the settlement price of either the CBMP or the LMP of germany
                slot_bids_filtered = [bid for bid in slot_bids_list_sorted_by_price if bid['settlement_price'] == price_filter]
                accumulated_replaced_capacity = 0

                slot_bids_filtered_size_sum = sum([bid['size'] for bid in slot_bids_filtered])
                # for the case the action_dict space is not dynamic and agent can choose any bid size,
                # it needs to be checked here if the agents_bid_size is too big and unrealistic
                if agents_bid_size >= slot_bids_filtered_size_sum:
                    logging.debug("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " unrealistic bid size")
                    # set auction won to false
                    self.delivery_results["slots_won"][slot] = -1
                    # set the old settlement_price_DE as the market price
                    self.delivery_results["slot_settlement_prices_DE"][slot] = settlement_price_DE
                else:
                    for bid in range(0, len(slot_bids_filtered)):
                        logging.debug("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " bid size = " + str(slot_bids_filtered[-(bid + 1)]["size"]))
                        logging.debug("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " bid price = " + str(slot_bids_filtered[-(bid + 1)]["price"]))
                        bid_capacity = slot_bids_filtered[-(bid + 1)]["size"]
                        accumulated_replaced_capacity += bid_capacity
                        logging.debug("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " accumulated_replaced_capacity = " + str(accumulated_replaced_capacity))

                        if accumulated_replaced_capacity >= agents_bid_size:
                            logging.debug("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " realistic bid size")
                            if slot_bids_filtered[-(bid + 1)]["indivisible"] is False:
                                logging.debug("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " bid is divisible, so current bids price is new settlement price")
                                new_settlement_price_DE = slot_bids_filtered[-(bid + 1)]["price"]
                                logging.info("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " new_settlement_price_DE = " + str(new_settlement_price_DE))
                                # set boolean for auction win
                                self.delivery_results["slots_won"][slot] = 1
                                # set settlement price for the current auctioned slot in slot_settlement_prices_DE list
                                self.delivery_results["slot_settlement_prices_DE"][slot] = new_settlement_price_DE
                                break
                            else:
                                logging.debug("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " bid is INDIVISIBLE, so move one bids further is new settlement price")
                                accumulated_replaced_capacity -= bid_capacity
                                continue

        logging.info("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " self.delivery_results['slots_won'] = ")
        logging.info("log_step: " + str(self.logging_step) + " slot: " + str(slot) + "\n" + " \n".join("slot won: \t{}".format(k) for k in self.delivery_results["slots_won"]))
        logging.info("log_step: " + str(self.logging_step) + " slot: " + str(slot) + "      agents bid_size = ")
        logging.info("log_step: " + str(self.logging_step) + " slot: " + str(slot) + "\n" + " \n".join("size: \t{}".format(round(k)) for k in action_dict["size"]))
        logging.info("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " self.delivery_results['slot_settlement_prices_DE'] = ")
        logging.info("log_step: " + str(self.logging_step) + " slot: " + str(slot) + "\n" + " \n".join("price: \t{}".format(k) for k in self.delivery_results["slot_settlement_prices_DE"]))
