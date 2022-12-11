def simulate_market(action_dict, slot_bids):
    auction_wins = [0] * 6
    auction_prices = [0] * 6

    for slot in range(0, 6):
        agents_bid_size = action_dict["size"][slot]
        agents_bid_price = action_dict["price"][slot]
        # get settlement price
        settlement_price_DE = [
            bid["settlement_price"] for bid in slot_bids if bid["country"] == "DE"][0]
        # first check if agents bid size is 0 , which means:
        # not participating in auciton
        if agents_bid_size == 0.0:
            auction_wins[slot] = 0
        else:
            # second check if agents bid price is higher than
            # the settlement price of Germany
            if agents_bid_price > settlement_price_DE:
                # if it is higher, the slot is LOST.
                auction_wins[slot] = -1
            else:
                # If agents bid price is lower than settlement
                # price (bid could be in awarded bids)
                # get CBMP of countries without LMP
                unique_country_bids = list(
                    {v["country"]: v for v in slot_bids}.values())
                grouped_prices = [
                    x["settlement_price"] for x in unique_country_bids]
                cbmp = max(set(grouped_prices), key=grouped_prices.count)
                # cbmp = cross border marginal price
                # check if settlement_price_DE is same as CBMP
                # (no limit constraints where hit)
                if cbmp == settlement_price_DE:
                    price_filter = cbmp
                else:
                    # if Germany has a price based on limit
                    # constraints
                    price_filter = settlement_price_DE
                # as the probability is high that the agents bid
                # moved the last bid out of the list,
                # we have to check which bids moved out of the
                # list and what is the new settlement price
                # sort the bid list based on the price
                slot_bids_list_sorted_by_price = sorted(
                    slot_bids, key=lambda x: x["price"])
                # filter the bid list by the settlement price of
                # either the CBMP or the LMP of germany
                slot_bids_filtered = [
                    bid
                    for bid in slot_bids_list_sorted_by_price
                    if bid["settlement_price"] == price_filter]
                accumulated_replaced_capacity = 0
                # fill the demand
                for bid in range(0, len(slot_bids_filtered)):
                    bid_capacity = slot_bids_filtered[-(bid + 1)]["size"]
                    accumulated_replaced_capacity += bid_capacity
                    if accumulated_replaced_capacity >= agents_bid_size:
                        if slot_bids_filtered[-(bid + 1)]["indivisible"] is False:
                            settlement_price_DE = slot_bids_filtered[-(bid + 1)][
                                "price"]
                            # set boolean for auction win
                            auction_wins[slot] = 1
                            # set settlement price for the
                            # current auctioned slot
                            auction_prices[slot] = settlement_price_DE
                            break
                        else:
                            # bid is INDIVISIBLE, so move one
                            # bids further for new settlement price
                            accumulated_replaced_capacity -= bid_capacity
                            continue
    return auction_wins, auction_prices
