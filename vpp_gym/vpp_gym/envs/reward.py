import logging

from .activation import simulate_activation
from .reservation import simulate_reservation
import random

def calculate_reward(self):  
    """_summary_

    Returns:
        _type_: _description_
    """      
    #print("in _calculate_reward() and logging_step = " + str(self.logging_step))
    # Step 1 of Reward Function: The Auction
    # did the agent win the auction? 
    # what was the revenue ?
    
    self.activation_results["total_not_reserved_energy"] = [0, 0, 0, 0, 0, 0]
    self.activation_results["total_not_activated_energy"] = [0, 0, 0, 0, 0, 0]
    
    # Concept: 
    # 1 Step is 1 complete day of activation 
    # 1 Step consists of : 
        # slot 0: slot_reward = auction_reward + reservation_reward + activation_reward
            # weighted_slot_reward = slot_reward / 3
        # slot 1: slot_reward = auction_reward + reservation_reward + activation_reward 
            # weighted_slot_reward = slot_reward / 3
        # slot 2: slot_reward = auction_reward + reservation_reward + activation_reward 
            # weighted_slot_reward = slot_reward / 3
        # slot 3: slot_reward = auction_reward + reservation_reward + activation_reward 
            # weighted_slot_reward = slot_reward / 3
        # slot 4: slot_reward = auction_reward + reservation_reward + activation_reward 
            # weighted_slot_reward = slot_reward / 3
        # slot 5: slot_reward = auction_reward + reservation_reward + activation_reward
            # weighted_slot_reward = slot_reward / 3
        
        # total_weighted_slot_reward = sum(weighted_slot_reward)
        # day_reward = sum(total_weighted_slot_reward)
        
        # weighted_day_reward = day_reward / 6
        
    day_profit: float = 0.
    day_revenue: float = 0.
    day_penalties: float = 0.

    day_reward: float = 0.
    
    logging.info("log_step: " + str(self.logging_step) + " slot: " +  "None" + " Reward Overview:")
    logging.info("log_step: " + str(self.logging_step) + " slot: " +  "None" + " self.activation_results['slots_won']: " + str(self.activation_results["slots_won"]))
    logging.info("log_step: " + str(self.logging_step) + " slot: " +  "None" + " len(self.activation_results['slots_won']) : "  + str(len(self.activation_results["slots_won"])))
    
    for slot in range(0, len(self.activation_results["slots_won"])):
        
        auction_reward = 0
        reservation_reward = 0 
        activation_reward = 0 
        
        slot_profit = 0
        slot_revenue = 0
        slot_penalty = 0
        
        logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " slot no. " + str(slot))
        
        # if agent lost the slot 
        if self.activation_results["slots_won"][slot] == -1:
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " slot no " + str(slot) + " was lost" )
            slot_settlement_price = self.activation_results["slot_settlement_prices_DE"][slot]
            
            agents_bid_price = self.activation_results["agents_bid_prices"][slot]

            distance_to_settlement_price = agents_bid_price - slot_settlement_price
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " distance_to_settlement_price = " + str(distance_to_settlement_price) )

            auction_reward = 1 - (distance_to_settlement_price / self.price_scaler.data_max_[0] )**0.4
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " auction_reward = " + str(auction_reward) )
            
            
        # If agent did not participate in auction, no hard negative reward, but we want to push him to participate in auction 
        if self.activation_results["slots_won"][slot] == 0:
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " agent did not participate in auction for slot no " + str(slot) )
            
            # get minimum possible VPP Capacity for the given slot 
            vpp_total_slot_min = min(self.activation_results["vpp_total"][slot *16 : (slot+1)*16])

            # IF we could have activated power (capacity was available), then create distance reward to possible capacity
            if vpp_total_slot_min > 0: 
                
                # as slot bid size from agent already was 0 , the distance is vpp_total_slot_min                   
                distance_to_vpp_capacity = vpp_total_slot_min
                
                # day reward based on distance to maximumm vpp capacity during slot divided by maximum vpp capacity overall. 
                # the greater the distance the lower the reward
                auction_reward = (1 - (distance_to_vpp_capacity / self.size_scaler.data_max_[0])**0.4) 
            
            # IF vpp_total_slot_min == 0 , so VPP wouldnt be able to activate any capacity the Agent was right and we reward him only with a distance reward for the price. 
            else: 
                slot_settlement_price = self.activation_results["slot_settlement_prices_DE"][slot]
                agents_bid_price = self.activation_results["agents_bid_prices"][slot]
                distance_to_settlement_price = agents_bid_price - slot_settlement_price
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " distance_to_settlement_price = " + str(distance_to_settlement_price) )

                # if distance_to_settlement_price is a negative number (occurs, when agent didnt participate in auction and price is smaller than settlement price)
                # then no negative reward based on price can be given, an alternative reward based on distance to possible vpp capacity is given.
                if distance_to_settlement_price < 0:
                    distance_to_settlement_price = 0
                    
                    # get minimum possible VPP Capacity for the given slot 
                    vpp_total_slot_min = min(self.activation_results["vpp_total"][slot *16 : (slot+1)*16])

                    # as slot bid size from agent already was 0 , the distance is vpp_total_slot_min                   
                    distance_to_vpp_capacity = vpp_total_slot_min
                    
                    # day reward based on distance to maximumm vpp capacity during slot divided by maximum vpp capacity overall. 
                    # the greater the distance the lower the reward
                    auction_reward = (1 - (distance_to_vpp_capacity / self.size_scaler.data_max_[0])**0.4) 
                
                # if price was higher than settlement price, a reward based on distance_to_settlement_price is given.
                else: 
                    auction_reward = 1 - (distance_to_settlement_price / self.price_scaler.data_max_[0])**0.4
                
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " auction_reward = " + str(auction_reward))
                
        # IF AGENT WON THE SLOT 
        if self.activation_results["slots_won"][slot] == 1:
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " slot no. " + str(slot)+  " was won" )

            # Approach 1 : first reward the won slot, then check if it could be activated and give huge negative reward (-1000)
            # Approach 2 : first check if won slot could be activated and then calculate partial reward (60 minutes - penalty minutes / 60 ) * price * size 
            # we try Approach 1 

            # Step 1: award the agent for a won slot
            auction_reward = 1
            
            # Step 2: Calculate the Profit of the bid if won 
            
            # extract the bid size of the agent 
            agents_bid_size = self.activation_results["agents_bid_sizes_round"][slot]
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " agents_bid_size: " + str(agents_bid_size))

            # and calculate the reward by multiplying the bid size with the settlement price of the slot
            basic_compensation = (agents_bid_size * self.activation_results["slot_settlement_prices_DE"][slot])
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " basic_compensation: " + str(basic_compensation))
            slot_revenue = basic_compensation
            
            # Step 3.1: simulate reservation: validate if the VPP can reserve the traded capacity
            simulate_reservation(self, slot)
            
            # IF RESERVATION WAS NOT SUCCESSFULL 
            if self.activation_results["reserved_slots"][slot] == -1:
                
                # Penalty Calculation from "MfRRA"
                # NO penalty calculation from Elia PDF: "20200317 TC BSP FCRFINAL ConsultEN.pdf"
                # ID AEP based on: https://www.regelleistung.net/ext/static/konsultation-aep?lang=de
                # based on :  Preisindex orientiert sich stark am Intraday-Viertelstundenhandel: https://www.next-kraftwerke.de/wissen/ausgleichsenergie#:~:text=Der%20Ausgleichsenergiepreis%20%E2%80%93%20%22reBAP%22%20(,auf%20die%20Verursacher%20der%20Regelenergie.
                # Marktdaten von: https://www.smard.de/home/marktdaten?marketDataAttributes=%7B%22resolution%22:%22month%22,%22region%22:%22Amprion%22,%22from%22:1589172592929,%22to%22:1654663792928,%22moduleIds%22:%5B8004169%5D,%22selectedCategory%22:17,%22activeChart%22:true,%22style%22:%22color%22,%22categoriesModuleOrder%22:%7B%7D%7D
                penalty_fee_1 = self.current_daily_mean_market_price * 1.25
                penalty_fee_2 = self.current_daily_mean_market_price + 10.0
                penalty_fee_3 = self.activation_results["slot_settlement_prices_DE"][slot]
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_1: " + str(penalty_fee_1))
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_2: " + str(penalty_fee_2) )
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_3: " + str(penalty_fee_3))
                penalty_list = [penalty_fee_1, penalty_fee_2, penalty_fee_3] 
                penalty_fee_reservation = self.activation_results["total_not_reserved_energy"][slot] * max(penalty_list) 
                logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " penalty_fee_reservation = " + str(penalty_fee_reservation))
                slot_penalty -= penalty_fee_reservation
                
                # REWARD
                # get minimum possible VPP Capacity for the given slot 
                vpp_total_slot_min = min(self.activation_results["vpp_total"][slot *16 : (slot+1)*16])
                # as agents_bid_size is higher than vpp_total_slot_min, substract vpp_total_slot_min from agents_bid_size to get distance      
                distance_to_vpp_capacity = agents_bid_size - vpp_total_slot_min                    
                # day reward based on distance to maximumm vpp capacity during slot divided by maximum vpp capacity overall. 
                # the greater the distance the lower the reward
                reservation_reward = (1 - (distance_to_vpp_capacity / self.size_scaler.data_max_[0])**0.4) 
                
            # IF RESERVATION IS SUCCESSFULL 
            if self.activation_results["reserved_slots"][slot] == 1:
                
                # give reward when capacity could be reserved
                reservation_reward = 1
                                    
                # Step 3.2: simulate activation: validate if the VPP can activate the traded capacity
                simulate_activation(self, slot)
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " self.activation_results['activated_slots']")
                logging.info(self.activation_results["activated_slots"])

                # Step 4: if the capacity can not be activated give a high Penalty
                if self.activation_results["activated_slots"][slot] == -1:
                    
                    # Penalty
                    penalty_fee_1 = self.current_daily_mean_market_price * 1.25
                    penalty_fee_2 = self.current_daily_mean_market_price + 10.0
                    penalty_fee_3 = self.activation_results["slot_settlement_prices_DE"][slot]
                    logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_1: " + str(penalty_fee_1))
                    logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_2: " + str(penalty_fee_2))
                    logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_3: " + str(penalty_fee_3))
                    penalty_list = [penalty_fee_1, penalty_fee_2, penalty_fee_3]
                    penalty_fee_activation = self.activation_results["total_not_activated_energy"][slot] * max(penalty_list) 
                    logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_activation = " + str(penalty_fee_activation))

                    slot_penalty -= penalty_fee_activation
                    
                    # Reward 
                    #activation_reward = 0
                    
                    # REWARD
                    positive_activation_possible_list = self.activation_results["positive_activation_possible_list"][slot] 
                    negative_activation_possible_list = self.activation_results["negative_activation_possible_list"][slot]
                    joined_activation_possible_lists  = positive_activation_possible_list + negative_activation_possible_list
                    activation_possible_count = sum(joined_activation_possible_lists)
                    
                    # day reward based on ratio between successfull activated 15 min steps (positive and negative FCR) and all 15 min steps. 
                    # the more steps were successfully activated, the higher the reward
                    activation_reward = (activation_possible_count / 16 )**0.4
                    
                if self.activation_results["activated_slots"][slot] == 1:
                    # give reward when capacity could be activated
                    activation_reward = 1
            
        slot_reward = auction_reward + reservation_reward + activation_reward
        # create weighted slot reward [0,1]
        weighted_slot_reward = slot_reward / 3
        day_reward += weighted_slot_reward
        
        if abs(slot_penalty) >= slot_revenue: 
            slot_profit = 0 
        else: 
            slot_profit = slot_revenue - abs(slot_penalty)
            
        day_revenue += slot_revenue
        day_penalties += slot_penalty
        day_profit += slot_profit
        
        logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " for slot no : " + str(slot))
        logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " self.activation_results['slot_settlement_prices_DE'][slot]: " + str(self.activation_results["slot_settlement_prices_DE"][slot]))
        logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " auction_reward: " + str(auction_reward))
        logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " reservation_reward: " + str(reservation_reward))
        logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " activation_reward: " + str(activation_reward))
        logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " slot_reward: " + str(slot_reward))
        logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " weighted_slot_reward (slot_reward/3) = " + str(weighted_slot_reward))
        logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " slot_profit: " + str(slot_profit))
    
    # create weighted day reward (Maximum of 1)
    weighted_day_reward = day_reward / 6
    
    logging.info("log_step: " + str(self.logging_step) + " day_reward (sum of all weighted_slot_reward ) = " + str(day_reward))
    logging.info("log_step: " + str(self.logging_step) + " weighted_day_reward (= day_reward / 6) for all 6 slots : " + str(weighted_day_reward))
    logging.info("log_step: " + str(self.logging_step) + " day_revenue (sum of all slot_revenue) = " + str(day_revenue))
    logging.info("log_step: " + str(self.logging_step) + " day_penalties (sum of all slot_penalty) = " + str(day_penalties))
    logging.info("log_step: " + str(self.logging_step) + " day_profit (sum of all slot_profit) = " + str(day_profit))

    return weighted_day_reward, day_revenue, day_penalties, day_profit
