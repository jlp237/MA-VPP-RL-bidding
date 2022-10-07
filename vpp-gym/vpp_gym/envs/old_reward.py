import logging

def _calculate_reward_OLD(self):        
    # Step 1 of Reward Function: The Auction
    # did the agent win the auction? 
    # what was the revenue ?
    
    self.activation_results["total_not_reserved_energy"] = [0, 0, 0, 0, 0, 0]
    self.activation_results["total_not_delivered_energy"] = [0, 0, 0, 0, 0, 0]

    step_reward = 0
    step_profit = 0
    
    # per slot won: + 100
    # per slot won: + (bid size *  marginal prize)
    # per slot lost: -100

    logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " Reward Overview:")
    logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " self.activation_results['slots_won']: " + str(self.activation_results["slots_won"]))
    logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " len(self.activation_results['slots_won']) : "  + str(len(self.activation_results["slots_won"]))) 
    
    for slot in range(0, len(self.activation_results["slots_won"])):
        
        logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " slot no. " + str(slot))
        
        if self.activation_results["slots_won"][slot] == 0:
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " slot no " + str(slot) + " was lost")
            step_reward -= 1000

        if self.activation_results["slots_won"][slot] == 1:
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " slot no. " + str(slot)+  " was won")

            # Approach 1 : first reward the won slot, then check if it could be activated and give huge negative reward (-1000)
            # Approach 2 : first check if won slot could be activated and then calculate partial reward (60 minutes - penalty minutes / 60 ) * price * size 
            # we try Approach 1 

            # Step 1: award the agent for a won slot
            step_reward += 10000
            
            # Step 2: Calculate the Profit of the bid if won 
            
            # extract the bid size of the agent 
            agents_bid_size = self.activation_results["agents_bid_sizes_round"][slot]
            # and calculate the reward by multiplying the bid size with the settlement price of the slot
            basic_compensation = (agents_bid_size * self.activation_results["slot_settlement_prices_DE"][slot])
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " basic_compensation: " + str(basic_compensation))
            step_reward += basic_compensation
            step_profit += basic_compensation
            
            # Step 3.1: simulate reservation: validate if the VPP can reserve the traded capacity
            total_reservation_possible = self._simulate_reservation(slot)
            
            if total_reservation_possible == False:                    
                # Penalty Calculation from "MfRRA"
                # NO penalty calculation from Elia PDF: "20200317 TC BSP FCRFINAL ConsultEN.pdf"
                # ID AEP based on: https://www.regelleistung.net/ext/static/konsultation-aep?lang=de
                # based on :  Preisindex orientiert sich stark am Intraday-Viertelstundenhandel: https://www.next-kraftwerke.de/wissen/ausgleichsenergie#:~:text=Der%20Ausgleichsenergiepreis%20%E2%80%93%20%22reBAP%22%20(,auf%20die%20Verursacher%20der%20Regelenergie.
                # Marktdaten von: https://www.smard.de/home/marktdaten?marketDataAttributes=%7B%22resolution%22:%22month%22,%22region%22:%22Amprion%22,%22from%22:1589172592929,%22to%22:1654663792928,%22moduleIds%22:%5B8004169%5D,%22selectedCategory%22:17,%22activeChart%22:true,%22style%22:%22color%22,%22categoriesModuleOrder%22:%7B%7D%7D
                penalty_fee_1 = self.current_daily_mean_market_price * 1.25
                penalty_fee_2 = self.current_daily_mean_market_price + 10.0
                penalty_fee_3 = self.activation_results["slot_settlement_prices_DE"][slot]
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_1: " + str(penalty_fee_1))
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_2: " + str(penalty_fee_2))
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_3: " + str(penalty_fee_3))
                penalty_list = [penalty_fee_1, penalty_fee_2, penalty_fee_3] 
            
                penalty_fee_reservation = self.activation_results["total_not_reserved_energy"][slot] * max(penalty_list) 
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_reservation = " + str(penalty_fee_reservation))
                step_reward -= penalty_fee_reservation
                step_profit -= penalty_fee_reservation 
            else: 
                # give reward when capacity could be reserved
                step_reward += 10000
                                    
            # Step 3.2: simulate activation: validate if the VPP can deliver the traded capacity
            self._simulate_activation(slot)
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " self.activation_results['delivered_slots']")
            logging.info(self.activation_results["delivered_slots"])

            # Step 4: if the capacity can not be delivered give a high Penalty
            if self.activation_results["delivered_slots"][slot] == False:
                penalty_fee_1 = self.current_daily_mean_market_price * 1.25
                penalty_fee_2 = self.current_daily_mean_market_price + 10.0
                penalty_fee_3 = self.activation_results["slot_settlement_prices_DE"][slot]
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_1: " + str(penalty_fee_1))
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_2: " + str(penalty_fee_2))
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_3: " + str(penalty_fee_3))
                penalty_list = [penalty_fee_1, penalty_fee_2, penalty_fee_3] 
            
                penalty_fee_activation = self.activation_results["total_not_delivered_energy"][slot] * max(penalty_list) 
                logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " penalty_fee_activation = " + str(penalty_fee_activation))
                step_reward -= penalty_fee_activation
                step_profit -= penalty_fee_activation
                #step_reward -= 5000
            else:
                # give reward when capacity could be activated
                step_reward += 10000

            
            # Update the total profit and Step Reward. 
            self._update_profit(step_profit)
            #step_reward +=  step_profit
            
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " agents_bid_size: " + str(agents_bid_size))
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " self.activation_results['slot_settlement_prices_DE'][slot]: " + str(self.activation_results["slot_settlement_prices_DE"][slot]))
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " step_profit: " + str(step_profit))
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " step_reward Slot " + str(slot) +" = " + str(step_reward))
                    
    # further rewards? 
    # diff to the settlement price
    # diff to the max. forecasted capacity of the VPP
    # incentive to go nearer to settlement price or forecasted capacity can be: 1- (abs(diff_to_capacity)/max_diff_to_capacity)^0.5
    # idea: reward for positive and negative reward separate. 
    
    # Alternative solution: 
    # A reward function, that combines penalty and delivered FCR: 
    # compensation = (60 minutes - penalty minutes / 60 ) * price * size 
    # penalty  = (penalty minutes / 60 ) * price * size 
    # reputation_damage = reputation_factor *  penalty_min/ 60 * size
        # penalty_min = number of minutes where capacity could not be provided
    # in total: r = compensation − penalty − reputation_damage,
    logging.info("log_step: " + str(self.logging_step) + " slot: " +  str(slot) + " total step_reward for all 6 slots : " + str(step_reward))

    return step_reward, step_profit