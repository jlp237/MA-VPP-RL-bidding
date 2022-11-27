import logging
import random
from scipy.stats import norm
import numpy as np


def prepare_activation(self):
    '''
    
    '''
    
    # extend slot bid size format from 6 slots to 96 time steps
    bid_sizes_list = []
    for slot_x in range (0,6): 
        for time_step in range(0,16):
            #bid_sizes_list.append(action_dict["size"][slot_x])
            bid_sizes_list.append(self.activation_results["agents_bid_sizes_round"][slot_x])
    bid_sizes_all_slots = np.array(bid_sizes_list)
    self.activation_results["agents_bid_sizes_round_all_slots"] = bid_sizes_all_slots
    self.activation_results["bid_sizes_all_slots"] = bid_sizes_all_slots
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + " self.activation_results['bid_sizes_all_slots'] : "  + str(self.activation_results['bid_sizes_all_slots']))
    
    # initialize slots dict
    self.activation_results["reserved_slots"] = [0,0,0,0,0,0]
    self.activation_results["activated_slots"] = [0,0,0,0,0,0]
    self.activation_results["positive_activation_possible_list"] = [None, None, None, None, None, None]
    self.activation_results["negative_activation_possible_list"] = [None, None, None, None, None, None]
           
           
def check_activation_possible(self, agent_bid_size, vpp_total_step):
    """_summary_

    Args:
        agent_bid_size (_type_): _description_
        vpp_total_step (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # assumption: mean number of deliveries per hour: every 60 seconds = once every minute
    # assumption: mean activation length: 30 Seconds 
    # assumption: positive and negative : 50 % / 50% = per 15 minutes:  
        #  7.5 times * 30 Seconds positive FCR 
        #  7.5 times * 30 Seconds negative FCR 
        
    not_activated_capacity = 0
    not_activated_energy = 0
    propab_of_activation = 0.995
    activation_possible = None

    # 2. Probability of maximum activation Amount (67,5 % :  0 - 5 %  Capacity , 23 % : 5 - 10 % , 6% : 10-15% and so on 
    max_activation_share = random.choices(
            population=[0.05, 0.1,  0.15, 0.2,  0.25,  0.5,   0.75,   1.0],
            weights=[0.675, 0.23, 0.06, 0.02, 0.01, 0.002, 0.002, 0.001],
            k=1
        )
    capacity_to_activate = 0. 
    capacity_to_activate = max_activation_share[0] * agent_bid_size

    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 2: agent_bid_size : " + str(agent_bid_size))
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 2: max_activation_share : " + str(max_activation_share[0]))
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 2: capacity_to_activate : " + str(capacity_to_activate))
    
    # 3. Probability of successfull activation (100%: 10% of HPP Capacity = Probability Curve)

    '''
    ### First approach with complex norm distribution function 
    
    mean = 0. # symmetrical normal distribution at 0 
    sd = self.maximum_possible_VPP_capacity/7

    max_at_10_percent = norm.pdf((self.maximum_possible_VPP_capacity*0.1),mean,sd)
    scale_factor = 1 / max_at_10_percent

    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 3: max_at_10_percent = " + str(max_at_10_percent))
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 3: scale_factor = " + str(scale_factor))
    
    # Plot between -max_power and max_power with .001 steps.
    #x_axis = np.arange(-max_power, max_power, 0.001)
    #plt.plot(x_axis, (norm.pdf(x_axis, mean, sd)) * scale_factor + shift_to_100)
    #plt.show()
    propab_of_activation = round(float(norm.pdf(x=capacity_to_activate, loc=mean,scale=sd) * scale_factor),3)

    if propab_of_activation > 1.0: 
        propab_of_activation =  1.0

    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 3: propab_of_activation = " + str(propab_of_activation))

    activation_possible = random.choices(
            population=[True, False],
            weights=   [propab_of_activation , (1-propab_of_activation)],
            k=1
        )
    activation_possible = activation_possible[0] # as bool is in list 
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 3: activation_possible : " + str(activation_possible))
    '''
    
    ### Second approach: take reliability from paper
   
    # 3. Check VPP Boundaries: 
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " Check No. 3")

    # check if absolute capacity_to_activate is bigger than vpp_total_step
    # abs for negative and positive FCR
    if abs(capacity_to_activate) > vpp_total_step:
        if capacity_to_activate < 0: 
            logging.error("Check No. 3: Error, no negative FCR possible, vpp_total_step is small and cant do more negative FCR")
        else: 
            logging.error("Check No. 3: Error, no positive FCR possible, FCR is larger than vpp_total_step")
        # calculate not activated capacity for reward calcutlation
        not_activated_capacity = abs((vpp_total_step - abs(capacity_to_activate)))
        
        logging.error("Check No. 4: probability of activaiton with 0.5 percent")
        # Check No. 4: Probability of successfull activation
        # Activation is possible based on available capacity. 
        # Based on the paper of Seidel and Haase we decide finally if the activaiton is 
        # possible based on their researched probability distribution for a VPP constisting of HPPs
        available_capacity = vpp_total_step
        not_activated_capacity += available_capacity * propab_of_activation
        
    # If activation POSSIBLE as enough capacity is available:
    else: 
        # Activation is possible, but still, some capacity could not be fully provided
        activation_possible = True
        # for positive AND negative FCR
        not_activated_capacity = abs(capacity_to_activate) * propab_of_activation
        
    # not_activated_capacity = MW
    # assumption: positive and negative : 50 % / 50% = per 15 minutes:  
    #  7.5 times * 30 Seconds positive FCR 
    #  7.5 times * 30 Seconds negative FCR 
    # per pos / neg  
    # 7.5 Minuten * 0.5 Minuten = 3.75 minutes
    # 3.75 minutes / 60 minuten = 0.0625h 
    # OR 60 minuten / 3.75 minutes = 16
    # 100 *  0.0625 # OR 
    # 100 /  16
    not_activated_energy = not_activated_capacity *  0.0625  # multiply power with time to get energy 

    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 4: activation_possible : " + str(activation_possible))
    logging.error("Check No. 4: not_activated_capacity = " + str(not_activated_capacity))
    logging.error("Check No. 4: not_activated_energy = " + str(not_activated_energy))
    
    return activation_possible, not_activated_energy


def simulate_activation(self, slot): 
    """_summary_

    Args:
        slot (_type_): _description_
    """    
    
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " activation Simulation for Slot No. " + str(slot))
    
    vpp_total_slot = self.activation_results["vpp_total"][slot *16 : (slot+1)*16]
    bid_sizes_per_slot = self.activation_results["bid_sizes_all_slots"][slot *16 : (slot+1)*16]
    
    #logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " vpp_total_FCR_slot " + str(vpp_total_FCR_slot))
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " vpp_total_slot " + str(vpp_total_slot))
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " bid_sizes_per_slot " + str(bid_sizes_per_slot))

    activation_possible = None
    positive_activation_possible_list = []
    negative_activation_possible_list = []
    total_not_activated_energy = 0.

    # check for every timestep
    for time_step in range(0, 16):
    
        agent_bid_size = bid_sizes_per_slot[time_step]
        
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " vpp_total_slot[time_step] for  time_step =" + str(time_step) + " : " + str(vpp_total_slot[time_step]))
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " bid_sizes_per_slot[time_step] for time_step ="  + str(time_step) + " :  " + str(bid_sizes_per_slot[time_step]))

        # check if positive FCR could be provided 
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " check positive FCR")
        activation_possible, not_activated_energy = check_activation_possible(self, agent_bid_size, vpp_total_slot[time_step])
        positive_activation_possible_list.append(activation_possible)
        total_not_activated_energy += not_activated_energy

        # check if negative FCR could be provided 
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " check negative FCR")
        activation_possible, not_activated_energy = check_activation_possible(self, (-agent_bid_size), vpp_total_slot[time_step])
        negative_activation_possible_list.append(activation_possible)
        total_not_activated_energy += not_activated_energy

    if all(positive_activation_possible_list) and all(negative_activation_possible_list): 
        self.activation_results["activated_slots"][slot] = 1
        total_activation_possible = True
    else: 
        self.activation_results["activated_slots"][slot] = -1
        total_activation_possible = False
        
    self.activation_results["total_not_activated_energy"][slot] = total_not_activated_energy
    self.activation_results["positive_activation_possible_list"][slot] = positive_activation_possible_list
    self.activation_results["negative_activation_possible_list"][slot] = negative_activation_possible_list
    
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " total_activation_possible for slot " + str(slot) + " : " + str(total_activation_possible))
    