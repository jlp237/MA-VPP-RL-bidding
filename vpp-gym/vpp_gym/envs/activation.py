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
    self.activation_results["delivered_slots"] = [0,0,0,0,0,0]
    self.activation_results["positive_activation_possible_list"] = [None, None, None, None, None, None]
    self.activation_results["negative_activation_possible_list"] = [None, None, None, None, None, None]
    #self.activation_results["not_delivered_capacity"] = [None,None,None,None,None,None]
           
           
def check_activation_possible(self, agent_bid_size, vpp_total_step):
    """_summary_

    Args:
        agent_bid_size (_type_): _description_
        vpp_total_step (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # assumption: mean activation length: 30 Seconds 
    # assumption: mean number of deliveries per hour: every 60 seconds = once every minute
    # assumption: positive and negative : 50 % / 50% = per 15 minutes:  
        #  7,5 times * 30 Seconds positive FCR 
        #  7,5 times * 30 Seconds negative FCR 
        
    not_delivered_capacity = 0
    not_delivered_energy = 0

    # 2. Probability of maximum activation Amount (74 % :  0 - 5 %  Capacity , 18 % : 5 - 10 % , 5% : 10-15% ( 97%: max 15 %)

    max_activation_share = random.choices(
            population=[0.05, 0.1,  0.15, 0.2,  0.25,  0.5,   0.75,   1.0],
            weights=   [0.74, 0.18, 0.05, 0.02, 0.007, 0.001, 0.001, 0.001],
            k=1
        )

    capacity_to_deliver = max_activation_share[0] * agent_bid_size

    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 2: agent_bid_size : " + str(agent_bid_size))
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 2: max_activation_share : " + str(max_activation_share[0]))
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 2: capacity_to_deliver : " + str(capacity_to_deliver))
    
    # 3. Probability of successfull activation (100%: 10% of HPP Capacity = Probability Curve)

    mean = 0 # symmetrical normal distribution at 0 
    sd = self.maximum_possible_VPP_capacity/7

    max_at_10_percent = norm.pdf(self.maximum_possible_VPP_capacity*0.1,mean,sd)
    scale_factor = 1 / max_at_10_percent

    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 3: max_at_10_percent = " + str(max_at_10_percent))
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 3: scale_factor = " + str(scale_factor))
    
    # Plot between -max_power and max_power with .001 steps.
    #x_axis = np.arange(-max_power, max_power, 0.001)
    #plt.plot(x_axis, (norm.pdf(x_axis, mean, sd)) * scale_factor + shift_to_100)
    #plt.show()

    propab_of_activation = round((norm.pdf(capacity_to_deliver, mean,sd) * scale_factor),3)

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
    
    # 4. Check VPP Boundaries: In case of a very high or low operating point (nearly 100% or 0% power output of the HPP): 
    # then the activation is not possible. 
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " Check No. 4")
    # check if probability of activation is high and capacity could be deliverd
    if activation_possible == True:
        # when negative FCR : 
        if capacity_to_deliver < 0:
            # check if capacity_to_deliver is bigger than vpp_total_step
            if (vpp_total_step - abs(capacity_to_deliver)) < 0:
                logging.error("Check No. 4: Error, vpp_total_step is small and cant do more negative FCR")
                not_delivered_capacity += abs((vpp_total_step - abs(capacity_to_deliver)))
                # not_delivered_capacity = MW
                # 0.5/60 = 30Seconds 
                # multiplied = MWh
                # 7.5 = only for negative FCR = half of 15minutes , other half for positive FCR.  
                not_delivered_energy = 7.5 * (not_delivered_capacity * 0.5/60)  # multiply power with time to get energy 

                activation_possible = False
        # if positive FCR
        else:
            # positive FCR is already included in vpp_total_step, so it should be possible
            '''# if the plant is already at maximum limit, then it cant produce more
            if (vpp_total_step + capacity_to_deliver) > self.maximum_possible_VPP_capacity: 
                logging.error("Error, FCR is larger than maximum_possible_VPP_capacity")
                not_delivered_capacity
                activation_possible = False'''
            # if the plant cant produce any power then positive FCR is also not possible
            if capacity_to_deliver >= vpp_total_step:
                logging.error("Check No. 4: Error, FCR is larger than vpp_total_step")
                not_delivered_capacity += capacity_to_deliver - vpp_total_step
                logging.error("Check No. 4: not_delivered_capacity = " + str(not_delivered_capacity))
                # not_delivered_capacity = MW
                # 0.5/60 = 30Seconds 
                # multiplied = MWh
                # 7.5 = only for positive FCR = half of 15minutes , other half for negative FCR.  
                not_delivered_energy = 7.5 * (not_delivered_capacity * 0.5/60)  # multiply power with time to get energy 

                activation_possible = False
                
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"   + " check No. 4: activation_possible : " + str(activation_possible))
        
    return activation_possible, not_delivered_energy


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
    total_not_delivered_energy = 0.

    # check for every timestep
    for time_step in range(0, 16):
    
        agent_bid_size = bid_sizes_per_slot[time_step]
        
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " vpp_total_slot[time_step] for  time_step =" + str(time_step) + " : " + str(vpp_total_slot[time_step]))
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " bid_sizes_per_slot[time_step] for time_step ="  + str(time_step) + " :  " + str(bid_sizes_per_slot[time_step]))

        # check if positive FCR could be provided 
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " check positive FCR")
        activation_possible, not_delivered_energy = check_activation_possible(self, agent_bid_size, vpp_total_slot[time_step])
        positive_activation_possible_list.append(activation_possible)
        if activation_possible == False:
                total_not_delivered_energy += not_delivered_energy

        # check if negative FCR could be provided 
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " check negative FCR")
        activation_possible, not_delivered_energy = check_activation_possible(self, (-agent_bid_size), vpp_total_slot[time_step])
        negative_activation_possible_list.append(activation_possible)
        if activation_possible == False:
            total_not_delivered_energy += not_delivered_energy

    if all(positive_activation_possible_list) and all(negative_activation_possible_list): 
        self.activation_results["delivered_slots"][slot] = 1
        total_activation_possible = True
    else: 
        self.activation_results["delivered_slots"][slot] = -1
        total_activation_possible = False
        
    self.activation_results["total_not_delivered_energy"][slot] = total_not_delivered_energy
    self.activation_results["positive_activation_possible_list"][slot] = positive_activation_possible_list
    self.activation_results["negative_activation_possible_list"][slot] = negative_activation_possible_list
    
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  str(slot)   + " total_activation_possible for slot " + str(slot) + " : " + str(total_activation_possible))
