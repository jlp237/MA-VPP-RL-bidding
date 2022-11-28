import logging


def simulate_reservation(self, slot: int):
    """_summary_

    Args:
        slot (_type_): _description_
    """
    logging.debug("log_step: " + str(self.logging_step) + " slot: " + str(slot) + " reservation Simulation for Slot No. " + str(slot))

    vpp_total_slot = self.delivery_results["vpp_total"][slot * 16 : (slot + 1) * 16]
    bid_sizes_per_slot = self.delivery_results["bid_sizes_all_slots"][slot * 16 : (slot + 1) * 16]

    reservation_possible_list = []
    total_not_reserved_energy = 0.0

    for time_step in range(0, 16):
        # for a 15 min time interval:
        # 1kwh/ 15min = * 4 =  4kw/15min Durchschnittsleistung
        # 4kw/15min = * 0,25 = 1kwh/15min Energie

        # 0     = 4 MW * 0,25h = 1 MWh / 15min
        # 15    = 3 MW * 0,25h = 0,75 MWh / 15min
        # 30    = 2 MW * 0,25h = 0,5 MWh / 15min
        # 45    = 1 MW * 0,25h = 0,25 MWh / 15min
        # Summe = 2,5 MWh

        agent_bid_size = bid_sizes_per_slot[time_step]
        available_vpp_capacity = vpp_total_slot[time_step]

        reservation_possible = False
        not_reserved_capacity = 0
        not_reserved_counter = 0
        not_reserved_energy = 0

        # 1. check negative reservation
        if (available_vpp_capacity - agent_bid_size) > 0:
            logging.debug("Negative reservation possible for slot " + str(slot) + "and timestep " + str(time_step))
            reservation_possible = True

        else:
            logging.debug("Negative reservation NOT possible for slot " + str(slot) + "and timestep " + str(time_step))
            not_reserved_counter += 1
            not_reserved_capacity += abs(available_vpp_capacity - agent_bid_size)
            not_reserved_energy = not_reserved_capacity * 0.25  # multiply power with time to get energy
            total_not_reserved_energy += not_reserved_energy

        # 2. check positive reservation
        if agent_bid_size < available_vpp_capacity:
            logging.debug("positive reservation possible for slot " + str(slot) + "and timestep " + str(time_step))
            reservation_possible = True
        else:
            # only calculate not_reserved_capacity if not already calculated for negative FCR
            if not_reserved_counter != 1:
                logging.debug("Positive reservation NOT possible (and negative reservation not given penalty yet) for slot " + str(slot) + "and timestep " + str(time_step))
                not_reserved_capacity += abs(agent_bid_size - available_vpp_capacity)
                not_reserved_energy = not_reserved_capacity * 0.25  # multiply power with time to get energy
                total_not_reserved_energy += not_reserved_energy

        reservation_possible_list.append(reservation_possible)

    if all(reservation_possible_list):
        self.delivery_results["reserved_slots"][slot] = 1
    else:
        self.delivery_results["reserved_slots"][slot] = -1
    logging.debug("total_not_reserved_energy for slot " + str(slot) + " is " + str(total_not_reserved_energy))
    self.delivery_results["total_not_reserved_energy"][slot] = total_not_reserved_energy
