import random


def check_activation_possible(agent_bid_size, vpp_total_step):

    not_delivered_capacity = 0
    not_delivered_energy = 0
    propab_of_activation = 0.995
    activation_possible = None
    # 2. Probability of maximum activation Amount
    # (67,5 % :  0 - 5 %  Capacity , 23 % : 5 - 10 % , 6% : 10-15% and so on
    max_activation_share = random.choices(
        population=[0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0],
        weights=[0.675, 0.23, 0.06, 0.02, 0.01, 0.002, 0.002, 0.001],
        k=1)
    capacity_to_activate = 0.0
    capacity_to_activate = max_activation_share[0] * agent_bid_size
    # 3. Check VPP Boundaries:
    # check if absolute capacity_to_activate is bigger than vpp_total_step
    # abs for negative and positive FCR
    if abs(capacity_to_activate) > vpp_total_step:
        not_delivered_capacity = abs(
            (vpp_total_step - abs(capacity_to_activate))
        )
        not_delivered_capacity += vpp_total_step * (
            1 - propab_of_activation
        )
    # If activation POSSIBLE as enough capacity is available:
    else:
        # Activation is possible, but still, some capacity could not be fully provided
        activation_possible = True
        # for positive AND negative FCR
        not_delivered_capacity = abs(capacity_to_activate) * (
            1 - propab_of_activation
        )
    # assumption: positive and negative : 50 % / 50% = per 15 minutes:
    # 7.5 times * 30 Seconds positive FCR = 3.75 minutes
    # 7.5 times * 30 Seconds negative FCR = 3.75 minutes
    # 3.75 minutes / 60 minuten = 0.0625h
    not_delivered_energy = (
        not_delivered_capacity * 0.0625
    )  # multiply power with time to get energy
    return activation_possible, not_delivered_energy


def simulate_activation(slot, vpp_total_slot, bid_sizes_per_slot):
    activation_results = [0] * 6
    activation_possible = None
    positive_activation_possible_list = []
    negative_activation_possible_list = []
    total_not_delivered_energy = 0.0
    # check for every timestep
    for time_step in range(0, 16):
        agent_bid_size = bid_sizes_per_slot[time_step]
        # check if positive FCR could be provided
        (
            activation_possible,
            not_delivered_energy,
        ) = check_activation_possible(
            agent_bid_size, vpp_total_slot[time_step]
        )
        positive_activation_possible_list.append(
            activation_possible
        )
        total_not_delivered_energy += not_delivered_energy
        # check if negative FCR could be provided
        (
            activation_possible,
            not_delivered_energy,
        ) = check_activation_possible(
            (-agent_bid_size), vpp_total_slot[time_step]
        )
        negative_activation_possible_list.append(
            activation_possible
        )
        total_not_delivered_energy += not_delivered_energy

    if all(positive_activation_possible_list) and all(
        negative_activation_possible_list
    ):
        activation_results[slot] = 1
    else:
        activation_results[slot] = -1

    return (
        total_not_delivered_energy,
        positive_activation_possible_list,
        negative_activation_possible_list,
    )
