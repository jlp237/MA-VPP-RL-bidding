# Logging
import logging
import wandb

# Plotting
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
from typing import List


def render(self, mode="human"):
    """_summary_
    """    
    
    # only plot to wandb when not in test mode
    if self.env_type != "test":
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  now in render()")        
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  self.activation_results['slots_won'] " + str(self.activation_results["slots_won"]))   

        # Render Won / Lost Slots 
        slots_won = self.activation_results["slots_won"]
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  slots_won " + str(slots_won))
        slots_won, slots_lost, slots_not_participated = list_generator(slots_won)
        won_data = {'Won': slots_won, 'Lost': slots_lost, 'not part.': slots_not_participated}
        won_df = pd.DataFrame(data=won_data, index=[1, 2, 3, 4, 5, 6])
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  won_df " + str(won_df))
        slots_won_plot = px.bar(won_df,  x= won_df.index, y=['Won', 'Lost', 'not part.'], color_discrete_sequence=[ "#186A3B", "#78281F", "gainsboro"] , labels={"index": "Slot", "value": "Auction Result"}, title="Won / Lost Slots", width=300, height=400)
        slots_won_plot.update_layout(legend=dict(orientation="h",yanchor="bottom", y=-0.6, xanchor="center", x=0.3, title=""))

        # Render Reservation for each Slot
        slots_reserved = self.activation_results["reserved_slots"]
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  slots_reserved " + str(slots_reserved))
        slots_reserved, slots_not_reserved, slots_not_participated = list_generator(slots_reserved)
        reserved_data = {'Reserved': slots_reserved, 'Not Reserv.': slots_not_reserved, 'lost/not part.': slots_not_participated}
        reserved_df = pd.DataFrame(data=reserved_data, index=[1, 2, 3, 4, 5, 6])
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  reserved_df " + str(reserved_df))
        reserved_plot = px.bar(reserved_df,  x=reserved_df.index, y=['Reserved', 'Not Reserv.', 'lost/not part.'], color_discrete_sequence=[ "#239B56", "#B03A2E", "gainsboro"] , labels={"index": "Slot", "value": "Reservation"}, title="Reservation per Slot", width=300, height=400)
        reserved_plot.update_layout(legend=dict(orientation="h",yanchor="bottom", y=-0.6, xanchor="center", x=0.3, title=""))

        # Render Activation for each Slot 
        slots_activated = self.activation_results["activated_slots"]
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  slots_activated " + str(slots_activated))
        slots_activated, slots_not_activated, slots_not_participated = list_generator(slots_activated)
        activated_data = {'Activated': slots_activated, 'Not Activ.': slots_not_activated, 'lost/not part./ not res.': slots_not_participated}
        activated_df = pd.DataFrame(data=activated_data, index=[1, 2, 3, 4, 5, 6])
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  activated_df " + str(activated_df))
        activated_plot = px.bar(activated_df,  x= activated_df.index, y=['Activated', 'Not Activ.', 'lost/not part./ not res.'], color_discrete_sequence=[ "#2ECC71", "#E74C3C", "gainsboro"] , labels={"index": "Slot", "value": "Activation"}, title="Activation per Slot", width=300, height=400)
        activated_plot.update_layout(legend=dict(orientation="h",yanchor="bottom", y=-0.6, xanchor="center", x=0.3, title=""))
        
        # Render Agents Slot Prices and Settlement Prices
        price_plot = go.Figure()
        price_plot.add_trace(go.Scatter(x=list(range(1,7)), y=self.activation_results["slot_settlement_prices_DE"], line_color="#3498DB", name="Market Price"))
        price_plot.add_trace(go.Scatter(x=list(range(1,7)), y=self.activation_results["agents_bid_prices"] , line_color="#D35400", name="Agents Price" )) 
        price_plot.update_layout(width=300, height=400, title="Agents and Settlement Prices", xaxis_title="Slots", legend=dict(orientation="h",yanchor="bottom", y=-0.6, xanchor="center", x=0.3, title=""))

        # Render activation for Capacity 
        capacity_plot = go.Figure()
        capacity_plot.add_trace(go.Scatter(x=list(range(1, 97)), y=self.activation_results["vpp_total"], fill='tozeroy', fillcolor='rgba(142, 68, 173 0.4)',  line_color="#8E44AD", name="VPP Cap."))
        capacity_plot.add_trace(go.Scatter(x=list(range(1, 97)), y=self.activation_results["bid_sizes_all_slots"], fill='tozeroy', fillcolor='rgba(211, 84, 0, 0.5)', line_color="#D35400", name="Agents Bid" )) 
        capacity_plot.update_layout(width=300, height=400, title="Sold and Available Capacity", xaxis_title="4h-Slot (15min Res.)", legend=dict(orientation="h",yanchor="bottom", y=-0.6, xanchor="center", x=0.3, title=""))

        if self.env_type != "test":
            if (self.env_type in ["training", "eval" ] and (self.render_mode == "human")):
                wandb.log({
                    "01. Won / Loss of Slots": slots_won_plot,
                    "02. Reservation per Slot": reserved_plot,
                    "03. Activation per Slot": activated_plot,
                    "04. Sold and Avail. Capacity" : capacity_plot,
                    "05. Agents and Market Price" : price_plot},
                    commit=True
                )


def list_generator(list_true): 
    """_summary_

    Args:
        list_true (_type_): _description_

    Returns:
        _type_: _description_
    """    
    
    list_false: List[int] = [0,0,0,0,0,0]
    list_none: List[int] = [0,0,0,0,0,0]
    
    for slot in range(6):
        if list_true[slot] == 1:
            list_false[slot] = 0
            list_none[slot] = 0
        elif list_true[slot] == 0:
            list_false[slot] = 0
            list_true[slot]  = 0
            list_none[slot] = 1
        elif list_true[slot] == -1:
            list_false[slot] = 1
            list_none[slot] = 0
            list_true[slot] = 0
            
    return list_true, list_false, list_none