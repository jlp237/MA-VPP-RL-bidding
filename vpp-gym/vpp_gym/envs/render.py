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
    
    #print("in render() and logging_step = " + str(self.logging_step))
    if not self.activation_results:
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + " self.activation_results is empty, not plotting it ")
    else:
        # only plot to wandb when not in test mode
        if self.env_type != "test":
            # if training step < 1 , (first training step) nothing will be rendered 
            if self.logging_step > 1: 
                logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  now in render()")        
                logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  self.previous_activation_results " + str(self.previous_activation_results))   
                logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  self.activation_results['slots_won'] " + str(self.activation_results["slots_won"]))   
                #print("self.previous_activation_results['slots_won'] in render() = " + str(self.previous_activation_results["slots_won"]))

                # Render Won / Lost Slots 
                slots_won = self.previous_activation_results["slots_won"]
                logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  slots_won " + str(slots_won))
                
                slots_won, slots_lost, slots_not_participated = list_generator(slots_won)
                
                won_data = {'Won': slots_won, 'Lost': slots_lost, 'not part.': slots_not_participated}
                won_df = pd.DataFrame(data=won_data, index=[1, 2, 3, 4, 5, 6])
                logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  won_df " + str(won_df))
                slots_won_plot = px.bar(won_df,  x= won_df.index, y=['Won', 'Lost', 'not part.'], color_discrete_sequence=[ "#186A3B", "#78281F", "gainsboro"] , labels={"index": "Slot", "value": "Won / Lost / not part."}, title="Won / Lost Slots")



                # Render Reservation for each Slot
                slots_reserved = self.previous_activation_results["reserved_slots"]
                logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  slots_reserved " + str(slots_reserved))
                
                slots_reserved, slots_not_reserved, slots_not_participated = list_generator(slots_reserved)
                
                reserved_data = {'Reserved': slots_reserved, 'NOT Reserv.': slots_not_reserved, 'lost/not part.': slots_not_participated}
                reserved_df = pd.DataFrame(data=reserved_data, index=[1, 2, 3, 4, 5, 6])
                logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  reserved_df " + str(reserved_df))
                reserved_plot = px.bar(reserved_df,  x=reserved_df.index, y=['Reserved', 'NOT Reserv.', 'lost/not part.'], color_discrete_sequence=[ "#239B56", "#B03A2E", "gainsboro"] , labels={"index": "Slot", "value": "Reserved, Not Reserv., lost/not part."}, title="Reservation per Slot")

               

                # Render Activation for each Slot 
                slots_delivered = self.previous_activation_results["delivered_slots"]
                logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  slots_delivered " + str(slots_delivered))
                
                slots_delivered, slots_not_delivered, slots_not_participated = list_generator(slots_delivered)
                
                delivered_data = {'Delivered': slots_delivered, 'NOT Deliv.': slots_not_delivered, 'not res.': slots_not_participated}
                delivered_df = pd.DataFrame(data=delivered_data, index=[1, 2, 3, 4, 5, 6])
                logging.debug("log_step: " + str(self.logging_step) + " slot: " +  "None"  + "  delivered_df " + str(delivered_df))
                delivered_plot = px.bar(delivered_df,  x= delivered_df.index, y=['Delivered', 'NOT Deliv.', 'not res.'], color_discrete_sequence=[ "#2ECC71", "#E74C3C", "gainsboro"] , labels={"index": "Slot", "value": "Delivered, Not Del., not reserv."}, title="Activation per Slot")


            
                # Render Agents Slot Prices and Settlement Prices
                price_plot = go.Figure()
                price_plot.add_trace(go.Scatter(x=list(range(1,7)), y=self.previous_activation_results["slot_settlement_prices_DE"], line_color="#3498DB", name="Market Price"))
                price_plot.add_trace(go.Scatter(x=list(range(1,7)), y=self.previous_activation_results["agents_bid_prices"] , line_color="#D35400", name="Agents Price" )) 
                price_plot.update_layout(title="Sold and Available Capacity", xaxis_title="Slots")


                # Render activation for Capacity 
                capacity_plot = go.Figure()
                capacity_plot.add_trace(go.Scatter(x=list(range(1, 97)), y=self.previous_activation_results["vpp_total"], fill='tozeroy', fillcolor='rgba(142, 68, 173 0.4)',  line_color="#8E44AD", name="VPP Cap."))
                capacity_plot.add_trace(go.Scatter(x=list(range(1, 97)), y=self.previous_activation_results["bid_sizes_all_slots"], fill='tozeroy', fillcolor='rgba(211, 84, 0, 0.5)', line_color="#D35400", name="Agents Bid" )) 
                capacity_plot.update_layout(title="Agents and Settlement Prices", xaxis_title="Timestep (15min)")


                if self.env_type != "test":
                    
                    if self.env_type == "training":
                        wandb.log({
                            "Won / Loss of Slots": slots_won_plot,
                            "Reservation per Slot": reserved_plot,
                            "Activation per Slot": delivered_plot,
                            "Sold and Available Capacity" : capacity_plot,
                            "Agents and Settlement Prices per Slot" : price_plot},
                            #step=self.logging_step,
                            commit=True
                        )
                        
                    if self.env_type == "eval":
                        
                        if self.render_mode == "human":
                            wandb.log({
                                "global_step": self.logging_step,
                                "Won / Loss of Slots": slots_won_plot,
                                "Reservation per Slot": reserved_plot,
                                "Activation per Slot": delivered_plot,
                                "Sold and Available Capacity" : capacity_plot,
                                "Agents and Settlement Prices per Slot" : price_plot},
                                #step=self.logging_step,
                                commit=False
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