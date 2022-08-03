# README

## Title of Thesis

"Reinforcement Learning based Strategic Bidding in the Balancing Market with a Virtual Power Plant"

## Installation 

In your console: 

1. ```git clone https://github.com/jlp-ue/MA-VPP-RL-bidding MA-VPP-RL-bidding ```
2. ```cd MA-VPP-RL-bidding```
3. ```pip install virtualenv```      (if you don't already have virtualenv installed)
4. ```virtualenv venv```             to create your new environment (called 'venv' here)
5. ```source venv/bin/activate```    to enter the virtual environment
6. ```pip3 install -r requirements.txt``` to install the requirements in the current environment
7. ```jupyter lab```
8. Wait for jupyter to start and open "02_VPPBiddingEnv.ipynb"
9. Run Notebook (Weights and Biases Visualisation may not be shown)
10. Code for the environment definition is in https://github.com/jlp-ue/MA-VPP-RL-bidding/blob/main/vpp-gym/vpp_gym/envs/vpp_env.py
