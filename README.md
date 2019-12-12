# DQBirds

DQ-Birds team AI Agent based on Double Dueling Deep Q-Network presented on AIBirds 2018, IJCAI.

Original Paper: https://aibirds.org/2018/DQ-birds.pdf

# Description

Double Dueling Deep Q-network to play Angry Birds game.

![Alt text](/ddqn.PNG "Architecture of neural network")

This agent takes a screenshot from specified folder and feeds it as input to the Double Dueling Deep Q-network. 
The predicted action then is placed into txt file to the specified folder.

![Alt text](/imgp.PNG "Image pre-processing")


# How to Run

Agent consists of the two parts Java agent and Python agent. To run agent execute the following steps:
1. Start server `java –jar ABServer.jar`
2. Start Java agent by running `ClientNaiveAgent.java`
3. Start Python agent 

`C:\<path_to_project>\Angry Birds AI> activate tensorflow`

`(tensorflow) C:\<path_to_project>\Angry Birds AI> python ddqn_agent.py`

4. Click 'start' in the server window.
5. Agent should run now.
