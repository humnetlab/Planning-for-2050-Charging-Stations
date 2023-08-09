# Planning for 2050: Charging stations to support flexible electric vehicle demand considering individual mobility patterns

#### Jiaman Wu, Siobhan Powell, Yanyan Xu, Ram Rajagopal, Marta C. Gonzalez

## What's this?
With the widespread adoption of electric vehicles (EVs), it is crucial to plan for EV charging in a way that considers both EV driver behavior and the electricity grid's demand. We integrate detailed mobility data with empirical charging preferences to estimate charging demand and demonstrate the power of personalized shifting recommendations to move individual EV drivers' demand on the electricity grid out of peak hours. We find an unbalanced geographical distribution of charging demand in the San Francisco Bay Area, with temporal peaks in both grid off-peak hours in the morning and grid on-peak hours in the evening. Our strategy effectively transfers demand to off-peak charging load, taking advantage of the mobility behavior. 

## Contents

1. [Overview](#Overview)
2. [Dataset](#Dataset)
3. [Method](#Method)
4. [Setup](#Setup)

<h2 id="Overview">Overview</h2>
Overview of the proposed framework for understanding and planning future EV charging needs. (a) We analyze the current charging demand by extracting residents’ travel behavior and individual features, including visiting places and time, energy consumption, income, house type, and charging access, to sample potential EV adopters and assign them a charging behavior group. Based on that, we simulate all EV adopters’ charging behavior in a week, this includes charging location, session start and end time, energy, and power level. We propose personalized shifting recommendations to mitigate the impact of EV charging on grid peak hours. For example, EV adopters may shift their charging sessions from day 1 peak hour to day 2 off-peak hour when feasible. (b) Supply-side management means planning for infrastructure capacity at the ZIP code level, considering demand both before and after the proposed personalized shifting recommendations. (c) Future scenarios capture the evolution of EV adopters’ demographic features, charging demand, and the public charging station supply for increasing adoption rates.
<br/>
<br/>
<p align="center">
  <img src="figures/fig1_overview.jpg" width="900">
  <br><i>Figure 1. Overview</i>
</p>

<h2 id="Dataset">Dataset</h2>
We use different datasets in this study: call detail records (CDRs), charging session records, charging infrastructure data, and survey data such as US Census Bureau American Community Survey, the California Plug-in Electric Vehicle Adopter Survey, the California Home Charging Access Survey, and the Clean Vehicle Rebate Project (CVRP) data. The charging data and nobility data used in this study cannot be made publicly available due to privacy concerns for individual users.
<br/>
<br/>

|                          Name                          	| Geograhical<br>Coverage 	| Temporal<br>Coverage 	| Geographical<br>Resolution 	| Temporal<br>Resolution 	| Aggregated<br>Level 	|
|:------------------------------------------------------:	|:-----------------------:	|:--------------------:	|:--------------------------:	|:----------------------:	|:-------------------:	|
|                   Call Detail Records                  	|         Bay Area        	|         2013         	|           Lat,Lon          	|         10-min         	|      Individual     	|
|          [Charging Session Records](https://www.chargepoint.com)            	|         Bay Area        	|         2019         	|          ZIP code          	|                        	|      Individual     	|
|                  [Charging Station Data](https://github.com/openchargemap)                 	|        Worldwide        	|         ~2023        	|           Lat,Lon          	|           Day          	|      Individual     	|
|                    [SPECCh Model Data](https://data.mendeley.com/datasets/y872vhtfrc/2)                   	|           WECC          	|         ~2022        	|            WECC            	|          1-min         	|        Group        	|
|    [California Clean Vehicle <br>Rebate Project data](https://cleanvehiclerebate.org/en)    	|        California       	|         ~2023        	|          ZIP code          	|           Day          	|      Individual     	|
|       [Census Bureau American <br>Community Survey](https://data.census.gov/)      	|      United States      	|         ~2022        	|        Census Tract        	|          Year          	|     Census Tract    	|
| [California Plug-in Electric <br>Vehicle Adopter Survey](https://energycenter.org/sites/default/files/docs/nav/transportation/cvrp/survey-results/California_Plug-in_Electric_Vehicle_Driver_Survey_Results-May_2013.pdf) 	|        California       	|         2013         	|         California         	|          Year          	|      California     	|
|       [California Home Charging <br>Access Survey](https://www.energy.ca.gov/publications/2022/home-charging-access-california)       	|        California       	|         2022         	|         California         	|          Year          	|      California     	|


<h2 id="Method">Method</h2>

The methodology includes four parts. We first use [TimeGeo](https://www.pnas.org/doi/10.1073/pnas.1524261113) model to estimate the travel behavior and energy consumption of each vehicle in the sample. Second, we connect the travel behavior and [SPEECh](https://github.com/SiobhanPowell/speech) model by energy consumption and charging access to obtain the original charging behavior of EV adopters. Third, we identify the feasibility of drivers moving their original sessions from peak hours to off-peak hours by checking several rules. Last, we use a Bayesian model to estimate the probability of each driver adopting an EV based on their income and travel distance. Figure 2 depicts the connection between the data source and models.
<br/>
<br/>
<p align="center">
  <img src="figures/sp_method.jpg" width="900">
  <br><i>Figure 2. Method</i>
</p>

<h2 id="Setup">Setup</h2>

### Installations
In order to install all the required files, create a virtual environment and install the files given in `requirements.txt` file.

```
pip install -r requirements.txt
```

### Running the demo scripts
The structure of code:
- [SimMobility.py](model/SimMobility.py): Mobility simulation based on TimeGeo outputs.
- [SimAdopter.py](model/SimAdopter.py): Simulate demographics of potential adopters, also estimate the probability of being an EV adopter.
- [SimBehavior.py](model/SimBehavior.py): Simulate charging behavior.
- [SimShift.py](model/SimShift.py): Simulate shifting recommendations.
- [SimSensitivity.py](model/SimSensitivity.py): Conductsensitivity analysis by changing parameters.
- [SimSpeechGroup.py](model/SimSpeechGroup.py): Calculate conditional probability P(SPEECh group | Home and/or Work charging access, Annual energy).
- [SimSpeechValidation.py](model/SimSpeechValidation.py): Prepare validation files from the speech charging data. 
- [AnaMobility.py](analysis/AnaMobility.py): Visualize mobility patterns.
- [AnaAdopter.py](analysis/AnaAdopter.py): Visualize demographics of potential adopters.
- [AnaBehavior.py](analysis/AnaBehavior.py): Visualize charging behavior.
- [AnaShift.py](analysis/AnaShift.py): Visualize shifting recommendations' impacts.
- [AnaSensitivity](analysis/AnaSensitivity.py): Visualize sensitivity analysis.

To run demo code for simulation and analysis:
- [model/RunMe.ipynb](model/RunMe.ipynb): Charging behavior and shifting recommendation simulation.
- [analysis/RunMe.ipynb](analysis/RunMe.ipynb): Analysis and visualization of simulation results.
