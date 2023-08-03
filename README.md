# Deep Sequence Learning with Auxiliary Information for Traffic Prediction. KDD 2018. (Accepted)

### Binbing Liao, Jingqing Zhang, Chao Wu, Douglas McIlwraith, Tong Chen, Shengwen Yang, Yike Guo, Fei Wu

###### Binbing Liao and Jingqing Zhang contributed equally to this article. 

Paper Link: [arXiv](https://arxiv.org/abs/1806.07380) or [KDD18](http://www.kdd.org/kdd2018/accepted-papers/view/deep-sequence-learning-with-auxiliary-information-for-traffic-prediction)

## Contents

1. [Overview](#Overview)
2. [Code](#Code)

<h2 id="Overview">Overview</h2>
With the widespread adoption of electric vehicles (EVs), it is crucial to plan for EV charging in a way that considers both EV driver behavior and the electricity grid's demand. We integrate detailed mobility data with empirical charging preferences to estimate charging demand and demonstrate the power of personalized shifting recommendations to move individual EV drivers' demand on the electricity grid out of peak hours. We find an unbalanced geographical distribution of charging demand in the San Francisco Bay Area, with temporal peaks in both grid off-peak hours in the morning and grid on-peak hours in the evening. Our strategy effectively transfers demand to off-peak charging load, taking advantage of the mobility behavior. 

<p align="center">
  <img src="figures/fig2_geo_group.pdf" width="600">
  <br><i>Figure 1. Overview</i>
</p>

<h2 id="Code">Code</h2>

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
- [SimSensitivity](model/SimSensitivity.py): Conducting sensitivity analysis by changing parameters.
- [AnaMobility.py](analysis/SimMobility.py): Visualize mobility patterns.
- [AnaAdopter.py](analysis/SimAdopter.py): Visualize demographics of potential adopters.
- [AnaBehavior.py](analysis/SimBehavior.py): Visualize charging behavior.
- [AnaShift.py](analysis/SimShift.py): Visualize shifting recommendations' impacts.
- [AnaSensitivity](analysis/SimSensitivity.py): Visualize sensitivity analysis.

To run demo code for simulation and analysis:
- [model/RunMe.ipynb](model/RunMe.ipynb): Charging behavior and shifting recommendation simulation.
- [analysis/RunMe.ipynb](analysis/RunMe.ipynb): Analysis and visualization of simulation results.
