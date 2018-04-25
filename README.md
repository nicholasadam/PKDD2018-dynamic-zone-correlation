# PKDD2018-dynamic-zone-correlation
# Description

This package is created to dynamic travel demand estimation as described in paper "Discovering Urban Travel Demands through
Dynamic Zone Correlation in Location-based Social Networks", which is submitted to PKDD2018. In this paper, we aim to exploit the rich check-in data to model dynamic travel demands in urban areas, which can support a wide variety of mobile business solutions. Specically, we first profile the functionality of city zones using the categorical density of POIs. Second, we use a Hawkes Process-based State-Space formulation to model the dynamic trip arrival patterns based on check-in arrival patterns. Third, we developed a joint model that integrates Pearson Product-Moment Correlation (PPMC) analysis into zone gravity modeling to perform dynamic Origin-Destination (OD) prediction. Last, we validated our methods using real-world LBSN and transportation data of New York City. The experimental results demonstrate the effectiveness of the proposed method for modeling dynamic urban travel demands. Our method achieves a significant improvement on OD prediction compared to baselines.

# General Framework
We propose a procedure for estimating the dynamic OD flow patterns (Fig. 1), including four major parts: zonal functionality profiling, zonal trip arrival estimation, zone correlation analysis, and zonal OD flow pattern estimation.

For zonal functionality profiling, we treat zonal functionality as a latent "topic" variable to discover from POI categorical density. By classifying zones by these zone topics, we can now analyze interactions between zones of different functionality.

For zonal trip arrival estimation, we applied HPSS formulation to model the trip arrival patterns based on the check-in arrival patterns and the discovered zone topics. The predicted trip arrivals are a form of dynamic urban travel demand patterns. The result is then applied as the input to the dynamic zonal OD estimation model.

For zone correlation analysis, we calculated the PPMC matrix to represent the pairwise zone correlation based on check-in arrival sequence. The result is then used in zonal OD estimation.

Finally, given the gravity model framework, we proposed a joint PPMCGM model to predict the dynamic OD flow patterns. The zone topics will be combined with the predicted OD flow patterns to discover time-of-day variations between pairwise zones of different zonal functionality.

# Baseline models
In this package we consider 4 zonal OD estimation baselins:1) Normalized gravity model with an exponential distance decay function (NGravExp).2) Normalized gravity model with a power distance decay function (NGravPow).3) Schneider intervening opportunities model (Schneider). 4) Radiation model (Rad).

# Contents of the package

DataFrameProcess.ipynb (process the raw check-in data to dataframe. We assigned zone ID and time slot for each check-in record) 

UrbanFunctionalityProfiling.ipynb (input is the dataframe file, output is the assigned zone topic list)

TripArrivalEstimation.ipynb (input is the dataframe file, output is the zonal trip arrival matrix)

zoneCorrelationAnalysis.ipynb (input is the dataframe file, output is the zone correlation matrix)

ModelPerformance.ipynb (input is the dataframe file, zone topic, zonal trip arrival matrix, zone correlation matrix, and the output is the OD matrix. We also have the code for visualization of the MOE result inside)
