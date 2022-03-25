# Master Thesis: Engineering and ICT at NTNU
## Title: Detecting Gear and Bearing Faults in Wind Turbines Using Vibration Analysis: 
### A Practical Application of Spectral Analysis and K- Means Clustering


#### Source Code
Master thesis with Morten Olsen Osvik.

The source code is found in `src`.


The code for the three implemented analyis methods are in folder: 
 1. `src/order_analysis`
 2. `src/envelope_order`
 3. `src/clustering`
 
Support functions are found in `src/utils`, `src/data_processing`, `src/hybrid_analysis_process_functions`. 
 
 
The signals were built from a uff-file into pickled objects, using the code in `data_processing` The `read_pyuff` was used to extract the uff-data.

`requirements.txt` shows the libraries used for the project.
 
 #### Abstract
 
 
Monitoring wind turbine components through vibration signals enable operators to detect faults at an early stage, reducing operation and maintenance costs and improving reliability. Consequently, operators such as TrønderEnergi wish to implement monitoring systems utilising vibration signals. This thesis examined vibration signals from four wind turbines owned by TrønderEnergi in order to detect faults and fault development of gears and bearings. In addition, the research explored whether a relationship between start-stop cycles and degradation existed. Most research relies on component dimensions and state when monitoring conditions to validate their results. This study, however, aims to demonstrate the potential for exploratory analysis using only vibration and operational data, when component dimensions and component status is unavailable from the manufacturer.


The signals, recorded from August 2018 to January 2020, were analysed using two approaches. The first was a traditional vibration analysis consisting of order analysis used to inspect gears and an envelope order analysis applied to study bearings. The traditional approach was used to detect faults and fault development over time by inspecting spectrums. Spectrum comparison was carried out. The second approach was a data-driven clustering method using the K-means clustering algorithm, with the aim of detecting fault development over time. Documented features from previous literature were extracted from the signals, enabling the clustering method to identify transient signals and non-linearities, thus detecting fault development over time.


The results of the traditional vibration analysis suggested that one of the turbines could have an early parallel gear and a bearing fault. The same turbine had the highest number of start-stop cycles, which suggested a relation between start-stop cycles and faults. The traditional vibration analysis and the clustering results indicated that no fault development had occurred during the time period. This either suggested that the proposed fault development methods were unable to detect an actual deterioration over time, or that no fault development existed in the signals. The missing information regarding the component dimensions limited the conclusiveness of the results. This study would greatly benefit from knowing this information, and it is recommended that efforts are made to obtain it from the wind turbine manufacturers in future projects.
