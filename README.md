# Diabetes
This repository contains the analysis results for machine learning models using the [diabetes data set](https://www.sciencedirect.com/science/article/pii/S0002870302000698) collected during the Heinz Nixdorf RECALL Study. To obtain access to the data set, the responsible principal investigator has to be contacted. 


# Data 
A detailed description of the data set can be found [here](https://www.sciencedirect.com/science/article/pii/S0933365719301083?via%3Dihub#bib0125). We split the data set into training, validation and test data set. In contrast to a different [analysis](https://www.sciencedirect.com/science/article/pii/S0933365719301083?via%3Dihub#bib0125) of the data set, we modified the features to align with the sensory equipment used in our project.

# Getting started
The model used for the predictions is saved in the model.py file in the src folder. The training routine is described in the training.py file in the src folder. For the final [GUESS](https://academic.oup.com/bioinformatics/article/35/14/2458/5216311) calibration we used the Python version which can be found [here](https://github.com/JanRland/GUESSPY).

## Requirements
The python scripts were tested with following packages and versions: 

   * torch 2.1.0
   * imblearn 0.11.0
   * sklearn 1.2.2
   * pandas 2.0.3
   * numpy 1.26.0
   * pickle 4.0
   * [AdaptiveNorm](https://github.com/JanRland/AdaptiveNormalizationLayer) 


# Publications
Please cite following publication if you use the results:


# Authors
   * Jan Benedikt Ruhland, jan.ruhland@hhu.de, main contributor
   * Prof. Dr. Dominik Heider, dominik.heider@hhu.de, principal inversitgator


# License
MIT license (see license file). 


# Acknowledgments
We want to thank the German state Hessen for the financial support of the project. Furthermore, the  Marburger Rechen-Cluster 3 (MaRC3) for providing the required computational ressources. 
