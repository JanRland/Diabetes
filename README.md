# Diabetes
This repository contains the analysis results for machine learning models using the [diabetes data set](https://www.sciencedirect.com/science/article/pii/S0002870302000698) collected during the Heinz Nixdorf RECALL Study. To obtain access to the data set, the responsible principal investigator has to be contacted. 


# Data 
A detailed description of the data set can be found [here](https://www.sciencedirect.com/science/article/pii/S0933365719301083?via%3Dihub#bib0125). We split the data set into training, validation and test data set. In contrast to the [first analysis](https://www.sciencedirect.com/science/article/pii/S0933365719301083?via%3Dihub#bib0125) of the data set, cross-validation and upsampling techniques were applied to improve the classification performance and avoid overfitting. 

# Getting started
The augmented data can be generated using the Preprocessing.py script in the src folder. The training routine is defined by the Training.py script in the src folder. The weights of the resulting DenseNet201 is saved in the result folder. In order to use 


![ROC for gausian noise upsampling](images/ROC_res0.png)


## Requirements
The python scripts were tested with following packages and versions: 

   * torch 
   * torchvision
   * PIL
   * imblearn
   * sklearn
   * torch.utils.data
   * pandas
   * numpy


# Publications
Please cite following publication if you use the results:


# Authors
   * Jan Benedikt Ruhland, main contributor
   * Prof. Dr. Dominik Heider, principal inversitgator


# License
MIT license (see license file). 


# Acknowledgments
We want to thank the German state Hessen for the financial support of the project. Furthermore, the  Marburger Rechen-Cluster 3 (MaRC3) for providing the required computational ressources. 
