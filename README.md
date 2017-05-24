## Synopsis

This sample script shows how to use a simple ML algorithm to predict heat of formation of cubic perovskites based on DFT data

## Motivation

Making **stable** perovskites with a given electronic structure is of interest for many applications, including water oxidation catalysis. Computational screening provides a tool to direct synthesis and therefore make the manufacturing of catalysts more efficient. However, traditional screening using Density Functional Theory is computationally demanding. Here is a sample code that provides a simple and inexpensive Machine Learning model that uses DFT data from https://cmr.fysik.dtu.dk/cubic_perovskites/cubic_perovskites.html#cubic-perovskites to predict heat of formation for cubic perovskites. The idea is taken from https://www.nature.com/articles/srep19375 . The ML algorithm below gives an error of the heat of formation on the level of 0.15 eV. 

## Installation

You will need a working Python2 installation. Then follow this link to install the python package manager **pip** https://pip.pypa.io/en/stable/installing/

Here are the steps to install the modules required by the python script:

```
pip install numpy scipy matplotlib pandas mendeleev==0.2.8 requests sqlalchemy 

pip install --upgrade --user ase

pip install -U scikit-learn 

```

## Run the Code

```
python heat_of_formation_Ridge.py 
```

This will create a csv file that is ready to use by any machine leanring module and an image with the predictions of the Kernel Ridge regressor. 

## Contributors

Mikhail Askerka (University of Toronto, Yale University, mikhail.askerka@aya.yale.edu) 

