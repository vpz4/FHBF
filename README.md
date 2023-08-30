# FHBF
FHBF: Federated hybrid boosted forests with dropout rates for supervised learning tasks across highly imbalanced clinical datasets

This is an open access repository which includes the script of the FHBF algorithm (also refered to as hybrid distributed boosted forests - HDBF). The algorithm was developed in Python 3.8.5. The source code (FHBF.py) compares the proposed FHBF algorithm (see the def HDBF(...) function within the script) with state of the art federated implementations of both conventional and straightforward supervised learning classifiers, such as, the XGBoost and the DART. The script has been tailored to solve a MALT lymphoma classification problem in the domain of primary Sjogren's Syndrome where the federated algorithms were trained across multiple curated and harmonized databases. 

For typical reasons the databases are considered to be batches within a local server to enable its execution from the typical users of this repository. However, the script was executed within a secure cloud infrastructure, where the curated and harmonized data from each cohort were stored in private cloud spaces. To this end, the cohort data providers from the 21 participating international centers of the HarmonicSS consortium (https://cordis.europa.eu/project/id/731944) signed data sharing agreements for GDPR compliance.
