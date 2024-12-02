This repository contains code necessary to replicate experiments and analyses necessary for the paper: Automated Citation Searching in Systematic Review Production : A Simulation Study (Currently under Review)

The repository contains 2 subdirectories 
* analysis : Containing code required to run the analysis and generate tables and figures as contained within the main manuscript.
* experiment: Containing code required to run the simulation study experiments

System Requirements: 
* Python Version 3.10.12 or later 
* For non-Windows users (MacOs / Linux), file reference

NB: Re-running the experiment requires an API Key from Semantic Scholar. This can be requested at https://www.semanticscholar.org/product/api 


--- 

# Instructions 

1. Firstly clone the repository 

## Analysis

1. Install the necessary python packages from requirements.txt (Tested on Windows 10, and WSL 2 Ubuntu 22.04)
2. Run the python notebook: simulation_study_working_manuscript.ipynb to recreate tables and figures 
3. See Results/citation_mining_results/ for consolidated automated citation searching results. 

---

## Experiment 
1. Install the necessary python packages from requirements.txt
2. Modify the .env file with a semantic scholar api key. See https://www.semanticscholar.org/product/api for details to request an API key. 
3. Run main.py
