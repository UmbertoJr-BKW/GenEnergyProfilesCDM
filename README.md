# Generating Future Energy Consumption Profiles Using Conditional Diffusion Models

[![Paper DOI](https://img.shields.io/badge/DOI-your_doi_here-blue)](your_doi_link_here) 

## Introduction

This repository provides the code implementation for the paper **"Generating Future Energy Consumption Profiles Using Conditional Diffusion Models"** ([Link to Paper]).
As energy systems transition towards higher integration of renewables (like PV) and electrification (EVs, heat pumps), accurately modeling future energy consumption becomes crucial for grid planning and operation. 
However, historical data often lacks these future characteristics, and real customer data raises privacy concerns.

This work addresses these challenges by developing a **conditional diffusion model** capable of generating realistic, diverse, and privacy-preserving synthetic daily energy consumption profiles based on specific future scenarios defined by conditioning variables.

## Model & Approach

We adapt the framework of Denoising Diffusion Probabilistic Models (DDPMs) for conditional time-series generation. The model learns the underlying distribution of energy consumption patterns from historical smart meter data and learns how this distribution changes based on external factors.

**Key Conditioning Factors Implemented/Supported:** *(Adjust list as needed)*
*   Photovoltaic (PV) System Presence/Capacity
*   Electric Vehicle (EV) Charging Behavior
*   Heat Pump Usage
*   Weather Parameters (Temperature, Solar Irradiance)
*   Day Type (Weekday/Weekend/Holiday)
*   Building Characteristics

## Repository Contents

*   `/src`: Python code for the diffusion model, training loops, data loading, and utility functions.
*   `/scripts`: Example scripts for training the model and generating samples.
*   `/config`: Configuration files for model hyperparameters and training settings.
*   `environment.yml` / `requirements.txt`: Dependencies required to run the code.
*   `README.md`: This file, including setup and usage instructions.
*   `LICENSE`: Project license information.

## Usage

Detailed instructions on environment setup, data preparation (including expected format), training procedures, and generating conditional profiles can be found in the main `/src/README.md`.

## Citation

If this work is useful for your research, please consider citing:

```bibtex
@inproceedings{YourLastNameYEARconditionaldiffusion,
  title={Generating Future Energy Consumption Profiles Using Conditional Diffusion Models},
  author={Author 1 and Author 2 and ...},
  booktitle={Conference Name},
  year={Year},
  % Add pages, doi, etc. when available
}
