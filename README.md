# Overview
This project explores Black‑Box Optimisation using Machine Learning techniques. The goal is to optimise unknown or partially known objective functions where gradients are unavailable or where you can't access the internal inner workings of a system, or where it is expensive to compute, or the objctive function is noisy.
  - Evaluations are expensive (limited queries).
  - Functions may exhibit non-linearity, noise and multiple local maxima.
  - Bayesian optimisation balances exploration (trying new areas) and exploitation (refining near promising points).

The work includes experiments across multiple functions, applying ML‑based optimisation strategies such as:
• Surrogate modelling (Fit a surrogate model) - choose a surrogate model to approximate the objective function. The most commonly used model is a Gaussian process (GP), which provides predictions along with uncertainty estimates. The GP was used for this project.

• 	Bayesian optimisation - 1) Fit a surrogate model: choose a surrogate model to approximate there 
      objective function. The most commonly used model is a Gaussian
      process (GP), which provides predictions along with uncertainty
      estimates. 2) Define an acquisition function: select an acquisition function to guide
      the search for new points. Common choices include expected
      improvement (EI) and upper confidence bound (UCB). The acquisition
      function helps decide where to query next, based on the surrogate
      model’s predictions and uncertainty. 3) Iteratively select new points: use the acquisition function to suggest
      new points at which to query the function. This is where the
      optimisation loop occurs, after each query, update the surrogate model
      with the new data point (the input–output pair). 4) Run the optimisation for a fixed number of iterations, repeat the process of querying, updating the surrogate model and
      suggesting new points. 5)Record the best input and output found:

Visualise progress and summarise results:
• 	Upper Confidence Bound (UCB) strategies
• 	Radial Basis Function (RBF) models
• 	Local and global search heuristics
Each notebook documents a different function, optimisation strategy, or experiment.

# Project Structure

## Objectives
• 	Implement and compare optimisation strategies for black‑box functions
• 	Evaluate performance across multiple functions - 8 functions in this case
• 	Explore ML‑based surrogate models for guiding optimisation - Gaussian Process in this case
• 	Analyse convergence behaviour and sample efficiency
• 	Document findings through reproducible Jupyter notebooks

## Key Concepts
Black‑Box Optimisation
Optimising a function without explicit knowledge of its internal structure.
Surrogate Modelling
Building an approximate model (e.g., RBF, Gaussian Process) to guide search.

Exploration vs Exploitation
  Balancing sampling new regions vs refining known promising areas.
  UCB (Upper Confidence Bound)
  A strategy that selects points based on predicted value + uncertainty.

## Technologies Used
• 	Python
• 	NumPy
• 	SciPy
• 	Scikit‑Learn
• 	Jupyter Notebook
• 	Matplotlib / Seaborn

# Results Summary
Each notebook includes:
• 	Visualisations of optimisation trajectories
• 	Surrogate model predictions
• 	Convergence plots
• 	Comparative performance metrics
These results demonstrate how ML‑based optimisation can outperform naive search on complex, multimodal functions.
