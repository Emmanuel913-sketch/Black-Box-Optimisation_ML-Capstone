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

# Resipository Contents
The zipped Data folder contains all inputs and outputs for the project and the results of each weekly query for all functions. This includes both intermediate and final datasets, allowing the full optimisation trajectory to be reconstructed and analysed.
The initial_data for functions zipped folder contains the initial provided data & observations.

The funtion_1.ipynb to funtion_8.ipynb Note book files are the main Jupyter notebooks used throughout the project. These notebooks include exploratory analysis, surrogate model development, acquisition strategy testing, and diagnostic visualisations. (These are the main Jupyter Notebooks containing the full code from Function 1 - Function 8. Addtionally, there's the weekly_capstone_feedback.ipynb file for the weekly run after receiving the functions outputs on a weekly basis).

The remaining files provide structured documentation of the project. The README presents the high-level narrative and evolution. The Model Card document documents the final hybrid optimisation approach, the version of the model, thetype of the optimisation model, the intended use of the model and its assumptions. The Datasheet documents the structure, constraints, and evolution of the input–output data.
- Datasheet for the BBO data set: this is the documention detailing the creation, composition, intended use and any information of any preprocessing of the data set - https://github.com/Emmanuel913-sketch/Black-Box-Optimisation_ML-Capstone/edit/main/README.md#:~:text=Datasheet%20for%20my%20BBO%20capstone%20project
- Model Card: This is a summary of the machine learning model, its limitations and perfomance metrics. This is also details Transperacy, ethics and accountability considerations - Model Card for my BBO optimisation.docx - https://github.com/Emmanuel913-sketch/Black-Box-Optimisation_ML-Capstone/blob/main/Model%20Card%20for%20my%20BBO%20optimisation.docx
- The repository is organised to clearly separate data, experimentation, and documentation.

## Key Concepts
Black‑Box Optimisation - Baysian Optimisation
Optimising a function without explicit knowledge of its internal structure.
Surrogate Modelling
Building an approximate model (e.g., RBF, Gaussian Process) to guide search.

Exploration vs Exploitation
  Balancing sampling new regions vs refining known promising areas.
  UCB (Upper Confidence Bound) vs Expected Improvement as acquisition functions
  A strategy that selects points based on predicted value + uncertainty.

  ## Inputs and Outputs
Each black-box function accepts a continuous numeric input vector of the form [x1, x2, …, xn], where n depends on the function. For example, Function 1 operates in two dimensions, while Function 8 operates in eight dimensions. In all cases, the first input value is fixed to zero, and remaining values must respect the specified numerical precision constraints.
Each query produces a single scalar output. Higher values indicate better performance, and no additional metadata or uncertainty estimates are provided by the black-box functions themselves. All uncertainty handling is therefore internal to the optimiser.

## Optimisation from Week 1 to Week 13

The optimisation strategy evolved continuously over the thirteen weeks, moving from broad exploration to focused exploitation as more information became available and the remaining budget shrank.

In the early weeks, the primary goal was to understand the global structure of each function. With very limited initial data, the risk of premature convergence was high. During this phase, the optimiser prioritised exploration across the full input domain. Gaussian Process surrogate models with Matérn kernels were used to capture smooth but flexible response surfaces while maintaining calibrated uncertainty estimates. Acquisition behaviour was tuned to favour uncertainty reduction and information gain rather than immediate performance.
As the project progressed into the middle weeks, sufficient data had accumulated to support more nuanced decision-making. The strategy shifted toward a balanced exploration–exploitation regime. A neural network surrogate was introduced alongside the Gaussian Process to capture more complex or non-stationary patterns that the GP might smooth over. A two-stage candidate selection process was adopted, with the GP used to screen broadly promising regions and the neural network used to rank candidates based on predicted performance. Decisions were increasingly guided by agreement between surrogates rather than uncertainty alone.
In the later weeks, the remaining query budget became the dominant constraint. At this stage, the strategy focused on controlled exploitation and local refinement. Sampling became progressively localised around the best-performing regions identified so far. Dimension relevance was estimated using GP length-scales and neural network sensitivity analysis, allowing weaker or less influential dimensions to be narrowed more aggressively while preserving the full input structure. Small, deliberate exploration steps were retained to guard against overconfidence and surrogate bias, but the emphasis was firmly on extracting maximum value from each remaining evaluation.
By week thirteen, the optimisation process had converged into a tightly focused local search guided by accumulated evidence, surrogate agreement, and stability considerations rather than aggressive exploration.

## Surrogate Models & Strategy
Optimisation was based on a combination of surrogate models rather than a single fixed approach. The Gaussian Processes had a central role throughout the project due to their ability to operate effectively in low-data environments and provide uncertainty estimates that directly support exploration decisions. Matérn kernels are used to balance smoothness and flexibility, and hyperparameters are re-optimised as new data arrive.

## Technologies Used
• 	Jupyter Note books
•   Pytorch
•   Python
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

## Using this Repository
Having a go at this project for a hands-on exploration, the zipped Data folder contains all recorded inputs and outputs, and the Notebooks files contain all the executable analyses and experiments. If running notebooks, ensure that any configurable function number is updated to match the specific black-box function of interest.
These materials aim to document not only the final results, but the full reasoning process behind them, also showing realistic optimisation under uncertainty (Black box) rather than a static or an idealised solution.
