ML-AGE-COF-Protocol

Machine Learning–Assisted Genetic Exploration of Base-Functionalized COFs for In Situ FLP Formation

<p align="center"> <img src="docs/figures/ml\_sage\_cof\_workflow.png" width="900"> </p>

Overview



ML-AGE-COF is a machine learning–surrogate assisted genetic exploration protocol for discovering base-functionalized COFs capable of stabilizing complementary Lewis acids and forming in situ FLP environments.



The framework integrates:



Topology-driven genome construction



Structural feasibility filtering



ML-based pore size prediction



FLP capacity estimation



Genetic algorithm–driven evolutionary optimization



The complete pipeline spans COF generation, structural optimization, pore analysis, FLP capacity calculation, ML dataset construction, and surrogate-assisted evolutionary ranking.



For the complete script execution sequence, refer to:



pipeline\_execution\_order.md

Required Software



Before running the workflow, the following software must be installed and accessible in your system PATH:



Pormake

Used for COF construction from topology and building blocks.



Zeo++

Used for Voronoi void network analysis and pore property extraction.



lammps\_interface

Used for conversion of CIF structures to LAMMPS data files.



LAMMPS

Used for geometry optimization of COF structures.



ASE (Atomic Simulation Environment)

Used for structure conversion and CIF/data handling.



All scripts assume these tools are properly configured.



Repository Structure

genetic\_algorithm\_code



Contains all scripts related to the genetic algorithm execution, including mutation, crossover, selection, and ranking routines.



ml\_model\_files



Contains trained ML model files and associated metadata used for surrogate predictions and ranking.



scripts



Contains all Python scripts used across different pipeline stages, including:



COF sampling



Structural filtering



LAMMPS preparation



Zeo++ analysis



FLP capacity computation



Metadata construction



docs



Contains Markdown documentation files explaining pipeline stages and execution logic.



Refer specifically to:



pipeline\_execution\_order.md



for the correct script execution order.



features\_csv\_files



Contains raw feature CSV files corresponding to building block molecules used for ML dataset construction.



structure\_xyz



Contains XYZ structure files of building blocks used during COF construction.



Pipeline Summary



The workflow proceeds as:



COF Sampling

→ COF Construction

→ Structural Filtering

→ LAMMPS Optimization

→ Zeo++ Void Analysis

→ Global Property Extraction

→ FLP Capacity Calculation

→ Model Metadata Construction

→ Genetic Algorithm Evolution



All execution stages are deterministic and intermediate outputs are preserved.







