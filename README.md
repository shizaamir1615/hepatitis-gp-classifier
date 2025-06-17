# Genetic Programming for Hepatitis Classification using ISBA

This project is my implementation of Genetic Programming (GP) and Structure-Based GP (SBGP) with Iterative Subtree-Based Adaptation (ISBA) for classifying hepatitis patient outcomes. The ISBA-enhanced approach consistently outperformed regular GP in terms of F1 score, stability, and generalisation—important metrics for any diagnostic model.

---

## Overview

I compared Regular GP and Structure-Based GP (ISBA) on the EpistasisLab hepatitis dataset. ISBA uses a two-phase evolutionary strategy:  
- **Global Exploration** using multiple seeded runs to find structurally diverse high-performing individuals.  
- **Local Refinement** which locks meaningful subtrees and evolves the rest, preserving useful patterns while refining structure.  

Cosine similarity filtering (GSim) is also used to maintain novelty during crossover.

---

## Dataset

- **Source**: EpistasisLab Hepatitis Dataset (`hepatitis.tsv`)
- **Goal**: Binary classification of patient survival outcome
- **Preprocessing**:
  - I manually oversampled the minority class
  - Re-mapped target values from `2 → 1` and `1 → 0`
  - Removed duplicate entries
  - Split into 80% training and 20% test data

---

## Evolution Setup

| Component              | Description |
|------------------------|-------------|
| **Population Size**    | 200 |
| **Max Tree Depth**     | 4 |
| **Max Generations**    | 50 |
| **Fitness Function**   | 1 - F1 Score |
| **Selection**          | Tournament (size 3) |
| **Operators**          | Crossover, Mutation, Pruning |
| **ISBA Parameters**    | 5 Global runs, 3 Local runs, subtree lock depth = 2 |
| **Novelty Control**    | Cosine Similarity Filter (threshold 0.7) |
| **Early Stopping**     | Triggered after 10 stagnant generations |

---

##  How to Run

>  **Note**: The code takes around **15 minutes** to run. It runs **10 random seeds**, once for Regular GP and once for Structure-Based GP.

###  Requirements

Ensure the following are installed:
- Python 3.x
- `numpy`
- `pandas`

Install dependencies with:
```bash
pip install numpy pandas
```

###  Run the Python Code

1. Place `hepatitis.tsv` in the **same directory** as `main.py`.
2. Run:
```bash
python main.py
```

###️ Run the Compiled Executable

If you’re using the compiled `.exe` or binary:
```bash
./main
```

---

## Results Summary

| Metric               | Regular GP | SBGP (ISBA) |
|----------------------|------------|-------------|
| **Average Train F1** | 0.820      | 0.857       |
| **Average Test F1**  | 0.823      | 0.835       |
| **Best Test F1**     | 0.880      | 0.894       |
| **Std Dev (Test F1)**| 0.047      | 0.036       |

Structure-Based GP achieved higher accuracy, more stable results across seeds, and fewer false positives—an important factor in medical prediction tasks.

---

## Reflection

Initially, I tried simplifying structures too aggressively using penalties and hard constraints. But this led to underfitting. Moving to ISBA helped me refine useful subtrees rather than discarding them, balancing expressivity with structure. Features like subtree locking and cosine similarity made the evolution process more stable and smarter—not just random mutation and crossover.

---

## Future Work

- Hyperparameter tuning for better ISBA performance  
- Integration with semantic-aware crossover  
- Visual analysis of evolved trees for interpretability  

---

**Keywords**: Genetic Programming, ISBA, Evolutionary Algorithms, Hepatitis Classification, Symbolic AI, Python
