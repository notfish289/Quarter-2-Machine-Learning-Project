Clustered Decision Trees

Overview

This repository contains the code, dataset, and report for our research on Clustered Decision Trees (CDTs), a novel decision tree splitting heuristic that incorporates K-means clustering to improve classification performance. Traditional heuristics like information gain and Gini impurity act greedily, considering only the immediate next split. Our approach leverages clustering to account for the future purity of splits, leading to better performance on complex, unbalanced datasets.

Repository Structure

main.py - Contains the implementation of Clustered Decision Trees.

credit_card.csv - Includes the original dataset, as well as processed training and testing sets.

Project Report.pdf - Full research paper detailing the methodology, experiments, and results.

Methodology

The ClusteredInfo heuristic modifies the traditional entropy formula by incorporating K-means clustering. By first clustering the data before calculating entropy, this method considers the future predictive power of each split, rather than just its immediate impact.

Dataset

We tested our approach on a highly imbalanced credit card fraud dataset with 30 continuous attributes. After preprocessing:

10,000 samples were selected (38 fraud cases, 9962 non-fraud cases).

Each attribute was discretized into 3 bins to enable decision tree splitting while retaining continuous information for clustering.

An 80-20 split was used for training and testing.

Results

We compared Clustered Decision Trees (CDTs) with a regular decision tree using Scikit-learn. Performance was evaluated using sensitivity (true positive rate), crucial in fraud detection.

Model

Sensitivity (Test Set)

Clustered Decision Tree

0.77

Regular Decision Tree

0.39

CDTs significantly improved performance, especially on the minority class, demonstrating the effectiveness of non-greedy splitting in unbalanced datasets.

Future Work

Testing on additional datasets with different distributions.

Optimizing K-means clustering parameters dynamically.

Exploring extensions beyond binary classification.

Authors & Contributions

Abhinav Palikala and Justin Lee - Developed the CDT algorithm, conducted experiments, and wrote the report.

Teammates - Contributions to preprocessing, evaluation, and documentation.
