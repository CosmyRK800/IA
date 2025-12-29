# IA
Association Rule Mining on FAO Food Consumption Data
This repository contains the source code used for the preprocessing and analysis of FAO/WHO GIFT food consumption data, as part of the case study “Analiza Algoritmilor pentru Descoperirea Regulilor de Asociere și Aplicarea lor pe Seturi de Date FAO privind Consumul Alimentar”.
The objective of this project is to compare the Apriori and FP-Growth algorithms for association rule mining and to identify frequent food consumption patterns in the Brazilian FAO dataset (BRA_00036).

Dataset
The dataset used in this project is provided by the FAO/WHO Global Individual Food Consumption Data Tool (GIFT) and corresponds to individual food consumption records from Brazil (BRA_00036).
Due to data usage restrictions, the raw dataset is not included in this repository and must be downloaded directly from the official FAO website: https://www.fao.org/gift-individual-food-consumption/data/en


Methodology
The analysis follows these main steps:
1.Data preprocessing
-Cleaning and filtering relevant records
-Defining transactions based on subject, survey round, day, and consumption time
-Removing transactions containing a single item
-Standardizing food item descriptions
2.Frequent itemset mining
-Extraction of frequent itemsets using the Apriori algorithm
-Extraction of frequent itemsets using the FP-Growth algorithm
3.Association rule generation
-Generation of association rules using minimum support and confidence thresholds
-Evaluation of rules using support, confidence, and lift metrics
4.Performance comparison
-Comparison of execution time between Apriori and FP-Growth under identical conditions

Results
The results confirm that FP-Growth significantly outperforms Apriori in terms of execution time on medium-to-large food consumption datasets, while producing equivalent frequent itemsets and association rules.
