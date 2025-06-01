# Order Success Prediction for Food Delivery

## Project Overview

This repository contains a machine‐learning solution for predicting whether a food‐delivery order will complete successfully or fail. By analyzing historical order data—covering six weeks of operations across major Indian cities—we developed and compared multiple models to identify orders at high risk of cancellation or late delivery. The ultimate goal is to enable proactive interventions (for example, courier reassignments or customer notifications) that reduce revenue loss and improve customer satisfaction.

---

## Data Description

The raw dataset includes approximately 120 000 orders, each with a rich set of fields combining geospatial, temporal, monetary, and categorical information. Key attributes include:

- **is_successful**: Binary label indicating whether the order was delivered successfully.  
- **order_id**: Unique identifier for each order.  
- **restaurant_lat/restaurant_lng** and **customer_lat/customer_lng**: GPS coordinates of pickup (restaurant) and drop‐off (customer).  
- **dropoff_distance_manhattan**: Manhattan‐distance (in kilometers) between restaurant and customer.  
- **order_time**, **promised_delivery_time**, **actual_delivery_time**: Timestamps indicating when the customer placed the order, when delivery was promised, and when it actually completed (or failed).  
- **order_delay**: Difference (in minutes) between actual and promised delivery times; negative values indicate early delivery.  
- **basket_amount_lc**, **gmv_amount_lc**: Numeric values of basket total and gross merchandise value (including taxes and fees).  
- **delivery_fee**: Fee charged for delivery.  
- **payment_method**: Categorical indicator (e.g., “cash,” “online_card,” “wallet,” “pay_later”).  
- **platform**: Device or channel used to place the order (e.g., “Android,” “iOS,” “Web”).  
- **vertical_class**: Indicates “food” versus “non‐food” order.  
- **delivery_arrangement**: Platform’s own fleet (TGO) vs. third‐party courier (TMP).  
- **acquisition_flag**: Binary flag indicating whether the order was placed by a newly acquired customer (1) or an existing/organic user (0).  
- **affordability_amt_total**: Derived feature summing basket + fees + discounts to approximate total customer spend.  
- **free_delivery_flag**, **affordable_item_flag**, **gem_flag**, **restaurant_affordability_flag**: Binary indicators used by the platform to flag promotional or “affordable” orders.

Additional fields include day-of-week, hour-of-day, weather conditions, traffic density levels, and multiple-deliveries flags. After cleaning (removing duplicates, invalid records, and missing critical values), the final dataset contains roughly 118 500 records.

---

## Exploratory Data Analysis (EDA)

A thorough EDA was conducted to understand data distributions, detect anomalies, and guide feature engineering:

- **Univariate Analysis**  
  Histograms and density plots for numeric variables (basket amount, GMV, delivery fee, delivery duration, order delay, and distance) revealed strong right-skewness. Boxplots highlighted outliers, which were either trimmed or flagged for downstream processing.  

- **Categorical Distributions**  
  Bar and pie charts for order success vs. failure, payment methods, platforms, delivery arrangements, and acquisition status illuminated class imbalance and dominant categories.  

- **Bivariate Relationships**  
  Scatter plots compared promised vs. actual delivery times and their difference (order delay). Countplots linked payment methods, platforms, and affordability flags to success rates, indicating how specific categories impacted cancellation risk.  

- **Temporal Trends**  
  Line charts tracked daily order volumes and average delays over the six-week window, both overall and broken down by major cities. This revealed a gradual decrease in volume and converging delivery-time performance as the period progressed.  

- **Geospatial Observations**  
  A map of order counts by city showed that a handful of metros accounted for most of the business, with a long tail of smaller markets. Correlation heatmaps among numeric features (distance, GMV, basket, fee, delay) guided the removal of highly collinear variables.

---

## Feature Engineering

Based on EDA insights, the following transformations and derived features were implemented:

- **Temporal Features**: Extracted `order_hour`, `order_dayofweek`, and `is_weekend` from timestamps to capture demand and traffic patterns.  
- **Distance Calculation**: Computed `dropoff_distance_manhattan` as Manhattan distance between restaurant and customer.  
- **Delay Derivation**: Defined `order_delay` = `actual_delivery_time` − `promised_delivery_time` (in minutes).  
- **Monetary Transformations**: Applied log-transforms to skewed features (`basket_amount_lc`, `gmv_amount_lc`, `delivery_fee`) and created `affordability_amt_total` as the sum of basket, fees, and any credits.  
- **Lag and Rolling Features**: Constructed lagged versions of `order_delay` and `gmv_amount_lc` for each order, as well as their rolling averages over the previous three orders to capture short-term trends.  
- **Categorical Encoding**: Used one-hot encoding for nominal variables (payment_method, weather, city), label encoding for ordinal features (traffic_density), and passed raw categories directly to CatBoost.  
- **Multicollinearity Reduction**: Performed VIF analysis to remove highly correlated pairs (e.g., `basket_amount_lc` vs. `gmv_amount_lc`, `order_week` vs. `order_dayofweek`/`order_month`).

The final dataset combined these numeric, transformed, and encoded features into a tabular matrix for modeling.

---

## Modeling Pipeline

1. **Data Splits**  
   - **Training**: 70 % of cleaned data, with five-fold stratified cross-validation for hyperparameter tuning.  
   - **Test**: 30 % held out for final evaluation.  

2. **Class Imbalance Handling**  
   - Applied **SMOTE** on training folds to balance 84 % successes / 16 % failures.  
   - For LightGBM and CatBoost, used built-in imbalance parameters (`is_unbalanced=True`, `scale_pos_weight`).  
   - For other models (Random Forest, GBM, Extra Trees), compared class weighting vs. SMOTE.

3. **Baseline Classifiers**  
   - **Logistic Regression** (L2 regularization) to assess linear separability.  
   - **Gaussian Naive Bayes** to test independence assumptions.  
   - **Elastic Net** (L1+L2) for sparse linear models.  

4. **Tree-Based & Gradient-Boosting Models**  
   - **Random Forest** and **Extra Trees** with 200 trees, `min_samples_leaf=10`, and class weights.  
   - **Gradient Boosting** (scikit-learn) with 500 estimators, `max_depth=4`, learning_rate=0.05, failure weights ≈ 5×.  
   - **LightGBM** with 1 000 boosting rounds, `max_depth=6`, feature_fraction=0.8, bagging_fraction=0.8, `is_unbalanced=True`.  
   - **CatBoost** with 800 iterations, `max_depth=6`, L2 regularization=3, `subsample=0.75`, `scale_pos_weight≈5`, and native categorical encoding.  

5. **Deep-Learning Architectures**  
   - **Sequence Construction**: For each order, created a sequence of up to three prior orders (same city), combining numeric features, embedded categorical features (embedding_dim=8), and sinusoidal time encodings.  
   - **RNN Variants**: LSTM, BiLSTM (with attention), and GRU networks processing these three-step sequences; all used weighted binary cross-entropy to handle class imbalance.  
   - **Transformer Encoders**: Encoder-only Transformer (T=3) and a variant including the current order as a fourth “token” (T=4), using self-attention to capture temporal dependencies.  

6. **Ensembling Strategies**  
   - **Voting Ensemble**: Soft-voting average of probabilities from Gradient Boosting, Gaussian Naive Bayes, and Random Forest.  
   - **Stacking Ensemble**: Out-of-fold predictions from CatBoost and LightGBM fed into a logistic-regression meta-learner; also tested stacking of weaker learners (RF, GNB, GBM) with a meta-learner.

---

## Evaluation

All models were assessed on the held-out test set using:

- **Overall Accuracy**  
- **Precision, Recall, and F1-Score** (focused on the failure class)  
- **AUC-ROC** for discrimination across probability thresholds  

Key findings:

- **CatBoost** achieved the highest accuracy and F1-score on the failure class, making it the top single model.  
- **LightGBM** closely matched CatBoost, with nearly identical performance.  
- **Gradient Boosting** performed slightly lower when SMOTE was applied; without SMOTE, it matched LightGBM’s performance closely.  
- **Random Forest** improved with SMOTE but remained below LightGBM/CatBoost.  
- **Gaussian Naive Bayes** and **Elastic Net** provided useful baselines but underperformed compared to tree-based methods.  
- **Deep-Learning Models** (LSTM, BiLSTM, GRU, Transformer) reached nearly the same accuracy as CatBoost but required 2–3× longer training and inference times.  
- **Ensembles** (voting and stacking) offered only marginal improvements over CatBoost alone, at the cost of doubled inference latency.

---

## Conclusion

This project demonstrates that a well-tuned gradient-boosting model—particularly CatBoost—can predict order success with exceptional accuracy (over 99 %) while maintaining low inference latency, making it suitable for real-time deployment. Feature importance analyses highlighted key predictors such as dropoff distance, prior order delay, affordability metrics, and time-of-day patterns. Although deep-learning architectures can capture some temporal dependencies, their marginal gains did not justify higher computational costs. Ensemble pipelines yielded only slight improvements over CatBoost and increased system complexity. In practice, deploying the CatBoost model enables operations teams to proactively mitigate cancellations, reduce late deliveries, and optimize resource allocation, ultimately improving customer satisfaction and profitability.

---

## Future Work

Possible extensions include:

1. **Incorporating Real-Time Data**  
   - Integrate courier GPS telemetry, live traffic feeds, and restaurant preparation logs to enhance delay forecasting and rerouting capabilities.

2. **Extended Sequence Modeling**  
   - Use hierarchical RNNs or Transformers over longer customer histories (beyond three prior orders) to capture long-term behavioral trends.

3. **Multi-Task Learning**  
   - Jointly predict both failure probability and exact minute-level delay, allowing operations to optimize for timeliness and completion risk.

4. **Cost-Sensitive and Reinforcement Learning**  
   - Implement cost-sensitive loss functions (e.g., focal loss) for better minority-class detection or explore RL for dynamic courier‐assignment policies.

5. **Automated Monitoring & Retraining**  
   - Build a production pipeline that monitors model drift, triggers retraining when performance degrades, and integrates A/B testing of proactive interventions.

---

## Repository Structure
EDA and models -- are the final version of the project work 
