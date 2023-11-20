# FIFA-Performance-Prediction-model-
FIFA Performance Prediction model 
# Team Members:
- Jack Dong
- Grant Ovsepyan

# Description and constraint for features: 
- sofifa_id: 
        id of a player.  
        constraint: cannot be null
        
- player_url: 
    the website containing data for player
- short_name and long_name: 
    name of player
- player_positions: 
    player's position in team
- overall and potential: 
    ability score for player.  
    constraint: integer 0-100
    
- value_eur and wage_eur: 
    player's value and wage
- age, dob, height_cm, weight_kg: 
    player's personal information
- club_team_id, club_name, league_name, league_level, club_position, club_jersey_number, club_loaned_from club_joined, club_contract_valid_until: 
        information regarding to player's club  
        constraint:    
            league_level: integer 1-4  
- nationality_id, nation_team_id, nation_position, nation_jersey_number:
    information regarding to player in national team
- preferred_foot, weak_foot, skill_moves:
    player's ability in game
- international_reputation: 
    international_reputation  
    constraint: integer 1-5
- work_rate:
    Work Rates affect where players position themselves on the pitch, in the context of their original starting point. Devided into attack work rate and defense work rate
- body_type:
    player's body type
- real_face:
    whether player use their real scanned face in game
- release_clause_eur
    release clause is a set fee agreed upon when a player signs a contract with a club, allowing another club to sign that player if the fee is met
- player_tags, player_traits:
    personal tags in game
- pace, shooting, passing, dribbling, defending, physic, attacking_crossing, attacking_finishing, attacking_heading_accuracy, attacking_short_passing, attacking_volleys, skill_dribbling, skill_curve, skill_fk_accuracy, skill_long_passing, skill_ball_control, movement_acceleration, movement_sprint_speed, movement_agility, movement_reactions, movement_balance, power_shot_power, power_jumping, power_stamina, power_strength, power_long_shots, mentality_aggression, mentality_interceptions, mentality_positioning, mentality_vision, mentality_penalties, mentality_composure, defending_marking_awareness, defending_standing_tackle, defending_sliding_tackle, goalkeeping_diving, goalkeeping_handling, goalkeeping_kicking, goalkeeping_positioning, goalkeeping_reflexes, goalkeeping_speed: 

    -player's personal stats  
    -constraint: from 0 to 100  
- ls, st, rs, lw, lf, cf, rf, rw, lam, cam, ram, lm, lcm, cm, rcm, rm, lwb, ldm, cdm, rdm, rwb, lb, lcb, cb, rcb, rb, gk:

    player's ability score in different positions. Scores are divided into 2 parts: overall score +/- boost score from reputation.
- player_face_url, club_logo_url, club_flag_url, nation_logo_url, nation_flag_url, 
    external resources about player

### Special note
We confirmed with professor Qu about the requirement for "Your tables should be created in schema with the name 'fifa.'" We thought it was not clear whether the spark schema should be named as fifa or the postgres schema should be named as fifa.

According to professor Qu, the data ingestion to postgres can be done using spark. and it is fine both way whether we decide to create the schema named "fifa" in spark or postgres. So, we manually defines a "fifa" schema in spark, and use it to read in data and write the data to postgres. I believe our current implementation should satisfy this requirement.
 


# Task III 
## Why we choose to use regressors
   we choose to use regressors instead of classifiers because we are predicting a continuous neumerical value of the player's overall value, rather than trying to classifier an outcome into different categories.
## 1. Pytorch
### Neural network hyperparameters:
   The Neural network uses three hyperparameters: learning rate, batch_size, and number of epochs.
- learning rate: controls how much the model's weights are updated during training.
- batch_size: the number of training samples used in one iteration of model training.
- number of epochs: controls how many times the entire dataset is iterated through.
### Linear regression hyperparameters:
   Similarly to neural network, linear regression also uses three hyperparameters: learning rate, batch_size, and number of epochs.
- learning rate: controls how much the model's weights are updated during training.
- batch_size: the number of training samples used in one iteration of model training.
- number of epochs: controls how many times the entire dataset is iterated through.
### Compare the 2 models used in pytorch
   Neural network is a non-linear model, which is able to capture more complex functions. On the other hand, linear regression is a linear model. In the case of our project, neural network has better performance compared to linear regression.  
    However, linear regression model is easier to train than neural network and is computationally efficient.



## 2. Spark ML
In SparkML models were used:  
## Support Vector Machine (SVM) Regressor:
In regression, known as Support Vector Regression (SVR), the model   tries to fit the best line within a threshold error margin. One of the key advantages of SVM is its flexibility through the use of different kernel functions (like linear, polynomial, radial basis function (RBF), and sigmoid). These kernels enable the SVM to capture complex, non-linear relationships in the data, which could be crucial for accurately predicting player values based on varied skill sets.  
### Hyperparameter Tuning for SVM: 
Important hyperparameters in SVM include the type of kernel, the kernel's parameters (like degree for polynomial), and the regularization parameter (C). The choice of kernel and its parameters can significantly influence the model's ability to capture non-linear patterns. The regularization parameter C controls the trade-off between achieving a low error on the training data and minimizing the model complexity to avoid overfitting.

SVM with an appropriate kernel can handle the non-linearity in your data effectively. The flexibility to choose and tune different kernels allows it to model complex relationships that might be present in the FIFA dataset, where the interaction between different skills and the overall value of a player can be intricate. SVM's ability to handle high-dimensional data is also a benefit, considering the variety of skills and attributes present in such datasets. However, it's worth noting that SVMs can be computationally intensive, especially for large datasets, and require careful tuning of hyperparameters to achieve the best performance.
 
Hyperparameters of choice:
- svm.regParam (Regularization Parameter):
Purpose: This parameter imposes a penalty on the model's complexity, with the aim of preventing overfitting. A higher value means more regularization.
Increasing regParam: Increases the strength of regularization, which can lead to a simpler model by penalizing the coefficients more heavily. This can be beneficial if your model is overfitting, but too much regularization might lead to underfitting.
Decreasing regParam: Reduces the strength of regularization, allowing the model to become more complex and fit the training data more closely. However, if it's too low, the model might overfit, capturing noise in the training data as patterns.
- svm.maxIter (Maximum Number of Iterations):
Purpose: This parameter defines the maximum number of iterations the optimizer will run to find the optimal coefficients.
Increasing maxIter: Allows the optimization algorithm more iterations to converge to the best coefficients, potentially leading to a better fit. However, it also increases the computational cost and time.
Decreasing maxIter: Reduces the number of iterations, which can decrease training time but might lead to a suboptimal solution if the algorithm hasn't converged yet.
- svm.elasticNetParam (Elastic Net Mixing Parameter):
Purpose: This parameter balances between L1 and L2 regularization. A value of 0 corresponds to L2 regularization (Ridge), 1 corresponds to L1 regularization (Lasso), and values in between indicate a mix of both.
Increasing elasticNetParam (towards 1): Increases the L1 regularization component, leading to sparser solutions (more coefficients become zero). This can be useful for feature selection but might miss out on some complex patterns.
Decreasing elasticNetParam (towards 0): Increases the L2 regularization component, which tends to spread out the penalty among all coefficients, leading to more distributed, non-sparse models. This can capture complex patterns better but might include unnecessary features.
 
## Random Forest Regressor:
 
Why Random Forest Regressor?  Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees. It's excellent for regression tasks and can handle the complexity of datasets like FIFA's, where there are many features (skills) to consider. Random Forests are less likely to overfit compared to single decision trees and can capture non-linear relationships between features and the target variable. In our hyperparameter tuning, the key hyperparameters include the number of trees, depth of trees, and the number of features to consider for splitting at each node. By tuning these, you can optimize the balance between bias and variance, potentially improving model performance.
 
### Hyperparameter tuning for Random Forest:
Selected hyper parameters for tuning and the reasoning behind choosing them:

- numTrees (Number of Trees):
Rationale: In a Random Forest, multiple decision trees are constructed, and their results are aggregated. The numTrees parameter specifies how many trees are in the forest. Generally, more trees increase the model's ability to capture complex patterns and reduce overfitting, but also increase computational complexity and time.
Smaller datasets might require fewer trees, while larger, more complex datasets can benefit from more trees.

- maxDepth (Maximum Depth of Each Tree):
Rationale: This parameter controls the maximum depth of each tree in the forest. Deeper trees can model more complex patterns but can also lead to overfitting. Conversely, shallower trees may be too simple to capture important patterns in the data.

## Performance analysis 
In both models the hyperparameter tuning was done by creating the instance of the model, constructing the parameter grid 

SVM :
- regParam, [0.1, 0.01]
- svm.maxIter, [10, 15, 20]
- elasticNetParam, [0.0, 0.5, 1.0]
 
Random Forest Tree: 
- numTrees, [5, 10, 15]
- axDepth, [3, 5, 7]
 
and using CrossValidator to divide the dataset into multiple parts to ensure the model is not overfitting and can generalize well to new data.

The model then was defined as an estimator with the parameter grid used for finetuning on the df_validation dataset. The metric for model performance was set to be Root Mean Squared Error and the number of folds was set to 3 for cross validation. The output is the best model determined from the cross-validation process, which is the one that showed the best performance according to the RMSE metric.
 
RMSE results:SVM Regression: est regularization parameter: 0.01
Best maximum iterations: 10
Best elastic net parameter: 0.0
RMSE on Test Data before fine tuning hyperparameters, trained on train dataset: 2.0274
RMSE on Test Data after fine tuning hyperparameters, trained on train dataset: 1.8127
 
Random Forest Tree Regression:
Best number of trees: 15
Best maximum depth: 7 RMSE on Test Data before fine tuning hyperparameters, trained on train dataset: 1.5177
RMSE on Test Data after fine tuning hyperparameters, trained on train dataset: 1.1031
 
## Analysis of hyperparameters:
### SVM: 
- Regularization Parameter (regParam): The fine-tuning shows a preference for lower regularization (0.01), indicating that a less complex model, without being overly penalized, was sufficient for the given data. This balance helped to avoid overfitting while still capturing the essential patterns in the data.
- Maximum Number of Iterations (maxIter): Setting this to 10 suggests that the algorithm was able to converge to a satisfactory solution relatively quickly, balancing computational efficiency with model accuracy.
- Elastic Net Parameter (elasticNetParam): The choice of 0.0, favoring L2 regularization, implies that a more distributed penalty across features (rather than sparsity) was effective for this dataset.
 
### Random Tree Forest: 
- Number of Trees (numTrees): The choice of 15 trees strikes a balance between complexity and computational efficiency. It suggests that this number was sufficient to achieve diversity in the model's predictions, reducing overfitting while capturing the essential patterns.
- Maximum Depth of Each Tree (maxDepth): A depth of 7 indicates a preference for moderately complex models. This depth allows the trees to explore interactions between features without becoming too specific to the training data (overfitting).
 
### Results and discussion of performance of the Spark ML models:
In comparing the performance of the Random Forest Regressor and the SVM Regressor, several key observations emerge. Firstly, in terms of RMSE (Root Mean Square Error), the Random Forest Regressor consistently outperformed the SVM Regressor, both before and after the hyperparameter tuning process. This superior performance suggests that the ensemble method employed by Random Forest was more adept at capturing the complex relationships present in the FIFA dataset, surpassing the SVM's capabilities even with its flexible kernel options.

Regarding computational efficiency, SVMs, particularly those with specific kernels, are known to be computationally demanding. In contrast, despite Random Forest also being resource-intensive due to its use of multiple trees, it appeared to strike a more favorable balance between computational demands and predictive accuracy in this particular scenario.

In terms of handling model complexity and the risk of overfitting, the Random Forest Regressor presents a more robust solution. Its ensemble approach inherently guards against overfitting, a significant advantage over the SVM, which requires careful tuning of its regularization parameters to avoid this issue.

Lastly, the type of data in the FIFA dataset, likely characterized by complex, non-linear patterns, seems to have been more suitably addressed by the Random Forest model. Its ensemble of decision trees is well-equipped to capture such intricate data structures, which might explain its superior performance over the SVM in this context.

# How to run the code:
- For local version of the code, put the data folder and the jupyter notebook "FIFA_Project_Local" in the same folder. Also, find the cell that does the job of writing to postgres, and change the corresponding postgres database properties such as the username, password, url... Finally, run all cells in the jupyter notebook from the beginning.
- For cloud version of the code, we run the data in a 32gb memory dataproc cluster. Before running the code, user should make sure that the csv files can be successfully put into the hadoop file system.  
    We put the 8 csv files in the local disk of cluster, without creating a data folder. So, the 8 csv files and the jupyter notebook "FIFA_Project_Cloud" should be placed under the root directory of local disk. Once these are done, run all cells in the notebook from the beginning.
    
# Videos
- We included 3 videos for demoing our code. The first video includes a thorough walkthrough of the code funcionality. The link is provided here:

https://cmu.box.com/s/purujhduxqszvu1net3n46x723otu6ne

- The second and third videos demonstrated running postgres on cloud for extra credit: The links are provdided here:

https://cmu.box.com/s/24p0k8n3ng90tfyj1t8hlb83lphzobfq

https://cmu.box.com/s/l30isuwqn27nyvwuptb5jhdb84z2f8gd

-Note: we discovered some minor bugs in the code after the video is recorded. If you see disparity between the actual code and the code shown in demo, please go with the actual code in github.
