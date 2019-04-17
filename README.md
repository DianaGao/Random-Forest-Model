# Random-Forest-Model
# Business problem statement
In this case, the company collected the 400 points from the customer historical purchase record. The marketing team intuitively believe the customer's age and estimated salary contribute to the final deal. The company want to find a new model using the customer's age and estimated salary to predict the customer buying result.
# Input raw dataset
![](DataFrame.PNG)

# Results
## Training/Testing set result visualization
 ![](TrainingSetResult.PNG)

 ![](TestingSetResult.PNG)

## Model confusion matrix visualization
The accuracy to predict right is 91.25%. Type one error is 5%, and Type two error is 3.75%. In the business case, the profit of making a final deal with a client is larger than the cost to spend time on providing the service but has no return, so we prefer Type one error than Type two error, asumming the client will buy in the future rather than ignoring the potential customers. 
# ![](ConfusionMatrix.PNG)

## K-fold cross validation evaluation
After iterated for 20 times, the mean of the accuracy is 0.866, and the standard deviation is 0.1. The evaluation results show the model is high in accuracy, low in bias and low in variance.

## Culmulative accuracy profile visualization
![](CAPcurve.PNG)

# Conclusion
The model trained by 20 decision trees and ensembled to build a high accurate model to predict the customer purchase by two inputs, age and estimated salary. The model will help the company to better target the potential buyer group and in return has a better ROI per labour force.  
