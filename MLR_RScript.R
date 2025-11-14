#Install and load packages
if (!require(glmnet)) {
  install.packages("glmnet")
}
library(glmnet)
library(MASS)


#Load and prepare data
data(Boston)
head(Boston)

X <- as.matrix(Boston[, -14])
y <- Boston$medv



#Fit a normal linear model
lm_model <- lm(medv ~ ., data = Boston)
lm_coef <- coef(lm_model)
lm_r_squared <- summary(lm_model)$r.squared


#Fit a lasso model (with cross-validation)
cv_model <- cv.glmnet(X, y, alpha = 1)
best_lambda <- cv_model$lambda.min
lasso_model <- glmnet(X, y, alpha = 1, lambda = best_lambda)


#Compare results
lasso_coef <- coef(lasso_model)
lasso_predictions <- predict(lasso_model, newx = X)
lasso_r_squared <- 1 - (sum((y - lasso_predictions)^2) / sum((y - mean(y))^2))

cat("--- Base lm() Coefficients ---\n")
print(lm_coef)

cat("\n--- Lasso Coefficients ---\n")
print(lasso_coef)

cat("\n--- R-squared Comparison ---\n")
cat("Base lm() R-squared:", lm_r_squared, "\n")
cat("Lasso glmnet R-squared:", lasso_r_squared, "\n")
