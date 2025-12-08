# Multiple Linear Regression and Lasso Regression Analysis

# Install packages if not already installed
if (!require(glmnet)) {
  install.packages("glmnet")
}
if (!require(ggplot2)) {
  install.packages("ggplot2")
}
if (!require(reshape2)) {
  install.packages("reshape2")
}

# Load libraries
library(glmnet)    # For Lasso regression
library(MASS)      # Contains Boston dataset
library(ggplot2)   # For advanced visualizations
library(reshape2)  # For data manipulation

# STEP 2: Load and Prepare Data

# Option 1: Load built-in Boston dataset
#data(Boston)
#head(Boston, 3)

#Option 2: Load your own CSV file from computer
my_data <- read.csv("/Users/jishnu/Desktop/Multiple_Linear_Regression_Using_R/Multiple_Linear_Analysis_Using_R/BostonHousing.csv")
# On Mac/Linux: "/Users/YourName/Documents/data.csv"

data(Boston)
head(Boston, 5)

# Prepare the data: Separate predictors (X) and target variable (y)
X <- as.matrix(Boston[, -14])  # All columns except the 14th (medv)
y <- Boston$medv               # Target variable: median home value

# STEP 3: Fit Linear Regression Model (lm)

# Fit a standard linear model using all predictors
lm_model <- lm(medv ~ ., data = Boston)

# Extract coefficients and R-squared
lm_coef <- coef(lm_model)
lm_r_squared <- summary(lm_model)$r.squared

# Display results
cat("--- Base lm() Coefficients ---\n")
print(lm_coef)
cat("\nR-squared:", lm_r_squared, "\n")


# STEP 4: Fit Lasso Regression Model with Cross-Validation

# Perform cross-validation to find the best lambda
cv_model <- cv.glmnet(X, y, alpha = 1)
best_lambda <- cv_model$lambda.min

cat("\nBest lambda from cross-validation:", best_lambda, "\n")

# Fit Lasso model with the optimal lambda
lasso_model <- glmnet(X, y, alpha = 1, lambda = best_lambda)


# STEP 5: Extract Lasso Results

# Extract Lasso coefficients
lasso_coef <- coef(lasso_model)

# Make predictions
lasso_predictions <- predict(lasso_model, newx = X)

# Calculate R-squared for Lasso
lasso_r_squared <- 1 - (sum((y - lasso_predictions)^2) / 
                          sum((y - mean(y))^2))

# Display results
cat("\n--- Lasso Coefficients ---\n")
print(lasso_coef)
cat("\nLasso R-squared:", lasso_r_squared, "\n")

# STEP 6: Compare Results

cat("\n--- R-squared Comparison ---\n")
cat("Base lm() R-squared:", lm_r_squared, "\n")
cat("Lasso glmnet R-squared:", lasso_r_squared, "\n")
cat("\nDifference:", lm_r_squared - lasso_r_squared, "\n")

# Plot: Predicted vs Actual Values
# Shows model accuracy
plot(y, lasso_predictions, 
     main = "Lasso: Predicted vs Actual Values",
     xlab = "Actual Median Value ($1000s)", 
     ylab = "Predicted Median Value ($1000s)",
     pch = 19, col = rgb(0, 0, 1, 0.5))
abline(0, 1, col = "red", lwd = 2)
legend("topleft", legend = "Perfect Prediction Line", 
       col = "red", lwd = 2, bty = "n")

