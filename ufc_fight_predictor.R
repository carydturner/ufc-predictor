####################################################
#                      HEAD                        #
####################################################

# File Name: ufc_fight_predictor.R
# Authors: Cary Turner and Tony Kim
# Description: Data stuff
####################################################

library(tidyverse)

master <- read.csv('ufc-master.csv')


# Create an indicator set to 1 if red fighter won and 0 otherwise.
# There were no draws in this data set.
master$red_win <- ifelse(master$Winner == 'Red', 1, 0)
names(master)


# Create a vector of all variables we want to keep
cols <- c(
  'R_odds',
  'B_odds',
  'title_bout',
  'weight_class',
  'no_of_rounds',
  'B_current_lose_streak',
  'B_current_win_streak',
  'B_draw',
  'B_longest_win_streak',
  'B_losses',
  'B_wins',
  'B_Stance',
  'B_total_rounds_fought',
  'B_total_title_bouts',
  'B_win_by_KO.TKO',
  'B_win_by_Submission',
  'B_Weight_lbs',
  'B_avg_SIG_STR_landed',
  'B_avg_SIG_STR_pct',
  'B_avg_SUB_ATT',
  'B_avg_TD_landed',
  'B_avg_TD_pct',
  'B_Height_cms',
  'B_Reach_cms',
  'B_age',
  'R_current_lose_streak',
  'R_current_win_streak',
  'R_draw',
  'R_longest_win_streak',
  'R_losses',
  'R_wins',
  'R_Stance',
  'R_total_rounds_fought',
  'R_total_title_bouts',
  'R_win_by_KO.TKO',
  'R_win_by_Submission',
  'R_Weight_lbs',
  'R_avg_SIG_STR_landed',
  'R_avg_SIG_STR_pct',
  'R_avg_SUB_ATT',
  'R_avg_TD_landed',
  'R_avg_TD_pct',
  'R_Height_cms',
  'R_Reach_cms',
  'R_age',
  'better_rank',
  'finish_round',
  'total_fight_time_secs',
  'red_win'
)



# Update data frame to include only relevant variables
fights <- master[,cols]

# Get rid of NA values
fights <- na.omit(fights)


# Subtract Blue height, weight, reach, and age from Red's
# to get Red's height, weight, reach, and age advantage
fights$height_adv <- fights$R_Height_cms - fights$B_Height_cms
fights$weight_adv <- fights$R_Weight_lbs - fights$B_Weight_lbs
fights$reach_adv <- fights$R_Reach_cms - fights$B_Reach_cms
fights$age_adv <- fights$R_age - fights$B_age

# Do same for avg strikes, and takedowns landed/percentage, and for submission attempts
fights$str_landed_adv <- fights$R_avg_SIG_STR_landed - fights$B_avg_SIG_STR_landed
fights$str_pct_adv <- fights$R_avg_SIG_STR_pct - fights$B_avg_SIG_STR_pct
fights$TD_landed_adv <- fights$R_avg_TD_landed - fights$B_avg_TD_landed
fights$TD_pct_adv <- fights$R_avg_TD_pct - fights$B_avg_TD_pct
fights$sub_att_adv <- fights$R_avg_SUB_ATT - fights$B_avg_SUB_ATT

fights$lose_streak_dif <- fights$R_current_lose_streak - fights$B_current_lose_streak
fights$win_streak_dif <- fights$R_current_win_streak - fights$B_current_win_streak
fights$longest_win_streak_dif <- fights$R_longest_win_streak - fights$B_longest_win_streak
fights$draws_dif <- fights$R_draw - fights$B_draw
fights$losses_dif <- fights$R_losses - fights$B_losses
fights$wins_dif <- fights$R_wins - fights$B_wins
fights$total_rounds_dif <- fights$R_total_rounds_fought - fights$B_total_rounds_fought
fights$title_bouts_dif <- fights$R_total_title_bouts - fights$B_total_title_bouts

# Determine number of wins by KO/TKO or submission for each fighter
fights$R_wins_by_KO_sub <- fights$R_win_by_KO.TKO + fights$R_win_by_Submission
fights$B_wins_by_KO_sub <- fights$B_win_by_KO.TKO + fights$B_win_by_Submission

# Determine number of wins by decision for each fighter.
# Wins by decision = wins - wins by KO/TKO or submission
fights$R_wins_by_dec <- fights$R_wins - fights$R_wins_by_KO_sub
fights$B_wins_by_dec <- fights$B_wins - fights$B_wins_by_KO_sub

# Get rid of nonsensical negative values
fights <- fights %>%
  filter(R_wins_by_dec >= 0, B_wins_by_dec >= 0)

# Calculate the ratio of wins by KO/TKO and submission (finishes) to wins by decision.
# We add one to each to avoid zero division.
fights$R_finish_dec_ratio <- (fights$R_wins_by_KO_sub + 1) / (fights$R_wins_by_dec + 1)
fights$B_finish_dec_ratio <- (fights$B_wins_by_KO_sub + 1) / (fights$B_wins_by_dec + 1)

# Calculate overall win loss ratio for each fighter. Add one again to avoid zero division.
fights$R_wl_ratio <- (fights$R_wins + 1) / (fights$R_losses + 1)
fights$B_wl_ratio <- (fights$R_wins + 1) / (fights$B_losses + 1)

# Difference between number of KO/submission finishes
fights$KO_sub_wins_dif <- fights$R_wins_by_KO_sub - fights$B_wins_by_KO_sub

# Finish ratio difference between fighters
fights$finish_ratio_adv <- fights$R_finish_dec_ratio - fights$B_finish_dec_ratio

# Win-loss ratio difference between fighters
fights$wl_ratio_adv <- fights$R_wl_ratio - fights$B_wl_ratio

# Create variables for different stances of the fighters
fights$R_switch <- ifelse(fights$R_Stance == 'Switch', 1, 0)
fights$B_switch <- ifelse(fights$B_Stance == 'Switch', 1, 0)

# Create indicator for each fighter's relative ranking
fights$R_better_rank <- ifelse(fights$better_rank == 'Red', 1, 0)
fights$B_better_rank <- ifelse(fights$better_rank == 'Blue',  1, 0)



# Create vector with names of all variables we are keeping for analysis
keep <- c(
  'R_odds',
  'red_win',
  'title_bout',
  'weight_class',
  'height_adv',
  'weight_adv',
  'age_adv',
  'reach_adv',
  'str_landed_adv',
  'str_pct_adv',
  'TD_landed_adv',
  'TD_pct_adv',
  'lose_streak_dif',
  'win_streak_dif',
  'longest_win_streak_dif',
  'draws_dif',
  'losses_dif',
  'wins_dif',
  'total_rounds_dif',
  'title_bouts_dif',
  'KO_sub_wins_dif',
  'finish_ratio_adv',
  'wl_ratio_adv',
  'R_better_rank',
  'B_better_rank',
  'R_switch',
  'B_switch'
)


############### Split Data Into Test and Train Sets ###################

# Split into train and test sets, using 80/20 split
set.seed(138)
in.test <- sample(nrow(fights), nrow(fights)/5)

# Sanity check
length(in.test)
dim(fights)[1]*0.2

# Assign train and test sets
test <- fights[in.test,keep]
train <- fights[-in.test,keep]



############## Initial Data Exploration ###############

# Indices of non-numeric variables
non_numeric <- c(3,4)

# cube predictors (would square them but we want negatives to stay negative)
cubed.vars <- train[,-c(2, 3, 4, 24, 25, 26, 27)]^3
cubed.vars$red_win <- train$red_win

# Check for correlation between variables and outputs
library(corrplot)
C <- cor(train[,-non_numeric])
corrplot(C, method = 'circle')
C

# Check for correlation between cubic vars and outputs
C.cubic <- cor(cubed.vars)
corrplot(C.cubic, method = 'circle')

colors <- c('1' = 'red', '0' = 'blue')

# Plot the density of all variables, separated by cases where red wins vs. blue wins
for (var in names(train[,-non_numeric])) {
  title <- cat('Density of',var)
  plot <- ggplot(train) +
    geom_density(aes(x = train[,var], color = as.factor(red_win))) +
    scale_color_manual(values = colors) +
    labs(x = var, color = 'red_win', title = title) +
    theme(plot.title = element_text(hjust = 0.5))
  print(plot)
  Sys.sleep(2)
}

# Scatter plot of each var vs. R_odds
for (var in names(train[,-non_numeric])) {
  title <- cat('Red Odds Vs.',var)
  plot <- ggplot(train, aes(x = train[,var], y = train$R_odds)) +
    geom_point() +
    geom_smooth(method = 'lm', color = 'blue', se = FALSE) +
    labs(x = var, y = 'R_odds', title = title) +
    theme(plot.title = element_text(hjust = 0.5))
  print(plot)
  Sys.sleep(2)
}



############### Predictive Modeling #################
library(ROCR)
library(glmnet)
library(cvTools)
library(boot)

# Function to compute the AUC of a binary classifier
compute_auc <- function(p, labels) {
  pred <- prediction(p, labels)
  auc <- performance(pred, 'auc')
  auc <- unlist(slot(auc, 'y.values'))
  auc
}

# Function to plot the ROC curve of a binary classifier
plot_roc <- function(p, labels, model_name) {
  pred <- prediction(p, labels)
  perf <- performance(pred,"tpr","fpr")
  plot(perf, col="black", main = paste("ROC for", model_name))
}

# Calculates accuracy of our logistic regression classifier
# to be used when running cross-validation
cost <- function(r, pi) mean(abs(r-pi) > 0.5)

# Build logistic regression model using all variables. This is our base
# model that we will judge other models against.
glm.base <- glm(red_win ~ ., data = train, family = 'binomial')

# Compute cross-validation error using the base model
base.err <- cv.glm(train, glm.base, cost, K = 10)
base.err[3]

glm.base.prob <- predict(glm.base, train, type='response')
glm.base.pred <- rep(0, length(train$red_win))
glm.base.pred[(glm.base.prob > 0.55)] <- 1
mean(glm.base.pred == train$red_win)

# Compute AUC on training set
base.train.auc <- compute_auc(predict(glm.base, data = train, type = 'response'), train$red_win)
base.train.auc



################## Ridge and Lasso Models for Binary Outcome #####################

### Predicting wins/losses ###

# Remove binary outcome variable (red_win) from cubed.vars data frame
cubed.vars <- cubed.vars[,-21]

# Change names of all columns to reflect that they are cubed.
cubed.cols <- rep('', dim(cubed.vars)[2])
for (i in 1:length(names(cubed.vars))) {
  cubed.cols[i] <- paste(names(cubed.vars)[i], '^3', sep='')
}


# Create model matrix with all cubed terms and all two way interactions
form <- red_win ~ . + .^2
x <- model.matrix(form, data=train)

# Use column bind to add all cubed variables and our outcome variable (red_win)
x <- cbind(train$red_win, x, cubed.vars)
x <- x[,-2]
colnames(x)[colnames(x) == 'train$red_win'] <- 'red_win'

# Recast it as a model matrix
x.train <- model.matrix(red_win ~ ., x)

# Create vector of our outcome variable
y.train <- train$red_win

### Ridge Regression
ridge.lr <- cv.glmnet(x.train, y.train, type.measure='auc', alpha=0, 
                      family='binomial', standardize=TRUE, seed=1)
plot(ridge.lr$lambda, ridge.lr$cvm, xlab='Lambda', ylab='AUC', main='Ridge')
ridge.lambda <- ridge.lr$lambda.min
ridge.auc <- max(ridge.lr$cvm)

# Make predictions on full training set (using lambda that maximizes AUC)
ridge.lr.pred <- predict(ridge.lr, s=ridge.lr$lambda.min, newx=x.train, type='response')

# Plot ROC curve for model using AUC-maximizing lambda
plot_roc(ridge.lr.pred, train$red_win, paste('Ridge (Lambda = ',ridge.lr$lambda.min,')', sep=''))
ridge.train.auc <- compute_auc(ridge.lr.pred, train$red_win)
ridge.train.auc

# Transform probabilities into actual predicted outcomes, using 0.55 as our threshold
ridge.lr.pred <- ifelse(ridge.lr.pred > 0.55, 1, 0)

# Create confusion matrix
ridge.conf <- table(ridge.lr.pred, train$red_win)
ridge.conf

# Calculate classification metrics
TP <- ridge.conf[2,2]
TN <- ridge.conf[1,1]
FP <- ridge.conf[2,1]
FN <- ridge.conf[1,2]
n <- TP + TN + FP + FN
accuracy <- (TP + TN) / n
zo.loss <- (FP + FN) / n
TPR <- TP / (FN + TP)
FPR <- FP / (TN + FP)
TNR <- TN / (TN + FP)
FNR <- FN / (FN + TP)
precision <- TP / (TP + FP)
false.discovery <- FP / (TP + FP)

Ridge <- c(accuracy, zo.loss, TPR, TNR, precision, FPR, FNR, false.discovery, ridge.auc, NA)



### Lasso
lasso.lr <- cv.glmnet(x.train, y.train, type.measure='auc', alpha=1, 
                      family='binomial', standardize=TRUE, seed=1)
plot(lasso.lr$lambda, lasso.lr$cvm, xlab='Lambda', ylab='AUC', main='Lasso')
lasso.lambda <- lasso.lr$lambda.min
lasso.auc <- max(lasso.lr$cvm)

# Make predictions on full training set (using AUC-maximizing lambda)
lasso.lr.pred <- predict(lasso.lr, s=lasso.lr$lambda.min, newx=x.train, type='response')

# Plot ROC curve for Lasso model using AUC-maximizing lambda
plot_roc(lasso.lr.pred, train$red_win, paste('Lasso (Lambda = ',lasso.lr$lambda.min,')', sep=''))
lasso.train.auc <- compute_auc(lasso.lr.pred, train$red_win)
lasso.train.auc

# Transform probabilities into actual predicted outcomes, using 0.55 as our threshold
lasso.lr.pred <- ifelse(lasso.lr.pred > 0.55, 1, 0)

# Create confusion matrix
lasso.conf <- table(lasso.lr.pred, train$red_win)
lasso.conf

# Compute classification metrics
TP <- lasso.conf[2,2]
TN <- lasso.conf[1,1]
FP <- lasso.conf[2,1]
FN <- lasso.conf[1,2]
n <- TP + TN + FP + FN
accuracy <- (TP + TN) / n
zo.loss <- (FP + FN) / n
TPR <- TP / (FN + TP)
FPR <- FP / (TN + FP)
TNR <- TN / (TN + FP)
FNR <- FN / (FN + TP)
precision <- TP / (TP + FP)
false.discovery <- FP / (TP + FP)

Lasso <- c(accuracy, zo.loss, TPR, TNR, precision, FPR, FNR, false.discovery, lasso.auc, NA)
Metric <- c('Accuracy', '0-1 Loss', 'Sensitivity', 'Specificity', 'Precision',
            'Type I Error Rate', 'Type II Error Rate', 'False Discovery Rate', 'CV AUC', 'Test AUC')

# Create data frame containing all metrics for both models
model.metrics.train <- tibble(Metric, Lasso, Ridge)
model.metrics.train

# Coefficients for both models
coef(lasso.lr, s='lambda.min')
coef(ridge.lr, s='lambda.min')






######### Predict for test ############
test.cubed.vars <- test[,-c(2, 3, 4, 24, 25, 26, 27)]^3
test.cubed.vars$red_win <- test$red_win

# Remove binary outcome variable (red_win) from cubed.vars data frame
test.cubed.vars <- test.cubed.vars[,-21]

# Change names of all columns to reflect that they are cubed.
test.cubed.cols <- rep('', dim(test.cubed.vars)[2])
for (i in 1:length(names(test.cubed.vars))) {
  test.cubed.cols[i] <- paste(names(test.cubed.vars)[i], '^3', sep='')
}


# Create model matrix with all cubed terms and all two way interactions
test.form <- red_win ~ . + .^2
x.test <- model.matrix(test.form, data=test)

# Use column bind to add all cubed variables and our outcome variable (red_win)
x.test <- cbind(test$red_win, x.test, test.cubed.vars)
x.test <- x.test[,-2]
colnames(x.test)[colnames(x.test) == 'test$red_win'] <- 'red_win'

# Recast it as a model matrix
x.test <- model.matrix(red_win ~ ., x.test)

# Create vector of our outcome variable
y.test <- test$red_win

ridge.lr.prob.test <- predict(ridge.lr, s=ridge.lr$lambda.min, 
                              newx=x.test, type='response')

plot_roc(ridge.lr.prob.test, test$red_win, paste('Ridge (Lambda = ',ridge.lr$lambda.min,')', sep=''))
ridge.test.auc <- compute_auc(ridge.lr.prob.test, test$red_win)
ridge.test.auc

# Transform probabilities into actual predicted outcomes, using 0.55 as our threshold
ridge.lr.pred.test <- ifelse(ridge.lr.prob.test > 0.55, 1, 0)

ridge.conf <- table(ridge.lr.pred.test, test$red_win)
ridge.conf

# Calculate classification metrics
TP <- ridge.conf[2,2]
TN <- ridge.conf[1,1]
FP <- ridge.conf[2,1]
FN <- ridge.conf[1,2]
n <- TP + TN + FP + FN
accuracy <- (TP + TN) / n
zo.loss <- (FP + FN) / n
TPR <- TP / (FN + TP)
FPR <- FP / (TN + FP)
TNR <- TN / (TN + FP)
FNR <- FN / (FN + TP)
precision <- TP / (TP + FP)
false.discovery <- FP / (TP + FP)

Ridge <- c(accuracy, zo.loss, TPR, TNR, precision, FPR, FNR, false.discovery, NA, ridge.test.auc)
TestMetric <- c('Accuracy', '0-1 Loss', 'Sensitivity', 'Specificity', 'Precision',
            'Type I Error Rate', 'Type II Error Rate', 'False Discovery Rate', 'CV AUC','Test AUC')
model.metrics.test.ridge <- tibble(TestMetric, Ridge)
model.metrics.test.ridge

lasso.lr.prob.test <- predict(lasso.lr, s=lasso.lr$lambda.min, 
                              newx=x.test, type='response')

plot_roc(lasso.lr.prob.test, test$red_win, paste('Lasso (Lambda = ',lasso.lr$lambda.min,')', sep=''))
lasso.test.auc <- compute_auc(lasso.lr.prob.test, test$red_win)
lasso.test.auc

# Transform probabilities into actual predicted outcomes, using 0.58 as our threshold
lasso.lr.pred.test <- ifelse(lasso.lr.prob.test > 0.58, 1, 0)

lasso.conf <- table(lasso.lr.pred.test, test$red_win)
lasso.conf

# Calculate classification metrics
TP <- lasso.conf[2,2]
TN <- lasso.conf[1,1]
FP <- lasso.conf[2,1]
FN <- lasso.conf[1,2]
n <- TP + TN + FP + FN
accuracy <- (TP + TN) / n
zo.loss <- (FP + FN) / n
TPR <- TP / (FN + TP)
FPR <- FP / (TN + FP)
TNR <- TN / (TN + FP)
FNR <- FN / (FN + TP)
precision <- TP / (TP + FP)
false.discovery <- FP / (TP + FP)

Lasso <- c(accuracy, zo.loss, TPR, TNR, precision, FPR, FNR, false.discovery, NA, lasso.test.auc)
model.metrics.test <- tibble(TestMetric, Lasso, Ridge)
model.metrics.test

# Compare Lasso results on training data vs. test data
train.test.metrics <- data.frame(TestMetric, model.metrics.train[,2], model.metrics.test[,2])
colnames(train.test.metrics) <- c('Metric', 'Train', 'Test')
train.test.metrics


################### Predicting Fighter Odds ######################

# take away red_win column because it would give what we're predicting for away
train_odds <- select(train, -(red_win))
test_odds <- select(test, -(red_win))

# baseline RMSE
rmse_baseline <- sqrt(mean((train_odds$R_odds - mean(train_odds$R_odds)) ^ 2))
print(paste("Baseline RMSE of always predicting sample mean:",
            rmse_baseline))

# make the non-numeric weight_class a categorical variable
train_odds$weight_class <- factor(train_odds$weight_class)
test_odds$weight_class <- factor(test_odds$weight_class)

# fit vanilla regression model
fit <- lm(R_odds ~ ., data=train_odds)
val_error_wo_transforms <-
  cvFit(fit, data=train_odds, y=train_odds$R_odds, K=10, seed=1)$cv
val_error_wo_transforms

# divide train data into X and Y
# will be useful for Ridge and Lasso especially
X.train_odds <- select(train_odds, -(R_odds))
Y.train_odds <- train_odds$R_odds
# cubic transform on all numeric variables
cubed_X.train_odds <- 
  select(X.train_odds, 
         -c(title_bout, weight_class, R_better_rank, B_better_rank,
            R_switch, B_switch))^3
# rename cubed column names to avoid confusion
new_col_names <- rep(NA, ncol(cubed_X.train_odds))
for (i in 1:length(new_col_names)) {
  new_col_names[i] <- paste(colnames(cubed_X.train_odds)[i], '^3', sep='')
}
colnames(cubed_X.train_odds) <- new_col_names

# column-bind cube-transformed data to predict
transformed_data_odds <- cbind(Y.train_odds, X.train_odds, cubed_X.train_odds)
fit_transform <- lm(Y.train_odds ~ ., data=transformed_data_odds)
val_error_w_transforms <-
  cvFit(fit_transform, data=transformed_data_odds,
        y=transformed_data_odds$Y.train_odds, K=10, seed=1)$cv
# error is going way up with cubic transform; it might be overfitting
val_error_w_transforms

# so let's predict on training set and measure training error
pred <- predict(fit_transform, transformed_data_odds)
rmse_transform_train <- sqrt(mean((transformed_data_odds$Y.train_odds 
                            -pred) ^ 2))
# in fact, training error is very low
print(paste("Training RMSE of transformed data:",
            rmse_transform_train))

# shoots way up for interaction variables; on top of that, getting additional
# error saying invalid rank for using cvFit

fit_inter <- lm(R_odds ~ . + .:., data=train_odds)
val_error_inter <-
  cvFit(fit_inter, data=train_odds, y=train_odds$R_odds, K=5, seed=1)$cv
val_error_inter


# Create model matrix with all two way interactions
form_inter <- R_odds ~ . + .^2
X.train_odds_matrix_inter <- model.matrix(form_inter, data=train_odds)
# column-bind cubic terms to the interaction terms
data_w_cubes_inter <- cbind(X.train_odds_matrix_inter, cubed_X.train_odds, Y.train_odds)
# ultimate model matrix with interactions and transforms
X.train_odds_matrix_inter <- model.matrix(Y.train_odds ~ ., data=data_w_cubes_inter)

# 10 fold cross validation for Ridge with all interactions and transforms
fm.ridge_inter <- cv.glmnet(X.train_odds_matrix_inter,
                            Y.train_odds, type.measure='mse',
                            alpha = 0, seed=1, standardized=TRUE)
# Value of lambda that gives minimum MSE
fm.ridge_inter$lambda.min
i <- which(fm.ridge_inter$lambda == fm.ridge_inter$lambda.min)
# minimum RMSE
rmse_ridge_inter <- sqrt(fm.ridge_inter$cvm[i])
print(rmse_ridge_inter)

# 10 fold cross validation for Ridge w/o interactions and transforms
form <- R_odds ~ .
X.train_odds_matrix <- model.matrix(form, data=train_odds)
fm.ridge <- cv.glmnet(X.train_odds_matrix, Y.train_odds, type.measure='mse',
                      alpha = 0, seed=1, standardize=TRUE)
# Value of lambda that gives minimum MSE
fm.ridge$lambda.min
plot(fm.ridge$lambda, fm.ridge$cvm, xlab='Lambda', ylab='MSE', main='Ridge', xlim=c(0,1000))
i <- which(fm.ridge$lambda == fm.ridge$lambda.min)
# minimum RMSE
rmse_ridge_wo_inter <- sqrt(fm.ridge$cvm[i])
print(rmse_ridge_wo_inter)

# 10 fold cross validation for Lasso with all interactions and transforms
fm.lasso_inter <- cv.glmnet(X.train_odds_matrix_inter,
                            Y.train_odds, type.measure='mse',
                            alpha = 1, seed=1, standardize=TRUE)
# Value of lambda that gives minimum MSE
fm.lasso_inter$lambda.min
i <- which(fm.lasso_inter$lambda == fm.lasso_inter$lambda.min)
rmse_lasso_inter <- sqrt(fm.lasso_inter$cvm[i])
# minimum RMSE
print(rmse_lasso_inter)
plot(fm.lasso_inter$lambda, fm.lasso_inter$cvm, xlab='Lambda', ylab='MSE', 
     main='MSE over Different Lambdas for Lasso', xlim=c(0,100))
# lowest RMSE

# 10 fold cross validation for Lasso w/o interactions and transforms
fm.lasso <- cv.glmnet(X.train_odds_matrix, Y.train_odds, type.measure='mse',
                      alpha = 1, seed=1, standardize=TRUE)
# Value of lambda that gives minimum MSE
fm.lasso$lambda.min
i <- which(fm.lasso$lambda == fm.lasso$lambda.min)
# minimum RMSE
rmse_lasso_wo_inter <- sqrt(fm.lasso$cvm[i])
print(rmse_lasso_wo_inter)

coef(fm.lasso_inter, s='lambda.min')
coef(fm.lasso, s='lambda.min')
coef(fm.ridge_inter, s='lambda.min')
coef(fm.ridge, s='lambda.min')

# create table of all RMSE
RMSE <- c(rmse_baseline, val_error_wo_transforms, val_error_w_transforms,
                 rmse_transform_train, rmse_ridge_wo_inter, rmse_ridge_inter,
                 rmse_lasso_wo_inter, rmse_lasso_inter)
Models <- c("Baseline Training Error", 
            "CV Error for Linear Regression w/o Transforms", 
            "CV Error for Linear Regression w/ Transforms", 
            "Training Error for Linear Regression w/ Transforms",
            "CV Error for Ridge w/o Transforms and Interactions",
            "CV Error for Ridge w/ Transforms and Interactions",
            "CV Error for Lasso w/o Transforms and Interactions",
            "CV Error for Lasso w/ Transforms and Interactions")
rmse_errors_df <- data.frame(Models, RMSE)
rmse_errors_df




X.test_odds <- select(test_odds, -(R_odds))
Y.test_odds <- test_odds$R_odds
# cubic transform on all numeric variables
cubed_X.test_odds <- 
  select(X.test_odds, 
         -c(title_bout, weight_class, R_better_rank, B_better_rank,
            R_switch, B_switch))^3
colnames(cubed_X.test_odds) <- new_col_names

# Create model matrix with all two way interactions
form_inter.test <- R_odds ~ . + .^2
X.test_odds_matrix_inter <- model.matrix(form_inter.test, data=test_odds)
# column-bind cubic terms to the interaction terms
data_w_cubes_inter.test <- cbind(X.test_odds_matrix_inter, cubed_X.test_odds, Y.test_odds)
# ultimate model matrix with interactions and transforms
X.test_odds_matrix_inter <- model.matrix(Y.test_odds ~ ., data=data_w_cubes_inter.test)

fm.lasso_inter.pred.test <- predict(fm.lasso_inter, s=fm.lasso$lambda.min, 
                              newx=X.test_odds_matrix_inter)
rmse_lasso_test <- sqrt(mean((Y.test_odds
                            - fm.lasso_inter.pred.test) ^ 2))
# in fact, training error is very low
print(paste("Test RMSE Lasso:", rmse_lasso_test))








########## Inference ############

# Fitting a logistic regression model using the subset of variables selected by the lasso
train.model <- glm(red_win ~ R_odds + age_adv, data=train, family='binomial')
summary(train.model) # Both appear to be very statistically significant

# Extract p-values for all coefficients
p.vals <- summary(train.model)$coef[,4]
p.vals

# Use Benji-Hoch to adjust p-values to account for multiple hypothesis testing
benji.hoch <- p.adjust(p.vals, method='BH', n=3)
benji.hoch

# Now try the ol' boneroni
boneroni <- p.adjust(p.vals, method='bonferroni', n=3)
boneroni

# Compare the results from all three methods
p.val.table <- data.frame(p.vals, benji.hoch, boneroni)
colnames(p.val.table) <- c('GLM Output', 'BH', 'Bonferroni')
t(p.val.table)



# Let's refit on the test data to compare
test.model <- glm(red_win ~ R_odds + age_adv, data=test, family='binomial')
summary(test.model)

# Extract p-values for all coefficients
p.vals <- summary(test.model)$coef[,4]
p.vals

# Use Benji-Hoch to adjust p-values to account for multiple hypothesis testing
benji.hoch <- p.adjust(p.vals, method='BH', n=3)
benji.hoch

# Now try the ol' boneroni
boneroni <- p.adjust(p.vals, method='bonferroni', n=3)
boneroni

# Compare the results from all three methods
p.vals.test <- data.frame(p.vals, benji.hoch, boneroni)
colnames(p.vals.test) <- c('GLM Output', 'BH', 'Bonferroni')
t(p.vals.test)



########### Confidence Intervals #############

# Construct 95% normal confidence intervals

summ <- summary(train.model)
summ$coef
low.norm <- summ$coef[,1] - 1.96*summ$coef[,2]
high.norm <- summ$coef[,1] + 1.96*summ$coef[,2]
CI.norm <- data.frame(low.norm, high.norm)
colnames(CI.norm) <- c('Low', 'High')
CI.norm
CI.norm$Size <- CI.norm$High - CI.norm$Low
CI.norm


# Bootstrap for confidence intervals
intercept <- rep(0, 10000)
r.odds <- rep(0, 10000)
age.adv <- rep(0, 10000)

for (i in 1:10000) {
  boot.samp <- sample(1:nrow(train), nrow(train), replace=TRUE)
  data <- train[boot.samp,]
  model <- glm(red_win ~ R_odds + age_adv, data=data, family='binomial')
  coefs <- summary(model)$coef
  intercept[i] <- coefs[1,1]
  r.odds[i] <- coefs[2,1]
  age.adv[i] <- coefs[3,1]
}

# Plot distribution of all bootstrapped estimates
hist(intercept, main='Bootstrapped Distribution of Intercept (Training Data)')
hist(r.odds, main='Bootstrapped Distribution of R_odds (Training Data)')
hist(age.adv, main='Bootstrapped Distribution of age_adv (Training Data)')
# All distributions look very close to normal distribution

# Calculate standard errors
se.int <- sd(intercept)
se.odds <- sd(r.odds)
se.age <- sd(age.adv)
se <- c(se.int, se.odds, se.age)

# Build 95% confidence interval using bootstrapped SE's
low.boot <- summ$coef[,1] - 1.96*se
high.boot <- summ$coef[,1] + 1.96*se

CI.boot <- data.frame(low.boot, high.boot)
colnames(CI.boot) <- c('Low', 'High')
CI.boot$Size <- CI.boot$High - CI.boot$Low
CI.boot





# Construct 95% normal confidence intervals on test data

summ.test <- summary(test.model)
summ.test$coef
low.norm.test <- summ.test$coef[,1] - 1.96*summ.test$coef[,2]
high.norm.test <- summ.test$coef[,1] + 1.96*summ.test$coef[,2]
CI.norm.test <- data.frame(low.norm.test, high.norm.test)
colnames(CI.norm.test) <- c('Low', 'High')
CI.norm.test
CI.norm.test$Size <- CI.norm.test$High - CI.norm.test$Low
CI.norm.test


# Bootstrap for confidence intervals
intercept.test <- rep(0, 10000)
r.odds.test <- rep(0, 10000)
age.adv.test <- rep(0, 10000)

for (i in 1:10000) {
  boot.samp <- sample(1:nrow(test), nrow(test), replace=TRUE)
  data <- test[boot.samp,]
  model <- glm(red_win ~ R_odds + age_adv, data=data, family='binomial')
  coefs <- summary(model)$coef
  intercept.test[i] <- coefs[1,1]
  r.odds.test[i] <- coefs[2,1]
  age.adv.test[i] <- coefs[3,1]
}

# Plot distribution of all bootstrapped estimates
hist(intercept.test, main='Bootstrapped Distribution of Intercept (Test Data)')
hist(r.odds.test, main='Bootstrapped Distribution of R_odds (Test Data)')
hist(age.adv.test, main='Bootstrapped Distribution of age_adv (Test Data)')
# All distributions look very close to normal distribution

# Calculate standard errors
se.int.test <- sd(intercept.test)
se.odds.test <- sd(r.odds.test)
se.age.test <- sd(age.adv.test)
se.test <- c(se.int.test, se.odds.test, se.age.test)

# Build 95% confidence interval using bootstrapped SE's
low.boot.test <- summ.test$coef[,1] - 1.96*se.test
high.boot.test <- summ.test$coef[,1] + 1.96*se.test

CI.boot.test <- data.frame(low.boot.test, high.boot.test)
colnames(CI.boot.test) <- c('Low', 'High')
CI.boot.test$Size <- CI.boot.test$High - CI.boot.test$Low
CI.boot.test

CI.norm.test
CI.boot.test


########### Compare to model with all covariates #############


full.mod <- glm(red_win ~ ., data=train, family='binomial')
summ.full <- summary(full.mod)
summ.full

# Extract p-values
p.vals.full <- summ.full$coef[,4]
p.vals.full

sig.p.vals <- p.vals.full[p.vals.full<0.05]
sig.p.vals

# Use Benji-Hoch to adjust p-values to account for multiple hypothesis testing
benji.hoch.full <- p.adjust(sig.p.vals, method='BH', n=length(p.vals.full))
benji.hoch.full

# Now try the ol' boneroni
boneroni.full <- p.adjust(sig.p.vals, method='bonferroni', n=length(p.vals.full))
boneroni.full

# Compare the results from all three methods
p.val.table.full <- data.frame(sig.p.vals, benji.hoch.full, boneroni.full)
colnames(p.val.table.full) <- c('GLM Output', 'BH', 'Bonferroni')
t(p.val.table.full)











########## Split 50/50 to correct post-selection inference ##########
set.seed(138)
infer.split <- sample(nrow(fights), nrow(fights)/2)

infer1 <- fights[infer.split,keep]
infer2 <- fights[-infer.split,keep]

# Create model matrix with no two way interactions for now
form.infer <- red_win ~ . + .^2
x.infer <- model.matrix(form.infer, data=infer1)

y.infer <- infer1$red_win

lasso.infer <- cv.glmnet(x.infer, y.infer, type.measure='auc', alpha=1, 
                         family='binomial', standardize=TRUE, seed=1)
plot(lasso.infer$lambda, lasso.infer$cvm, xlab='Lambda', ylab='AUC', main='Lasso')
lasso.infer.lambda <- lasso.infer$lambda.min
lasso.infer.auc <- max(lasso.infer$cvm)

# Coefficients for both models
coef(lasso.infer, s='lambda.min')

infer2.model <- glm(red_win ~ R_odds, data=infer2, family='binomial')
summary(infer2.model)

# Construct 95% normal confidence intervals
infer.low.norm <- rep(0, 3)
infer.high.norm <- rep(0, 3)
infer.summ <- summary(infer2.model)
infer.summ$coef
infer.low.norm <- infer.summ$coef[,1] - 1.96*infer.summ$coef[,2]
infer.high.norm <- infer.summ$coef[,1] + 1.96*infer.summ$coef[,2]
infer.CI.norm <- data.frame(infer.low.norm, infer.high.norm)
colnames(infer.CI.norm) <- c('Low', 'High')
infer.CI.norm








