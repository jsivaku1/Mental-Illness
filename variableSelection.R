library(glmnet)
library(aod)
library(ggplot2)
library(stringr)
library(stats)
library(leaps)

#############################Depression#####################################
df = read.csv("odds_df.csv",header = TRUE)
class_var = subset(df, select = c("Anxiety","Schizophrenia", "Disease") )
df = df[,!(names(df) %in% c("Anxiety","Schizophrenia", "Disease"))]
head(df)
head(class_var)
df$Gender = as.integer(as.factor(df$Gender))
df$age = as.integer(as.factor(df$age))
regfit_full = regsubsets(Depression~.-1,df, nvmax = 24)
reg_summary = summary(regfit_full)
names(reg_summary)
par(mfrow = c(2,2))
plot(reg_summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(reg_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
adj_r2_max = which.max(reg_summary$adjr2) # 11
points(adj_r2_max, reg_summary$adjr2[adj_r2_max], col ="red", cex = 2, pch = 20)
plot(reg_summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
bic_min = which.min(reg_summary$bic) # 6
points(bic_min, reg_summary$bic[bic_min], col = "red", cex = 2, pch = 20)
cat(paste(shQuote(names(coef(regfit_full, 21))), collapse=","))
plot(regfit_full, scale = "adjr2")
cat(paste(shQuote(names(coef(regfit_full, 12))), collapse=","))
plot(regfit_full, scale = "bic")


#############################Anxiety#####################################
df = read.csv("odds_df.csv",header = TRUE)
class_var = subset(df, select = c("Depression","Schizophrenia", "Disease") )
df = df[,!(names(df) %in% c("Depression","Schizophrenia", "Disease"))]
df$Gender = as.integer(as.factor(df$Gender))
df$age = as.integer(as.factor(df$age))
regfit_full = regsubsets(Anxiety~.-1,df, nvmax = 24)
reg_summary = summary(regfit_full)
names(reg_summary)
par(mfrow = c(2,2))
plot(reg_summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(reg_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
adj_r2_max = which.max(reg_summary$adjr2) # 11
points(adj_r2_max, reg_summary$adjr2[adj_r2_max], col ="red", cex = 2, pch = 20)
plot(reg_summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
bic_min = which.min(reg_summary$bic) # 6
points(bic_min, reg_summary$bic[bic_min], col = "red", cex = 2, pch = 20)
cat(paste(shQuote(names(coef(regfit_full, 21))), collapse=","))
plot(regfit_full, scale = "adjr2")
cat(paste(shQuote(names(coef(regfit_full, 15))), collapse=","))
plot(regfit_full, scale = "bic")


#############################Schizophrenia#####################################
df = read.csv("odds_df.csv",header = TRUE)
class_var = subset(df, select = c("Depression","Anxiety", "Disease") )
df = df[,!(names(df) %in% c("Depression","Anxiety", "Disease"))]
df$Gender = as.integer(as.factor(df$Gender))
df$age = as.integer(as.factor(df$age))
regfit_full = regsubsets(Schizophrenia~.-1,df, nvmax = 24)
reg_summary = summary(regfit_full)
names(reg_summary)
par(mfrow = c(2,2))
plot(reg_summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(reg_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
adj_r2_max = which.max(reg_summary$adjr2) # 11
points(adj_r2_max, reg_summary$adjr2[adj_r2_max], col ="red", cex = 2, pch = 20)
plot(reg_summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
bic_min = which.min(reg_summary$bic) # 6
points(bic_min, reg_summary$bic[bic_min], col = "red", cex = 2, pch = 20)
cat(paste(shQuote(names(coef(regfit_full, 13))), collapse=","))
plot(regfit_full, scale = "adjr2")
cat(paste(shQuote(names(coef(regfit_full, 8))), collapse=","))
plot(regfit_full, scale = "bic")


#############################Disease#####################################
df = read.csv("odds_df.csv",header = TRUE)
class_var = subset(df, select = c("Depression","Anxiety", "Schizophrenia") )
df = df[,!(names(df) %in% c("Depression","Anxiety", "Schizophrenia"))]
df$Gender = as.integer(as.factor(df$Gender))
df$age = as.integer(as.factor(df$age))
regfit_full = regsubsets(Disease~.-1,df, nvmax = 24)
reg_summary = summary(regfit_full)
names(reg_summary)
par(mfrow = c(2,2))
plot(reg_summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(reg_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
adj_r2_max = which.max(reg_summary$adjr2) # 11
points(adj_r2_max, reg_summary$adjr2[adj_r2_max], col ="red", cex = 2, pch = 20)
plot(reg_summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
bic_min = which.min(reg_summary$bic) # 6
points(bic_min, reg_summary$bic[bic_min], col = "red", cex = 2, pch = 20)
cat(paste(shQuote(names(coef(regfit_full, 20))), collapse=","))
plot(regfit_full, scale = "adjr2")
cat(paste(shQuote(names(coef(regfit_full, 15))), collapse=","))
plot(regfit_full, scale = "bic")




