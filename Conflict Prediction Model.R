# Load in required packages 
library(xgboost)
library(stargazer)
library(ggplot2)
library(diag)
library(DiagrammeR)
library(haven)
library(cowplot)
library(googleway)
library(ggrepel)
library(ggspatial)
library(libwgeom)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(dplyr)

#Load in data
data = read.dta("estimationdata_withlags.dta")
data.trim = data %>% filter(year <= 2009) %>%
            select(conflict, lc1, lc2, ltsc0, ltsc1, ltsc2, loi, loic1, loic2, lois0, let, letc1, letc2, 
                   lets0, lli, limc1, limc2, lims0, lyo, lyoc1, lyoc2, lyos0, llpo, lpoc1, lpoc2, lpos0, 
                   led, ledc1, ledc2, leds0, llin, ledn, lyon, lnc1, lnc1c1, lnc1c2, lnc1ts0, r4, r6, r7) 

#Create training and test data
train = data.trim %>% filter(year <= 2000)
test = data.trim %>% filter(year > 2000)
train.labels = train$conflict
test.labels = test$conflict
train.data  = train %>% select(!c(statename, year, ltsc1, ltsc2, conflict))
test.data  = test %>% select(!c(statename, year, ltsc1, ltsc2, conflict))

#Convert to xgb format
train.data = as.matrix(train.data)
test.data = as.matrix(test.data)

xgb.train = xgb.DMatrix(data = train.data, label = train.labels)
xgb.test = xgb.DMatrix(data = test.data, label = test.labels)

###Hyperparameter tuning
# Create empty lists
lowest_error = c()
iter = c()

# Create 10,000 rows with random hyperparameters
set.seed(123)
for (iter in 1:10000){
  param <- list(booster = "gbtree",
                objective = "multi:softprob",
                max_depth = sample(3:10, 1),
                eta = runif(1, .01, .3),
                subsample = runif(1, .7, 1),
                colsample_bytree = runif(1, .6, 1),
                min_child_weight = sample(0:10, 1)
  )
  parameters <- as.data.frame(param)
  parameters_list[[iter]] <- parameters
}

# Create object that contains all randomly created hyperparameters
parameters_df = do.call(rbind, parameters_list)

# Use randomly created parameters to create 10,000 XGBoost-models
for (row in 1:nrow(parameters_df)){
  set.seed(20)
  mdcv <- xgb.train(data=xgb.train,
                    booster = "gbtree",
                    objective = "multi:softprob",
                    max_depth = parameters_df$max_depth[row],
                    eta = parameters_df$eta[row],
                    subsample = parameters_df$subsample[row],
                    colsample_bytree = parameters_df$colsample_bytree[row],
                    min_child_weight = parameters_df$min_child_weight[row],
                    num_class = 3,
                    nrounds= 300,
                    eval_metric = "mlogloss",
                    early_stopping_rounds= 30,
                    print_every_n = 100,
                    watchlist = list(val1= xgb.train, val2= xgb.test),
                    verbose = T
  )
  iter[row] = row
  lowest_error[row] = mdcv$best_score
}

# Create object that contains all accuracy's
lowest_error_df = do.call(rbind, lowest_error_list)

# Bind columns of accuracy values and random hyperparameter values
randomsearch = cbind(lowest_error_df, parameters_df)
min(lowest_error)
iter[lowest_error == min(lowest_error)]

#Train tuned model
params = list(
  booster = "gbtree",
  eta = 0.1949223,
  max_depth = 7,
  subsample = 0.7375644,
  colsample_by_tree = 0.7264748,
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = 3
)

xgb.fit = xgb.train(params = params, 
                    data = xgb.train, 
                    nrounds = 10000, 
                    nthreads = 1, 
                    early_stopping_rounds = 100, 
                    watchlist=list(val1=xgb.train,val2=xgb.test), 
                    verbose = 1)

#Get predictions
xgb.pred = as.data.frame(predict(xgb.fit, test.data, reshape = T))
colnames(xgb.pred) = c(0, 1, 2)
xgb.pred$Prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])
xgb.pred$Country = test$statename
xgb.pred$Year = test$year
colnames(xgb.pred) = c("P.None", "P.Small", "P.Large", "Prediction", "Country", "Year")

#Print 10 countries with highest chance of conflict in 2009
xgb.pred %>% filter(Year == 2009) %>% 
  mutate(P.Conflict = 1- P.None) %>%
  select(Country, P.Conflict) %>%
  slice_max(P.Conflict, n = 10) %>%
  stargazer(summary = F, rownames = F, type = "html", title = "Countries with Highest Risk of Conflict, 2009", 
            out = "table.htm")

#Plot the probability of conflict in Somalia over time
somalia = xgb.pred %>% filter(Country == "Somalia") %>% mutate(P.Conflict = P.Large + P.Small)
somalia.graph = somalia %>% ggplot(aes(x = Year, y = P.Conflict)) + 
  geom_line() +
  geom_point() + 
  ggtitle("Probability of Conflict in Somalia") + 
  scale_x_continuous(n.breaks = 9) +
  theme(plot.title = element_text(size=22))

###Plot a world map colored by the type of predicted conflict

#Load in geographic data
world = ne_countries(scale = "medium", returnclass = "sf")
joined = world %>% left_join(subset(xgb.pred, Year == 2009), by = c("sovereignt" = "Country"))
joined$Prediction = ifelse(joined$Prediction == 2, "Large", ifelse(joined$Prediction == 1, "Small", "None"))
joined$Prediction = factor(joined$Prediction, levels = c("None", "Small", "Large"), ordered = T)
joined = joined %>% mutate(P.Conflict = P.Small + P.Large)

#Create graph
pred.map = ggplot(data = joined) + 
  geom_sf(aes(fill = Prediction)) + 
  #theme_void() + 
  scale_color_viridis_d(palette = "plasma") + 
  ggtitle("Map of Predicted Conflicts in 2009") +
  theme(plot.title = element_text(size=18)) +
  theme(legend.position = "bottom")

####################################
###Summary of variable importance###
####################################
imp = xgb.importance(colnames(train), xgb.fit)
imp = imp[order(-Gain), ]
imp$Feature[1:10] = c("Duration of Peace", "ln(Population)", "Nducation", "Neighbor Education", "Youth x Lag(Large Conflict)", 
                      "ln(Population) x Lag(Small Conflict)", "ln(IMR)", "Youth", "ln(Neighbor IMR)", "Neighbor Youth")

#Create dot plot
imp[1:10,] %>% ggplot() + geom_point(aes(x = Gain, y = reorder(Feature, Gain))) + 
  ggtitle("Gain of 10 Most Important Variables") + ylab("Variable") + 
  theme(axis.text.y = element_text(angle = 45), axis.title.y = element_blank(),
        title = element_text(size = 16), axis.text.x = element_text(size=11))

#Graph of model accuracy over time
xgb.pred$Outcome = test.labels
xgb.pred %>% group_by(Year) %>% mutate(Correct = Prediction == Outcome) %>%
  summarize(Accuracy = mean(Correct)) %>% ggplot(aes(x = Year, y = Accuracy)) +
  geom_point() + geom_line() + ggtitle("Model Accuracy over Time") + 
  scale_x_continuous(n.breaks = 9) +
  theme(plot.title = element_text(size=16), 
        axis.text.x = element_text(size = 11),
        axis.text.y = element_text(size =11),
        text)

###########################################
### Binary Prediction of Small Conflict ###
###########################################
#Create variable to predict and clean data
data.trim$small.ind =ifelse(data.trim$conflict == 1, 1, 0)
train = data.trim %>% filter(year <= 2000)
test = data.trim %>% filter(year > 2000)
train.labels = train$small.ind
test.labels = test$small.ind
train.data  = train %>% select(!c(statename, year, ltsc1, ltsc2, conflict, lc2, small.ind))
test.data  = test %>% select(!c(statename, year, ltsc1, ltsc2, conflict, lc2, small.ind))

#Convert to xgb format
train.data = as.matrix(train.data)
test.data = as.matrix(test.data)

xgb.train = xgb.DMatrix(data = train.data, label = train.labels)
xgb.test = xgb.DMatrix(data = test.data, label = test.labels)

#Specify the parameters for the decision forest
params = list(
  booster = "gbtree",
  eta = 0.1949223,
  max_depth = 7,
  subsample = 0.7375644,
  colsample_by_tree = 0.7264748,
  objective = "binary:logistic",
  eval_metric = "logloss"
)

#Train the model
xgb.fit = xgb.train(params = params, 
                    data = xgb.train, 
                    nrounds = 10000, 
                    nthreads = 1, 
                    early_stopping_rounds = 100, 
                    watchlist=list(val1=xgb.train,val2=xgb.test), 
                    verbose = 1)

#Get predictions
xgb.pred = as.data.frame(predict(xgb.fit, test.data, reshape = T))
colnames(xgb.pred) = c("Prob")

#Establish 0.5 as the cutoff for predicting conflict
xgb.pred$Prediction = ifelse(xgb.pred$Prob >= 0.5, 1, 0)

#Print accuracy
mean(xgb.pred$Prediction == test.labels)

###########################################
### Binary Prediction of Large Conflict ###
###########################################
#Create variable to be predicted and clean data
data.trim$large.ind = ifelse(data.trim$conflict == 2, 1, 0)
train = subset(data.trim, year <= 2000)
test = subset(data.trim, year > 2000)
train.labels = train$large.ind
test.labels = test$large.ind
train  = train %>% select(!c(statename, year, ltsc1, ltsc2, conflict, lc1, small.ind, large.ind))
test  = test %>% select(!c(statename, year, ltsc1, ltsc2, conflict, lc1, small.ind, large.ind))

#Convert data tto xgb formatt
train.data = as.matrix(train)
test.data = as.matrix(test)

xgb.train = xgb.DMatrix(data = train.data, label = train.labels)
xgb.test = xgb.DMatrix(data = test.data, label = test.labels)

#Specify the parameters for the decision forest
params = list(
  booster = "gbtree",
  eta = 0.1949223,
  max_depth = 7,
  subsample = 0.7375644,
  colsample_by_tree = 0.7264748,
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = 3
)

#Fit the model
xgb.fit = xgb.train(params = params, 
                    data = xgb.train, 
                    nrounds = 10000, 
                    nthreads = 1, 
                    early_stopping_rounds = 100, 
                    watchlist=list(val1=xgb.train,val2=xgb.test), 
                    verbose = 1)

#Get preedictions
xgb.pred = as.data.frame(predict(xgb.fit, test.data, reshape = T))
colnames(xgb.pred) = c("Prob")

#Establish 0.5 as the prediction cutoff
xgb.pred$Prediction = ifelse(xgb.pred$Prob >= 0.5, 1, 0)

#Print accuracy
mean(xgb.pred$Prediction == test.labels)

###########################################################
### Predicting th Outbreak of a Conflict Within 3/5 Years ###
###########################################################
data.trim = data %>% filter(year <= 2009) %>% 
                     select(statename, year, conflict, lc1, lc2, ltsc0, ltsc1, ltsc2, loi, loic1, loic2, lois0, 
                            let, letc1, letc2, lets0, lli, limc1, limc2, lims0, lyo, lyoc1, lyoc2, lyos0, llpo, 
                            lpoc1, lpoc2, lpos0, led, ledc1, ledc2, leds0, llin, ledn, lyon, lnc1, lnc1c1, lnc1c2, 
                            lnc1ts0, r4, r6, r7)

#Create leading variables for conflict within the next 5 years
data.trim = data.trim %>% group_by(statename) %>% 
  mutate(lead1 = lead(conflict, 1, default = 0), lead2 = lead(conflict, 2, default = 0), 
         lead3 = lead(conflict, 3, default = 0), lead4 = lead(conflict, 4, default = 0))

#Define variables for the presence of conflict within the next 3/5 years
data.trim = data.trim %>% rowwise() %>% 
  mutate(within3 = max(conflict, lead1, lead2), within5 = max(conflict, lead1, lead2, lead3, lead4)) %>%
  ungroup()

#Clean data
data.trim$within3 = factor(data.trim$within3)
data.trim$within5 = factor(data.trim$within5)
train = data.trim %>% filter(year <= 2000)
test = data.trim %>% filter(year > 2000 & year < 2005) 
train.labels = train$within5
test.labels = test$within5
train.data  = train %>% select(!c(statename, year, ltsc1, ltsc2, conflict, within3, within5, lead1, lead2, lead3, lead4))
test.data  = test %>% select(!c(statename, year, ltsc1, ltsc2, conflict, within3, within5, lead1, lead2, lead3, lead4))

#Convert to xgb format
train.data = as.matrix(train.data)
test.data = as.matrix(test.data)

xgb.train = xgb.DMatrix(data = train.data, label = train.labels)
xgb.test = xgb.DMatrix(data = test.data, label = test.labels)

#Specify parameters
params = list(
  booster = "gbtree",
  eta = 0.1949223,
  max_depth = 7,
  subsample = 0.7375644,
  colsample_by_tree = 0.7264748,
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = 3
)

#Train model
xgb.fit = xgb.train(params = params, 
                    data = xgb.train, 
                    nrounds = 10000, 
                    nthreads = 1, 
                    early_stopping_rounds = 100, 
                    watchlist=list(val1=xgb.train,val2=xgb.test), 
                    verbose = 1)

#Get predictions
xgb.pred = as.data.frame(predict(xgb.fit, test.data, reshape = T))
colnames(xgb.pred) = c(0, 1, 2)
xgb.pred$Prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])

#Print accuracy
mean(xgb.pred$Prediction == test.labels)