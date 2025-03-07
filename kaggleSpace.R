library(readr)
library(caret)
library(discretization)
library(arc)
library(rJava)
library(RWeka)
library(corrplot)
library(ggplot2)
library(labeling)
library(farver)
library(tidyr)
library(recipes)
library(stringr)

# Data load
dataTrain <- read.csv("~/train.csv", sep = ",", na.strings = c("", "NA"))
dataTest <- read.csv("~/test.csv", sep = ",", na.strings = c("", "NA"))

# Data visualization
ggplot(dataTrain, mapping = aes(x = Transported, fill = factor(Transported))) + geom_bar(color="black")
ggplot(dataTrain, mapping = aes(x = VIP, fill = factor(Transported))) + geom_bar(color="black")
ggplot(dataTrain, mapping = aes(x = VRDeck, fill = factor(Transported))) + geom_histogram(color="black")

valores_na_por_variable <- colSums(is.na(dataTrain))
print(valores_na_por_variable)
summary(dataTrain$Age)

# Data processing with recipes
obj_recipe <- recipe(Transported ~., data =  dataTrain)
rm_var1 <- obj_recipe %>% step_rm(VIP, Name, HomePlanet, Destination)
na_impute <- rm_var1 %>%  step_impute_mode(CryoSleep) %>% 
                            step_impute_mean(Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck)

mutate_var <- na_impute %>% step_mutate(CabinDeck = str_split(Cabin, "/", simplify = T)[,1],
                                     Num = ifelse(is.na(Cabin), NA, str_split(Cabin, "/", simplify = T)[,2]),
                                     CabinSide = ifelse(is.na(Cabin), NA, str_split(Cabin, "/", simplify = T)[,3]),
                                     Group = str_split(PassengerId, "_", simplify = T)[,1]) %>%
                            step_factor2string(PassengerId) %>% 
                            step_string2factor(CabinDeck, CabinSide) %>% 
                            step_mutate(GroupNum = as.numeric(Group), CabinNum = as.numeric(Num))

rm_var2 <- mutate_var %>% step_rm(Cabin, Group, Num)
na_impute2 <- rm_var2 %>% step_impute_mode(CabinDeck, CabinSide) %>% step_impute_mean(CabinNum)
norm_tran <- na_impute2 %>% step_normalize(Age, GroupNum, CabinNum, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck)

trained_recipe <- prep(norm_tran, training = dataTrain)
data_train_prep <- bake(trained_recipe, new_data = dataTrain)
data_test_prep  <- bake(trained_recipe, new_data = dataTest)

# Another data visualization
valores_na_por_variable <- colSums(is.na(data_train_prep))
print(valores_na_por_variable)

ggplot(data_train_prep, mapping = aes(x = CabinSide, fill = factor(Transported))) + geom_bar(color="black")
ggplot(data_train_prep, mapping = aes(x = CabinDeck, fill = factor(Transported))) + geom_bar(color="black")
ggplot(data_train_prep, mapping = aes(x = CryoSleep, fill = factor(Transported))) + geom_bar(color="black")
ggplot(data_train_prep, mapping = aes(x = Age, fill = factor(Transported))) + geom_histogram(color="black")
ggplot(data_train_prep, mapping = aes(x = GroupNum, fill = factor(Transported))) + geom_histogram(color="black")
ggplot(data_train_prep, mapping = aes(x = CabinNum, fill = factor(Transported))) + geom_histogram(color="black")
ggplot(data_train_prep, mapping = aes(x = CabinSide, fill = factor(Transported))) + geom_bar(color="black")
ggplot(data_train_prep, mapping = aes(x = RoomService, fill = factor(Transported))) + geom_histogram(color="black")

# Model training
fitControl <- trainControl(
  method = "cv",
  number = 5)

set.seed(22)
data_train_prep <- subset(data_train_prep, select = -PassengerId)
fitResults <- 
  train(
    Transported ~., 
    data = data_train_prep, 
    method = "gbm", 
    trControl = fitControl
  )

# Predictions
predictions <- predict(fitResults, data_test_prep)
print(predictions)

# Exporting predictions
result_df <- data.frame(PassengerId = data_test_prep$PassengerId, Transported = predictions)
write.csv(result_df, file = "~/resultado_pred.csv", row.names = FALSE)

