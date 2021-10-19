#############################################
# Forecasting SC decisions
# Matthew Ingram
# UAlbany
# Last revised: 2021-10-19
#
#############################################

########################
# SET DIRECTORY
########################

#path <- 'C:/Users/mi122167/Dropbox/SUNYAlbany/PredictingSCOTUS/Data/SCDB'

setwd(path)

########################
# SET ENVIRONMENT
########################

#install.packages("pacman", repos="https://cloud.r-project.org")
library(pacman)
p_load(aod,
       ggplot2,
       caTools,
       rpart,
       rpart.plot,
       randomForest,
       magrittr,
       visreg,
       rattle # fancy rpart plots
)

sessionInfo()

################################################
# Load data
################################################

# load raw data
#load("./SCDB_2021_01_justiceCentered_Docket.Rdata/SCDB_2021_01_justiceCentered_Docket.RData")
# rename object to simpler name
#data <- SCDB_2021_01_justiceCentered_Docket # need this originally with first loading of data
# remove original object that was loaded
#rm(SCDB_2021_01_justiceCentered_Docket)

# load data from web; not working
#temp <- tempfile()
#download.file("http://scdb.wustl.edu/_brickFiles/2021_01/SCDB_2021_01_justiceCentered_Docket.Rdata.zip",temp)
#data <- load(unz(temp, "SCDB_2021_01_justiceCentered_Docket.RData"))
#unlink(temp)

# save partial file with only data since 2000
#temp <- subset(data, term>1999)
#write.csv(temp, file="./data/working/SCDB_2021_01_justiceCentered_Docket_2000s.csv")

# read csv file
data <- read.csv(file="./data/working/SCDB_2021_01_justiceCentered_Docket_2000s.csv")

#################################################
# Inspect/explore data

names(data)
str(data)
head(data)
summary(data)

#################################################
# Define outcome variables
#################################################

#The variable "direction" captures the ideological direction of the individual justice's vote in each case. 
#See online codebook for SCDB data with description of all variables here: <http://http://scdb.wustl.edu//documentation.php?s=1>.
#The variable is coded 1=conservative and 2=liberal. 
#We want to recode to make simpler to interpret: 0=conservative and 1=liberal. 
#So, we generate a new variable 'directiondum' that captures this binary character.

data$directiondum <- as.factor(data$direction - 1)  # 1 = liberal; 0 = conservative
# check
data %>%
  subset(, select=c("direction", "directiondum")) %>%
  summary()

table(data$direction, data$directiondum)


# later, after prediction, translate to affirm/reverse based on lower court direction
# see coding notes at WUSTL site: http://wusct.wustl.edu/media/trees.pdf

#############################################################
# Define predictors
#############################################################

#We want to define explanatory variables (predictors) that we can use. 
#We should start with the variables used by Martin et al. (2002).
#Later, we can also consider expanding to the variables raised by critiques and commentaries on their work, 
#as well as other variables we have raised in class.

#Here is the variable that captures the direction of the lower court decision.

data$lcdirectiondum[data$lcDispositionDirection==1] <- 1
data$lcdirectiondum[data$lcDispositionDirection==2] <- 0

# note that lcDispositionDirection has value of 3 (undefinable), but there 
# do not appear to be any observations there

table(data$lcdirectiondum, data$lcDispositionDirection)

# later, after prediction, translate to affirm/reverse based on lower court direction
# see coding notes at WUSTL site: http://wusct.wustl.edu/media/trees.pdf

####################################
# If want to exclude all nonunanimous cases
#data <- subset(data, data$minVotes>0)
####################################


#Other variables we could include are identified below, but we could consider recoding them here.

# Subset by variable value

#Shows most welfare-related cases were decided by 1990.

#data <- subset(data, issue==20180 | issue==20190)
#summary(data)
hist(subset(data, (issue==20180 | issue==20190))$term)

##################################################
### Basic logit model:

# commented out if using only two issues (21080 and 21090)
m1 <- glm(directiondum ~ 
            # as.factor(issueArea) + 
            as.factor(lawType), data = subset(data, issue==20180 | issue==20190), family = "binomial", na.action(na.omit))
summary(m1)


### Odds ratios:

exp(cbind(OR = coef(m1), confint(m1)))


### Probit model:

m2 <- glm(directiondum ~ 
            #as.factor(issueArea) + 
            as.factor(lawType), data = subset(data, issue==20180 | issue==20190), family = "binomial"(link="probit"), na.action(na.omit))
summary(m2)


### Odds ratios of probit model:

exp(cbind(OR = coef(m2), confint(m2)))

###########################################################
### Logit model with justices factored out:

m3 <- glm(directiondum ~ 
            as.factor(lawType) +
            as.factor(lcDispositionDirection) +
            as.factor(justiceName), data = subset(data, 
                                                  (issue==20180 | issue==20190) 
                                                  & (term >= 1967)), 
                                                  family = "binomial", 
                                                  na.action(na.omit))
summary(m3)

##############################################################################

##############################################################################
# The Method: Classification Trees

#See Kastellec (2010) for nice, accessible introduction to classification trees.

## Install and open required library (only need to open if already have installed).

# install and open rpart
#install.packages("rpart", repos="https://cloud.r-project.org")
library(rpart)
#install.packages("rpart.plot", repos="https://cloud.r-project.org")
library(rpart.plot)

## Subset data according to each justice.

#Here, we break the data up according to each justice, so that we end up with several mini-data sets. 
#Each of the mini-data sets contains the votes of only that justice.

# For justices on court as of OT2021
barrett <- subset(data, justiceName=="ACBarrett") # all barrett votes
# susbet to keep only cases from OT2020; this would be same for barrett
barrett20 <- subset(data, justiceName=="ACBarrett" & term==2020) # all barrett votes

gorsuch <- subset(data, justiceName=="NMGorsuch") # all gorsuch
gorsuch1819 <- subset(data, justiceName=="NMGorsuch" & term>2017 & term<2020) #  gorsuch OT2018 and 2019
gorsuch20 <- subset(data, justiceName=="NMGorsuch" & term==2020) #  gorsuch OT2020

roberts <- subset(data, justiceName=="JGRoberts") # all roberts
roberts1819 <- subset(data, justiceName=="JGRoberts" & term>2017 & term<2020) #  roberts OT2018 and 2019
roberts20 <- subset(data, justiceName=="JGRoberts" & term==2020) #  roberts OT2020

breyer <- subset(data, justiceName=="SGBreyer") # all breyer
breyer1819 <- subset(data, justiceName=="SGBreyer" & term>2017 & term<2020) #  breyer OT2018 and 2019
breyer20 <- subset(data, justiceName=="SGBreyer" & term==2020) #  breyer OT2020

sotomayor <- subset(data, justiceName=="SSotomayor") # all sotomayor
sotomayor1819 <- subset(data, justiceName=="SSotomayor" & term>2017 & term<2020) #  sotomayor OT2018 and 2019
sotomayor20 <- subset(data, justiceName=="SSotomayor" & term==2020) #  sotomayor OT2020

kagan <- subset(data, justiceName=="EKagan") # all kagan
kagan1819 <- subset(data, justiceName=="EKagan" & term>2017 & term<2020) #  kagan OT2018 and 2019
kagan20 <- subset(data, justiceName=="EKagan" & term==2020) #  kagan OT2020

thomas <- subset(data, justiceName=="CThomas") # all thomas
thomas1819 <- subset(data, justiceName=="CThomas" & term>2017 & term<2020) #  thomas OT2018 and 2019
thomas20 <- subset(data, justiceName=="CThomas" & term==2020) #  thomas OT2020

kavanaugh <- subset(data, justiceName=="BMKavanaugh") # all kavanaugh
kavanaugh1819 <- subset(data, justiceName=="BMKavanaugh" & term>2017 & term<2020) #  kavanaugh OT2018 and 2019
kavanaugh20 <- subset(data, justiceName=="BMKavanaugh" & term==2020) #  kavanaugh OT2020

alito <- subset(data, justiceName=="SAAlito") # all alito
alito1819 <- subset(data, justiceName=="SAAlito" & term>2017 & term<2020) #  alito from kavanaugh to death of RBG
alito20 <- subset(data, justiceName=="SAAlito" & term==2020) #  alito  OT2020


# Other justices
ginsburg1 <- subset(data, justiceName=="RBGinsburg" & term<2002 & term>1995) # term< 2002 to see if can 
# match natural court and tree from Martin et al 2002
ginsburg2 <- subset(data, justiceName=="RBGinsburg") # all ginsburg
oconnor1 <- subset(data, justiceName=="SDOConnor" & term<2002 & term>1994)  # term<2002 to see if can
# match tree from Martin et al 2002
oconnor2 <- subset(data, justiceName=="SDOConnor") # all oconnor
scalia <- subset(data, justiceName=="AScalia") # all scalia
kennedy <- subset(data, justiceName=="AMKennedy") # all kennedy

#
#And we could continue to include all other justices, if interested.


###########################################################################3
# Cautionary Notes

#If you are having a hard time following this thus far, it would be a good idea 
#to go back and review some of the basic documentation regarding the data at 
# the SCDB website.

#CHECK THE PDF AT WUSTL SITE REGARDING CODINGS.

#ALSO, NOTE USE OF VOTE OF OTHER JUSTICES IN RELATED DOCS (i.e., INTERDEPENDENCE).

#GENERATE FACTOR VARIRABLES WITH LOW, REASONABLE AND READABLE NUMBER OF CODINGS.

#ALSO, COULD RECODE OUTCOME AS AFFIRM OR REVERSE (BASED on xtabs showing low frequency of contadictory combos)

#ALSO, TRY ADDING OTHER VARS USED BY KATZ, BOMMARITO, AND BLACKMAN, includign disagreement below.

##############################################################################
#Split the data.
##############################################################################

# Sotomayor Tree, OT2020

### Subset training and testing data 

#using package caTools
df <- sotomayor20
set.seed(3000)
spl = sample.split(df$directiondum, SplitRatio = 0.7)
Train = subset(df, spl==TRUE)
Test = subset(df, spl==FALSE)

SotomayorTree = rpart(directiondum ~ caseSource + issueArea + petitioner + respondent + 
                        lcdirectiondum + lawType,# + term, 
                      data = Train, method="class", control = rpart.control(minsplit = 5, minbucket= 2))
prp(SotomayorTree)
fancyRpartPlot(SotomayorTree)
par(oma=c(0,0,0,1))


## Predictions

PredictCART = predict(SotomayorTree, newdata = Test, type = "class")
# if use factored out predictors above, predict command may generate errors if have categories
# that were in training data but not in testing data
preds <- table(Test$directiondum, PredictCART)
preds

## Accuracy
#Accuracy = (topleft + bottomright)/total

(preds[1]+preds[4])/(sum(preds[1:4]))
# 72.73%


# Sotomayor Tree, all with term, but just criminal procedure cases

### Subset training and testing data 

#using package caTools
set.seed(3000)
sotomayorsub <- subset(sotomayor, issueArea==1)
spl = sample.split(sotomayorsub$directiondum, SplitRatio = 0.7)
Train = subset(sotomayorsub, spl==TRUE)
Test = subset(sotomayorsub, spl==FALSE)

SotomayorTree = rpart(directiondum ~ 
                        caseSource  + petitioner + respondent +
                        lcDispositionDirection + lawType,# + term, 
                      data = Train, method="class", control = rpart.control(minsplit = 5, minbucket= 2))
# took issueArea out since focused only on issueArea==1
prp(SotomayorTree)
fancyRpartPlot(SotomayorTree)
par(oma=c(0,0,0,1))


## Predictions

PredictCART = predict(SotomayorTree, newdata = Test, type = "class")
# if use factored out predictors above, predict command may generate errors if have categories
# that were in training data but not in testing data
preds <- table(Test$directiondum, PredictCART)
preds

## Accuracy

#Accuracy = (topleft + bottomright)/total

(preds[1]+preds[4])/(sum(preds[1:4]))
# 67.1%

###############################################
# Breyer Tree, OT2020

### Subset training and testing data 

#using package caTools
df <- breyer20
set.seed(3000)
spl = sample.split(df$directiondum, SplitRatio = 0.7)
Train = subset(df, spl==TRUE)
Test = subset(df, spl==FALSE)

BreyerTree = rpart(directiondum ~ caseSource + issueArea + petitioner + respondent +
                     lcDispositionDirection + lawType,# + term, 
                   data = Train, method="class", control = rpart.control(minsplit = 5, minbucket= 2))
prp(BreyerTree)
fancyRpartPlot(BreyerTree)
par(oma=c(0,0,0,1))


## Predictions

PredictCART = predict(BreyerTree, newdata = Test, type = "class")
# if use factored out predictors above, predict command may generate errors if have categories
# that were in training data but not in testing data
preds <- table(Test$directiondum, PredictCART)
preds

## Accuracy
#Accuracy = (topleft + bottomright)/total

(preds[1]+preds[4])/(sum(preds[1:4]))
# 50%

###############################################################
# Kagan Tree, OT2020

### Subset training and testing data 

#using package caTools
df <- kagan20
set.seed(3000)
spl = sample.split(df$directiondum, SplitRatio = 0.7)
Train = subset(df, spl==TRUE)
Test = subset(df, spl==FALSE)

KaganTree = rpart(directiondum ~ caseSource + issueArea + lcDispositionDirection + lawType,# + term, 
                  data = Train, method="class", control = rpart.control(minsplit = 5, minbucket= 2))
prp(KaganTree)
fancyRpartPlot(KaganTree)
par(oma=c(0,0,0,1))


## Predictions

PredictCART = predict(KaganTree, newdata = Test, type = "class")
# if use factored out predictors above, predict command may generate errors if have categories
# that were in training data but not in testing data
preds <- table(Test$directiondum, PredictCART)
preds

## Accuracy
#Accuracy = (topleft + bottomright)/total

(preds[1]+preds[4])/(sum(preds[1:4]))
# 45%

##############################################################################
## Thomas tree OT2020

#Note: generally hinged on Scalia vote while two justices were on court

### Subset training and testing data 

df <- thomas20
set.seed(3000)
spl = sample.split(df$directiondum, SplitRatio = 0.7)
Train = subset(df, spl==TRUE)
Test = subset(df, spl==FALSE)

### Run tree

ThomasTree = rpart(directiondum ~ 
                     caseSource + issueArea + petitioner + respondent + lcDispositionDirection + lawType, 
                   data = Train, method="class", control = rpart.control(minsplit = 5, minbucket= 2))

# to complete, would need to recode caseSource and lawType to simplify values of each
prp(ThomasTree)

### Predictions

PredictCART = predict(ThomasTree, newdata = Test, type = "class")
# if use factored out predictors above, predict command may generate errors if have categories
# that were in training data but not in testing data
preds <- table(Test$directiondum, PredictCART)
preds

### Accuracy
#Accuracy = (topleft + bottomright)/total

(preds[1]+preds[4])/(sum(preds[1:4]))
# 72.73%

######################################################################
# Gorsuch Tree, OT2020

### Subset training and testing data 

df <- gorsuch20
set.seed(3000)
spl = sample.split(df$directiondum, SplitRatio = 0.7)
Train = subset(df, spl==TRUE)
Test = subset(df, spl==FALSE)

GorsuchTree = rpart(directiondum ~ caseSource + issueArea + petitioner + respondent +
                      lcDispositionDirection + lawType,# + term, 
                    data = Train, method="class", control = rpart.control(minsplit = 5, minbucket= 2))
prp(GorsuchTree)
fancyRpartPlot(GorsuchTree)
par(oma=c(0,0,0,1))


## Predictions

PredictCART = predict(GorsuchTree, newdata = Test, type = "class")
# if use factored out predictors above, predict command may generate errors if have categories
# that were in training data but not in testing data
preds <- table(Test$directiondum, PredictCART)
preds

## Accuracy
#Accuracy = (topleft + bottomright)/total

(preds[1]+preds[4])/(sum(preds[1:4]))
# 50.%

##################################################################33
# Kavanaugh Tree, just OT2020

### Subset training and testing data 

df <- kavanaugh20
set.seed(3000)
spl = sample.split(df$directiondum, SplitRatio = 0.7)
Train = subset(df, spl==TRUE)
Test = subset(df, spl==FALSE)

KavanaughTree = rpart(directiondum ~ caseSource + issueArea + petitioner + respondent +
                        lcDispositionDirection + lawType,# + term, 
                      data = Train, method="class", control = rpart.control(minsplit = 5, minbucket= 2))
prp(KavanaughTree)
fancyRpartPlot(KavanaughTree)
par(oma=c(0,0,0,1))


## Predictions

PredictCART = predict(KavanaughTree, newdata = Test, type = "class")
# if use factored out predictors above, predict command may generate errors if have categories
# that were in training data but not in testing data
preds <- table(Test$directiondum, PredictCART)
preds

## Accuracy
#Accuracy = (topleft + bottomright)/total

(preds[1]+preds[4])/(sum(preds[1:4]))
# 72.73%

############################################################################

############################################################################

# Additional Material

Additional, more advanced material follows below. Please do not move on to that material unless you are confident you understand above material. We will return to material below as a class at a later point in semester.


# Approach 3: 
# Random Forest (e.g. with Sotomayor, OT 2020)

p_load(reprtree)
# may not load correctly
# if errors, then backup is to source file with code
source("reprtree.R")

#Harder to interpret, but this is not a problem if goal is empirical prediction rather than causal explanation.

df <- sotomayor20
set.seed(3000)
spl = sample.split(df$directiondum, SplitRatio = 0.7)
Train = subset(df, spl==TRUE)
Test = subset(df, spl==FALSE)


# using randomForest package

# Build random forest model
#GinsburgForest = randomForest(directiondum ~ caseSource + issueArea + petitioner + respondent + lcDispositionDirection + lawType, data = Train, ntree=200, nodesize=25 ) #gives error

# Convert outcome to factor // This should be done already above
#Train$directiondum = as.factor(Train$directiondum)
#Test$directiondum = as.factor(Test$directiondum)

# Try again
SotomayorForest = randomForest(directiondum ~ 
                                 caseSource + issueArea + petitioner + respondent +
                                 lcDispositionDirection + lawType, 
                               data = Train, ntree=200, nodesize=25, na.action=na.omit) 
#plot(SotomayorForest)
visreg(SotomayorForest)
reprtree:::plot.getTree(SotomayorForest) # gets a single tree
reprtree:::plot.reprtree(SotomayorForest) # plots a representative tree
par(oma=c(0,0,0,1))

# gave error re 'issueArea', so dropped that var

### Predictions

# Make predictions
PredictForest = predict(SotomayorForest, newdata = Test)
# if use factored out predictors above, predict command may generate errors if have categories
# that were in training data but not in testing data
preds <- table(Test$directiondum, PredictForest)
preds


### Accuracy
#Accuracy = (topleft + bottomright)/total

(preds[1]+preds[4])/(sum(preds[1:4]))
# 63.64%



###########################################################################
## Basic logit/probit models:

### Basic logit model:

# commented out if using only two issues (21080 and 21090)
m1 <- glm(directiondum ~ as.factor(issueArea) + as.factor(lawType), data = sotomayor20, family = "binomial", na.action(na.omit))
summary(m1)

### Odds ratios:

#exp(cbind(OR = coef(m1), confint(m1)))

# predict probabilities
probabilities <- m1 %>% predict(sotomayor20, type = "response")
head(probabilities)

predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
head(predicted.classes)

mean(predicted.classes == test.data$diabetes)

### Probit model:

#m2 <- glm(directiondum ~ as.factor(issueArea) + as.factor(lawType), data = subset(data, justice==106 & naturalCourt==1705), family = "binomial"(link="probit"), na.action(na.omit))
#summary(m1)

### Odds ratios of probit model:

#exp(cbind(OR = coef(m1), confint(m1)))

### Logit model with justices factored out:

#m3 <- glm(directiondum ~ as.factor(justiceName), data = subset(data, naturalCourt==1705), family = "binomial", na.action(na.omit))
#summary(m2)



######################################################################

# save data
save.image("./data/working/working20211019.RData")

#end
