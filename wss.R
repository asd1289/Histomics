library(skitools)
library(data.table)

df = fread("100_subsample.csv", header = T) #Reading in the file
drop <- c("Unnamed: 0") #Cleaning the file
df <- df[, !(colnames(df) %in% drop)]
df_s = scale(df) #Scaling the df
wss<- (nrow(df_s)-1)*sum(apply(df_s,2,var)) # Calculating variance for k == 1
for (i in 2:15) wss[i] <- sum(kmeans(df_s, centers = i)$withinss) # wss for k > 1
plot(1:15, wss, type="b", xlab="Number of Clusters",ylab="Within groups sum of squares", main="Seven samples together") # plotting the graph

