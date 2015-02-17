# *****************************************
# Author: Ruth Dannenfelser
# Date: Feb 17, 2015
#
# Construct various plots for the
# Spam / Ham data.
#
# *****************************************
library("ggplot2")
setwd("/Genomics/ogtr04/rd6/COS424/hw1")

# feature variance.
# ---------------------
features = read.delim("feature_variance.txt", header=F)
features <- features[order(features$V1),]
clean = subset(features, V1 > 1)

# some stats. 
length(features$V1) # total features.
sum (features$V1 < 1) # features with variance less than 1.  8776 
tail(features) # words with the highest variance (font, style, td, 3d, com, size)
head(features) # words with the lowest variance (pmda, 1eq, 3jw)
head(clean) # words with the lowest var after filtering (see, cid, gid, d0, simsun)

# histograms before and after filtering.
q1 <- qplot(V1, data=features, binwidth = 5, geom="histogram") + xlab("variance") + theme_bw()
q1
q2 <- qplot(V1, data=clean, binwidth = 5, geom="histogram") + xlab("variance") + theme_bw() 
q2

# save the histograms
ggsave(file="figures/feature_var_before.pdf", plot=q1, width=8, height=5)
ggsave(file="figures/feature_var_after.pdf", plot=q2, width=8, height=5)

