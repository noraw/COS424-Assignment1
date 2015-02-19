# *****************************************
# Author: Ruth Dannenfelser
# Date: Feb 17, 2015
#
# Construct various plots for the
# Spam / Ham data.
#
# *****************************************
library("ggplot2")
library("pROC")
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


# ROC curves
# ------------------
y = read.delim("trec07p_data/Test/test_emails_classes_0.txt", header=F)
predmNB = read.delim("predictions_mNB.txt", header=F)
predbNB = read.delim("predictions_bNB.txt", header=F)

mat <- data.frame(y, predmNB$V1, predmNB$V2)
names(mat) <- c("true", "predicted", "score")
mat <- mat[order(mat$score, decreasing=T),]

mat2 <- data.frame(y, predbNB$V1, predbNB$V2)
names(mat2) <- c("true", "predicted", "score")
mat2 <- mat2[order(mat2$score, decreasing=T),]

plot.roc(roc(mat$true, mat$score), col="#1c61b6")
lines.roc(roc(mat2$true, mat2$score), col="#008600")
legend("bottomright", legend=c("Multinomial NB", "Bernoulli NB"), col=c("#1c61b6", "#008600"), lwd=2)
       
       
