#!/usr/bin/R
# Copyright (c)2018 Accenture and/or its affiliates.  All Rights Reserved.  
# You may not use, copy, modify, and/or distribute this code and/or its documentation without permission from Accenture.
# Please contact the Applied Intelligence team and/or Frode Huse Gjendem (lead) with any questions.

# brief: This is the starter script for the Accenture Datathon 2018 Competition.

rm(list = ls()) # Clear workspace.

# -----1. Set configuration & Data Import.----- 
Sys.setlocale("LC_TIME", "English")

# ----- AUC function -----
#' It computes the AUC for 64bit numbers.
#' @param actual is the actual output (i.e., gound truth).
#' @param predicted is the prediction itself.
#' @param decimals are the number of decimals to compute AUC.
#' @return the AUC of the prediction.
my.AUC <- function (actual, predicted, decimals = 6) {
  predicted <- round(predicted, decimals)
  r <- as.numeric(rank(predicted))
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos *  n_neg)
  return(auc)
}

# Enter your input data and output data paths below.
PATH = getwd() # Otherwise use your own path
OUTPATH = getwd()
# Set the input data folder as default path.
setwd(PATH)

# ----- Data Import -----
# Read the test files.
test       <- read.csv("test.csv", header=T, stringsAsFactors = F)

# -----2. Data Transformation.----- 
submission <- test[,c("ID")]

# -----3. Model.----- 
submission$MDR <- 0

# -----5. Save the submission.----- 
# Write the final CSV file.
write.csv(submission, file=paste0(OUTPATH,'/sample-submission.csv'), row.names = F)

# Please, remember than in order to make the submission you need to create a .zip file with the csv

# Clear memory.
rm(list = ls()) 