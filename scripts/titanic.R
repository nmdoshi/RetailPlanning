# This file is generated and can be overwritten.
# This is generated base on titanic.csv
suppressWarnings(library("v2viz"))
library(dplyr)
library(stringr)

refine_dataframe <-function(df) {
  df_new <- df %>% 
      mutate_all(funs(as.character)) %>%
      mutate(survived_value=ifelse(survived==1,"Y","N"))  %>%
      mutate(pclass_value=ifelse(pclass==1,"first",ifelse(pclass==2,"second","third"))) %>%
      select(-`cabin`) %>%
      select(-`boat`) %>%
      select(-`body`) %>%
      select(-`home-dest`) %>%
      filter(!is.na(`embarked`) & `embarked` != '') %>%
      filter(!is.na(`age`) & `age` != '') %>%
      mutate(`fare` = as.double(`fare`)) %>%
      mutate(log_fare=log10(fare)) %>%
      mutate(`age` = as.integer(`age`)) %>%
      mutate(age_bin=ifelse(age<6,0,ifelse(age<12,1,ifelse(age<18,2,ifelse(age<40,3,ifelse(age<65,4,ifelse(age<80,5,6))))))) %>%
      mutate(log_fare_bin=ifelse(log_fare<0,0,ifelse(log_fare>8,9,as.integer(log_fare)+1))) %>%
      select(-`age`) %>%
      select(-`fare`) %>%
      select(-`log_fare`) 
  return (df_new)
}

# output file if NULL return df, otherwise, write to file 
refine_file <- function (input, output=NULL, overwrite="FALSE") { 
  if (is.na(overwrite)) { overwrite = "FALSE"}
  if (is.na(output)) { output = NULL}
  if(!file.exists(input)) { 
    return (paste(input, "file does not exist")); 
  } 
  if (!is.null(output) && file.exists(output) && !(overwrite == "TRUE")) { 
    return (paste(output, "already exists.")); 
  } 
  df <- read.csv(input, check.names=FALSE, stringsAsFactors=FALSE) 
  df <- refine_dataframe(df) 
  if (!is.null(output)) { 
    write.csv(df, file = output, row.names=FALSE) 
    return(paste("Writing to", output, "file is complete")) 
  } else { 
    return (df) 
  } 
}

# main entry for Rscript
args <- commandArgs(trailingOnly = TRUE)
opts<-c();
for (arg in args) {
   x <- lapply(strsplit(arg, split="="), trimws);
   opts[x[[1]][1]] <-x[[1]][2];
}
# validate
required <- c(opts['input']);
missingRequired <- any(is.na(required));
if (missingRequired) {
   print("Missing required parameter");
} else {
   refine_file(opts['input'], opts['output'], opts['overwrite']);
}
