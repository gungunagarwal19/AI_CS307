library(bnlearn)
library(gRain)  
library(bnclassify)   
library(gRbase)
library(caret)       
set.seed(123)

datafile <- "codes/2020_bn_nb_data.txt"
df <- read.table(datafile, header = TRUE, sep = "", stringsAsFactors = TRUE)

str(df)
target_col <- names(df)[ncol(df)]
df[ , 1:(ncol(df)-1)] <- lapply(df[ , 1:(ncol(df)-1)], function(x) factor(as.character(x)))
df[[target_col]] <- factor(as.character(df[[target_col]]))


bn_struct <- hc(df)         
print(bn_struct)
bn_fitted <- bn.fit(bn_struct, data = df, method = "bayes") 

node_name <- "PH100"  
if(! node_name %in% names(bn_fitted)) stop("PH100 not found in data columns. Edit node_name variable.")
print(bn_fitted[[node_name]])

bn.fit.barchart(bn_fitted[[node_name]])

grain_net <- as.grain(bn_fitted)
grain_net <- compile(grain_net)
evidence_nodes <- c("EC100","IT101","MA101")
evidence_states <- c("DD","CC","CD")
grain_net_evid <- setEvidence(grain_net, nodes = evidence_nodes, states = evidence_states)
q <- querygrain(grain_net_evid, nodes = node_name)
print(q)

library(bnclassify)
runs <- 20
accs_nb <- numeric(runs)
set.seed(2025)
for(i in seq_len(runs)){
  # create stratified split to keep class distribution
  train_idx <- createDataPartition(df[[target_col]], p = 0.70, list = FALSE)
  train <- df[train_idx, ]
  test  <- df[-train_idx, ]

  # learn naive Bayes using bnclassify; use smoothing (alpha) to avoid zeros
  nb_model <- bnc('nb', class = target_col, dataset = train) 
  nb_model <- lp(nb_model, train, smooth = 1)   # learn parameters (laplace smoothing)

  # predict
  preds_prob <- predict(nb_model, test, prob = TRUE)
  preds <- predict(nb_model, test, prob = FALSE)

  # accuracy
  accs_nb[i] <- bnclassify:::accuracy(preds, test[[target_col]])
}
cat(sprintf("Naive Bayes: mean acc = %.4f, sd = %.4f (over %d runs)\n", mean(accs_nb), sd(accs_nb), runs))

print(accs_nb)

accs_tan <- numeric(runs)
set.seed(2025)
for(i in seq_len(runs)){
  train_idx <- createDataPartition(df[[target_col]], p = 0.70, list = FALSE)
  train <- df[train_idx, ]
  test  <- df[-train_idx, ]

  # learn TAN using Chow-Liu
  tan_model <- tan_cl(class = target_col, dataset = train)  # structure
  tan_model <- lp(tan_model, train, smooth = 1)             # parameters

  preds <- predict(tan_model, test)                        # class predictions
  accs_tan[i] <- bnclassify:::accuracy(preds, test[[target_col]])
}
cat(sprintf("TAN: mean acc = %.4f, sd = %.4f (over %d runs)\n", mean(accs_tan), sd(accs_tan), runs))
print(accs_tan)

results <- list(nb_acc = accs_nb, tan_acc = accs_tan, bn_struct = bn_struct, bn_fitted = bn_fitted, latest_PH100_query = q)
save(results, file = "bn_nb_results.RData")
cat("Done. Results saved in bn_nb_results.RData\n")
