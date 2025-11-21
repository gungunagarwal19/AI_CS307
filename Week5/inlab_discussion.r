
data_path <- "2020_bn_nb_data.txt"


needed <- c("bnlearn", "e1071", "bnclassify", "dplyr")
for (p in needed) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cloud.r-project.org")
  library(p, character.only = TRUE)
}

df <- read.table(data_path, header = TRUE, sep = "\t", stringsAsFactors = TRUE, strip.white = TRUE)
cat("Rows, columns:", dim(df), "\n")
str(df)

if (!is.factor(df$QP)) df$QP <- as.factor(df$QP)
levels(df$QP) <- sort(levels(df$QP))  


set.seed(123)
bn_struct <- hc(df)       
cat("Learned BN arcs (first 20 shown):\n")
print(arcs(bn_struct)[1:min(20, nrow(arcs(bn_struct))), ])

bn_fitted <- bn.fit(bn_struct, df, method = "mle")
cat("\nCPT for node PH100:\n")
print(bn_fitted$PH100)  


evidence_expr <- (EC100 == "DD") & (IT101 == "CC") & (MA101 == "CD")

ph_levels <- levels(df$PH100)
probs <- setNames(numeric(length(ph_levels)), ph_levels)

n_samples <- 1e5
for (lev in ph_levels) {
  event_expr <- as.formula(paste0("PH100 == '", lev, "'"))
  prob_est <- cpquery(bn_fitted,
                      event = (PH100 == lev),
                      evidence = eval(parse(text = "EC100 == 'DD' & IT101 == 'CC' & MA101 == 'CD'")),
                      method = "lw",
                      n = n_samples)
  probs[lev] <- prob_est
}
cat("\nEstimated P(PH100 | EC100=DD, IT101=CC, MA101=CD):\n")
print(round(probs, 4))

most_likely <- names(which.max(probs))
cat("\nMost likely grade for PH100 given the evidence:", most_likely, "\n")

library(e1071)
set.seed(2024)
n_reps <- 20
acc_nb <- numeric(n_reps)  
seeds <- sample.int(10000, n_reps)

for (i in seq_len(n_reps)) {
  set.seed(seeds[i])
  train_idx <- sample(seq_len(nrow(df)), size = floor(0.7 * nrow(df)), replace = FALSE)
  train <- df[train_idx, , drop = FALSE]
  test  <- df[-train_idx, , drop = FALSE]
  

  nb_model <- naiveBayes(QP ~ ., data = train, laplace = 1)  
  preds <- predict(nb_model, test, type = "class")
  acc <- mean(preds == test$QP)
  acc_nb[i] <- acc
  cat(sprintf("NB run %2d: seed=%d, test n=%d, accuracy=%.4f\n", i, seeds[i], nrow(test), acc))
}
cat("\nNaive Bayes (independent) results over", n_reps, "runs:\n")
cat("Mean accuracy:", mean(acc_nb), "\n")
cat("SD accuracy:  ", sd(acc_nb), "\n")


library(bnclassify)
set.seed(2024)
acc_tan <- numeric(n_reps)

for (i in seq_len(n_reps)) {
  set.seed(seeds[i])
  train_idx <- sample(seq_len(nrow(df)), size = floor(0.7 * nrow(df)), replace = FALSE)
  train <- df[train_idx, , drop = FALSE]
  test  <- df[-train_idx, , drop = FALSE]
  
  tan_model <- tan_cl("QP", data = train)

  preds <- predict(tan_model, test) 
  if (is.list(preds) && !is.null(preds$class)) preds_vec <- preds$class else preds_vec <- as.factor(preds)
  
  acc <- mean(preds_vec == test$QP)
  acc_tan[i] <- acc
  cat(sprintf("TAN run %2d: seed=%d, test n=%d, accuracy=%.4f\n", i, seeds[i], nrow(test), acc))
}

cat("\nTAN (dependent) results over", n_reps, "runs:\n")
cat("Mean accuracy:", mean(acc_tan), "\n")
cat("SD accuracy:  ", sd(acc_tan), "\n")

results_df <- data.frame(
  run = seq_len(n_reps),
  seed = seeds,
  acc_naivebayes = acc_nb,
  acc_tan = acc_tan
)
print(results_df)
cat("\nSummary:\n")
cat(sprintf("NaiveBayes: mean=%.4f, sd=%.4f\n", mean(acc_nb), sd(acc_nb)))
cat(sprintf("TAN (dependent): mean=%.4f, sd=%.4f\n", mean(acc_tan), sd(acc_tan)))
