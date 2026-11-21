library(tidyverse)
library(fpp3)
library(BVAR)


install.packages("BVAR", repos = "https://cloud.r-project.org")

library(BVAR)


# Part 2 - ARIMA vs. ETS

# 2.1a - Import the liquor sales data

liquor_raw <- read_csv(
  "data/MRTSSM4453USN.csv",
  na = ".",
  col_types = cols(
    observation_date = col_date(format = "%Y-%m-%d"),
    MRTSSM4453USN = col_double()
  )
)

liquor <- liquor_raw |>
  transmute(
    Month = yearmonth(observation_date),
    sales = MRTSSM4453USN
  ) |>
  as_tsibble(index = Month) |>
  filter(Month <= yearmonth("2019 Dec"))

has_gaps(liquor)



# 2.1b - Monthly liquor sales

liquor |>
  autoplot(sales) +
  labs(
    title = "Monthly Liquor Store Sales",
    x = "Year",
    y = "Sales (Millions of Dollars)"
  ) +
  theme_minimal()



# 2.1b - Seasonal pattern of log sales

liquor |>
  gg_season(log(sales)) +
  labs(
    title = "Seasonal Pattern of Liquor Store Sales",
    x = "Month",
    y = "Log of Sales"
  ) +
  theme_minimal()



# 2.2a - Fit ARIMA and ETS models

liquor_fit <- liquor |>
  model(
    arima = ARIMA(log(sales)),
    ets = ETS(log(sales))
  )

# ARIMA model results
liquor_fit |>
  select(arima) |>
  report()

# ETS model results
liquor_fit |>
  select(ets) |>
  report()



# 2.2b - Check ARIMA residuals

liquor_fit |>
  select(arima) |>
  gg_tsresiduals()


# Ljung-Box test at lag 24

liquor_fit |>
  select(arima) |>
  augment() |>
  features(.innov, ljung_box, lag = 24, dof = 6)



# 2.3a - Forecasting horse race

# Create expanding training samples
liquor_cv <- liquor |>
  stretch_tsibble(.init = 120, .step = 1) |>
  filter(.id < max(.id))

# Fit the three forecasting models
liquor_cv_fit <- liquor_cv |>
  model(
    snaive = SNAIVE(sales),
    ets = ETS(log(sales)),
    arima = ARIMA(log(sales))
  )

# Save the fitted models
saveRDS(liquor_cv_fit, "results/liquor_cv_fit.rds")

# Count the forecast origins
n_distinct(liquor_cv$.id)



# 2.3b - Compare forecasting accuracy

# Generate one-month-ahead forecasts
liquor_cv_fc <- liquor_cv_fit |>
  forecast(h = 1)

# Calculate RMSE and MAE
liquor_accuracy <- liquor_cv_fc |>
  accuracy(
    liquor,
    measures = list(RMSE = RMSE, MAE = MAE)
  )

# Display results
liquor_accuracy

# Save results for the Quarto report
saveRDS(liquor_accuracy, "results/liquor_accuracy.rds")



# 2.4 - Diebold-Mariano tests

liquor_errs <- liquor_cv_fc |>
  as_tibble() |>
  left_join(
    as_tibble(liquor) |>
      select(Month, actual = sales),
    by = "Month"
  ) |>
  mutate(e = actual - .mean) |>
  select(Month, .model, e) |>
  pivot_wider(
    names_from = .model,
    values_from = e
  )

# Check the forecast errors
head(liquor_errs)



# DM test: ARIMA vs ETS

dm_arima_ets <- forecast::dm.test(
  liquor_errs$arima,
  liquor_errs$ets,
  h = 1,
  power = 2
)

dm_arima_ets



# DM test: ARIMA vs Seasonal Naive

dm_arima_snaive <- forecast::dm.test(
  liquor_errs$arima,
  liquor_errs$snaive,
  h = 1,
  power = 2
)

dm_arima_snaive



# Save both DM test results

saveRDS(dm_arima_ets, "results/dm_arima_ets.rds")

saveRDS(dm_arima_snaive, "results/dm_arima_snaive.rds")


saveRDS(dm_arima_ets, "results/dm_arima_ets.rds")
saveRDS(dm_arima_snaive, "results/dm_arima_snaive.rds")



# Part 3 - Forecast Combination

# 3a - Combine ARIMA and ETS forecasts

liquor_combo_fc <- liquor_cv_fit |>
  mutate(combo = (arima + ets) / 2) |>
  forecast(h = 1)

# Calculate the forecasting accuracy
combo_accuracy <- liquor_combo_fc |>
  accuracy(
    liquor,
    measures = list(RMSE = RMSE, MAE = MAE)
  )

# Display results
combo_accuracy


# Save results for Quarto
saveRDS(combo_accuracy, "results/combo_accuracy.rds")


# 3b - Correlation between ARIMA and ETS errors

error_correlation <- cor(
  liquor_errs$arima,
  liquor_errs$ets
)

# Display the correlation
error_correlation

# Save the result for Quarto
saveRDS(error_correlation, "results/error_correlation.rds")



# Part 4 - VAR vs. BVAR

# 4.1a - Import quarterly FRED-QD data

qd_raw <- read_csv(
  "data/fredqd_2026-09.csv",
  show_col_types = FALSE
) |>
  slice(-(1:2))

# Check the imported data
head(qd_raw)



# 4.1b - Prepare the eight economic variables

# Function to calculate growth rates
growth <- function(x) 100 * difference(log(x))

# Create the quarterly dataset
qd <- qd_raw |>
  mutate(date = as.Date(sasdate, "%m/%d/%Y")) |>
  transmute(
    date,
    gdp    = growth(GDPC1),
    cons   = growth(PCECC96),
    inv    = growth(GPDIC1),
    emp    = growth(PAYEMS),
    infl   = growth(GDPCTPI),
    unrate = difference(UNRATE),
    ffr    = difference(FEDFUNDS),
    gs10   = difference(GS10)
  ) |>
  filter(
    date >= as.Date("1960-03-01"),
    date <= as.Date("2019-12-01")
  )

# Convert to a matrix
y <- qd |> select(-date) |> as.matrix()

# Check the data
dim(y)
head(y)



# 4.1c - Set up expanding-window cross-validation

# Number of observations
N <- nrow(y)

# Forecast origins
origins <- 120:(N - 1)

# Actual values for evaluating forecasts
actual <- y[121:N, ]

# Check the number of forecasts
length(origins)

# Check the first and last forecast dates
qd$date[origins[1] + 1]
qd$date[tail(origins, 1) + 1]


# 4.2 - Comparing VAR(1) and VAR(4)

# Function to estimate a VAR model and forecast
var_ols_fc <- function(z, p) {
  e <- embed(z, p + 1)
  M <- ncol(z)
  B <- coef(lm(e[, 1:M] ~ e[, -(1:M)]))
  x <- c(1, t(z[nrow(z):(nrow(z) - p + 1), ]))
  x %*% B
}

# Generate forecasts for both models
fc_var1 <- NULL
fc_var4 <- NULL

for (T in origins) {
  fc_var1 <- rbind(fc_var1, var_ols_fc(y[1:T, ], 1))
  fc_var4 <- rbind(fc_var4, var_ols_fc(y[1:T, ], 4))
}

# Calculate RMSE for each variable
rmse_by_var <- function(f) {
  sqrt(colMeans((actual - f)^2))
}

# VAR(1) results
rmse_by_var(fc_var1)

# VAR(4) results
rmse_by_var(fc_var4)



# Save VAR results for Quarto

var_ols_results <- tibble(
  Variable = colnames(y),
  `VAR(1)` = rmse_by_var(fc_var1),
  `VAR(4)` = rmse_by_var(fc_var4)
)

saveRDS(var_ols_results, "results/var_ols_results.rds")


# 4.3a - Bayesian VAR with Minnesota prior

# Function to estimate BVAR(4)
bvar_cv <- function(lambda) {
  
  pr <- bv_priors(
    hyper = "lambda",
    mn = bv_mn(
      lambda = bv_lambda(
        mode = lambda,
        sd = 0.4,
        min = 0.999 * lambda,
        max = 1.001 * lambda
      ),
      b = 0
    )
  )
  
  fc <- NULL
  
  for (T in origins) {
    fit_T <- bvar(
      y[1:T, ],
      lags = 4,
      priors = pr,
      n_draw = 2000,
      n_burn = 1000,
      verbose = FALSE
    )
    
    x <- c(1, t(y[T:(T - 3), ]))
    fc <- rbind(fc, x %*% coef(fit_T))
  }
  
  fc
}



# Run BVAR with lambda = 0.3

set.seed(422)

fc_bvar_03 <- bvar_cv(0.3)

# Save the forecasts
saveRDS(fc_bvar_03, "results/fc_bvar_03.rds")

# Check the results
dim(fc_bvar_03)


# 4.3b - BVAR with stronger shrinkage

set.seed(422)

fc_bvar_005 <- bvar_cv(0.05)

# Save the forecasts
saveRDS(fc_bvar_005, "results/fc_bvar_005.rds")

# Check the dimensions
dim(fc_bvar_005)



# Compare all four forecasting models

var_results <- tibble(
  Variable = colnames(y),
  `VAR(1)` = rmse_by_var(fc_var1),
  `VAR(4)` = rmse_by_var(fc_var4),
  `BVAR(4), lambda = 0.3` = rmse_by_var(fc_bvar_03),
  `BVAR(4), lambda = 0.05` = rmse_by_var(fc_bvar_005)
)

# Display the RMSE table
var_results |>
  knitr::kable(digits = 3)

# Save the table for Quarto
saveRDS(var_results, "results/var_results.rds")


# 4.3c - Diebold-Mariano tests

# GDP growth: VAR(4) vs BVAR(4)
e_var4_gdp <- actual[, "gdp"] - fc_var4[, 1]
e_bvar_gdp <- actual[, "gdp"] - fc_bvar_03[, 1]

dm_gdp <- forecast::dm.test(
  e_var4_gdp,
  e_bvar_gdp,
  h = 1,
  power = 2
)

# Display GDP results
dm_gdp



# Inflation: VAR(4) vs BVAR(4)

e_var4_infl <- actual[, "infl"] - fc_var4[, 5]
e_bvar_infl <- actual[, "infl"] - fc_bvar_03[, 5]

dm_infl <- forecast::dm.test(
  e_var4_infl,
  e_bvar_infl,
  h = 1,
  power = 2
)

# Display inflation results
dm_infl



# Save DM tests for Quarto

saveRDS(dm_gdp, "results/dm_gdp.rds")

saveRDS(dm_infl, "results/dm_infl.rds")



# Part 5 - Factor Models with FRED-MD

# 5.1a - Import the monthly FRED-MD data

md_raw <- read_csv(
  "data/fredmd_2026-09.csv",
  show_col_types = FALSE
)

# Save the transformation codes
tcode <- unlist(md_raw[1, -1])

# Remove the transformation row and create dates
md <- md_raw[-1, ] |>
  mutate(date = as.Date(sasdate, "%m/%d/%Y"))

# Check the imported data
head(md |> select(date, INDPRO, PAYEMS, CPIAUCSL, FEDFUNDS))



# 5.1b - Transform the FRED-MD data

transform_series <- function(x, code) {
  d <- function(v) c(NA, diff(v))
  
  switch(
    as.character(code),
    "1" = x,
    "2" = d(x),
    "3" = d(d(x)),
    "4" = log(x),
    "5" = 100 * d(log(x)),
    "6" = 100 * d(d(log(x))),
    "7" = d(x / dplyr::lag(x) - 1)
  )
}



# Apply transformations to all series

fm <- map2_dfc(
  md[names(tcode)],
  tcode,
  transform_series
) |>
  mutate(date = md$date, .before = 1) |>
  filter(
    date >= as.Date("1960-03-01"),
    date <= as.Date("2019-12-01")
  )

fm <- fm[, colSums(is.na(fm)) == 0]

dim(fm)





# 5.1c - Check transformation codes

tcode[c("INDPRO", "PAYEMS", "CPIAUCSL", "FEDFUNDS")]



# 5.2a - Principal Component Analysis

# Create a matrix without the date column
Z <- fm |>
  select(-date) |>
  as.matrix()

# Standardize the data and estimate principal components
pc <- prcomp(scale(Z))

# Show the variation explained by the first five factors
summary(pc)$importance[, 1:5]

# Calculate the total variation explained by the first three
sum(summary(pc)$importance[2, 1:3])



# 5.2b - Plot the first economic factor

# Adjust the factor's sign to match industrial production
F1 <- pc$x[, 1] * sign(cor(pc$x[, 1], fm$INDPRO))

# Create the graph
tibble(date = fm$date, F1 = F1) |>
  ggplot(aes(x = date, y = F1)) +
  geom_line() +
  labs(
    title = "First Factor from FRED-MD",
    x = "Year",
    y = "First Principal Component"
  ) +
  theme_minimal()



# 5.2c - Variables most correlated with the first two factors

# Top 8 variables correlated with Factor 1
factor1_corr <- sort(
  abs(cor(pc$x[, 1], Z))[1, ],
  decreasing = TRUE
)[1:8]

# Top 8 variables correlated with Factor 2
factor2_corr <- sort(
  abs(cor(pc$x[, 2], Z))[1, ],
  decreasing = TRUE
)[1:8]

# Display results
factor1_corr
factor2_corr



# 5.3 - Industrial production forecasting

# Industrial production growth
ip <- fm$INDPRO

# Number of observations
n <- nrow(Z)

# Initial training sample ends in December 1989
init <- which(fm$date == as.Date("1989-12-01"))

# Store forecasting errors
e_ar <- NULL
e_fa <- NULL

# Expanding-window forecasting
for (T in init:(n - 1)) {
  
  # Re-estimate factors using data available at time T
  F_T <- prcomp(scale(Z[1:T, ]))$x[, 1:3]
  
  # Prepare the regression data
  d <- data.frame(
    y_next = ip[2:T],
    y = ip[1:(T - 1)],
    F_T[1:(T - 1), ]
  )
  
  # Latest observed values
  new <- data.frame(
    y = ip[T],
    t(F_T[T, ])
  )
  
  # Forecast errors
  e_ar <- c(
    e_ar,
    ip[T + 1] - predict(lm(y_next ~ y, d), new)
  )
  
  e_fa <- c(
    e_fa,
    ip[T + 1] - predict(lm(y_next ~ ., d), new)
  )
}

# Compare RMSE
ip_results <- c(
  RMSE_AR = sqrt(mean(e_ar^2)),
  RMSE_FA = sqrt(mean(e_fa^2))
)

ip_results

saveRDS(ip_results, "results/ip_results.rds")



# 5.4 - Direct forecast of 12-month CPI inflation

direct_race <- function(series, h = 12, r = 3) {
  
  lv <- md |>
    mutate(
      log_x = log(.data[[series]]),
      g = 100 * (log_x - lag(log_x))
    ) |>
    filter(
      date >= as.Date("1960-03-01"),
      date <= as.Date("2019-12-01")
    )
  
  # Growth over the next h months
  yh <- 100 * (lead(lv$log_x, h) - lv$log_x)
  
  # Latest monthly growth
  g <- lv$g
  
  e_ar <- NULL
  e_fa <- NULL
  
  for (T in init:(n - h)) {
    
    # Re-estimate factors at each forecast origin
    F_T <- prcomp(scale(Z[1:T, ]))$x[, 1:r, drop = FALSE]
    
    # Use only observations with known future outcomes
    i <- 1:(T - h)
    
    d <- data.frame(
      yh = yh[i],
      g = g[i],
      F_T[i, , drop = FALSE]
    )
    
    new <- data.frame(
      g = g[T],
      F_T[T, , drop = FALSE]
    )
    
    # Forecast errors for both models
    e_ar <- c(
      e_ar,
      yh[T] - predict(lm(yh ~ g, d), new)
    )
    
    e_fa <- c(
      e_fa,
      yh[T] - predict(lm(yh ~ ., d), new)
    )
  }
  
  tibble(
    series,
    forecasts = length(e_ar),
    RMSE_AR = sqrt(mean(e_ar^2)),
    RMSE_FA = sqrt(mean(e_fa^2)),
    MAE_AR = mean(abs(e_ar)),
    MAE_FA = mean(abs(e_fa)),
    DM_pvalue = forecast::dm.test(
      e_ar, e_fa,
      h = h,
      power = 2,
      varestimator = "bartlett"
    )$p.value
  )
}


# Forecast 12-month CPI inflation

inflation_results <- direct_race("CPIAUCSL")

# Display results
inflation_results

# Calculate percentage improvement in RMSE
100 * (
  inflation_results$RMSE_AR - inflation_results$RMSE_FA
) / inflation_results$RMSE_AR

# Save results for Quarto
saveRDS(inflation_results, "results/inflation_results.rds")
