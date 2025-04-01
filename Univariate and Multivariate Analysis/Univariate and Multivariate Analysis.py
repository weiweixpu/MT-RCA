import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import logit

# Read data from CSV file (please modify the file path)
file_path = "data/xiaoyingzhen/pas/paslinchuangxxtrain.csv"  # Replace with your file path
data = pd.read_csv(file_path)
print("First 5 rows of the data:")
print(data.head())

# Define the analysis variables
outcome_var = 'Label'
predictors = ['PAS_score', 'age', 'MRwork', 'Gravidities', 'Number_of_births', 'Prior_cesarean_deliveries', 'Placenta_Previa', 'Abortion_history', 'History_of_other_uterine_cavity_procedures', 'Assisted_reproduction']  # Replace with your list of predictor variable names


# ====================== Univariate Analysis ======================
def univariate_logistic_regression(data, outcome_var, predictors):
    results = []
    for predictor in predictors:
        try:
            formula = f"{outcome_var} ~ {predictor}"
            model = logit(formula, data=data).fit(disp=0)
            coef = model.params[1]
            or_ = np.exp(coef)
            p_value = model.pvalues[1]
            conf_int = np.exp(model.conf_int().iloc[1, :].values)
            results.append({
                'Predictor': predictor,
                'OR': round(or_, 3),
                '95% CI Lower': round(conf_int[0], 3),
                '95% CI Upper': round(conf_int[1], 3),
                'p-value': round(p_value, 3)
            })
        except Exception as e:
            print(f"Error while analyzing predictor {predictor}: {str(e)}")
            continue
    return pd.DataFrame(results)


# ====================== Run Analysis ======================
if __name__ == "__main__":
    # Univariate analysis
    uni_results = univariate_logistic_regression(data, outcome_var, predictors)
    print("\nUnivariate analysis results:")
    print(uni_results.to_string(index=False)) 
    # Multivariate analysis
    formula = f"{outcome_var} ~ {' + '.join(predictors)}"
    try:
        model = logit(formula, data=data).fit(disp=0)
        print("\nMultivariate analysis results:")
        print(model.summary())

        # Extract Multivariate OR and CI
        multi_or = pd.DataFrame({
            'Predictor': predictors,
            'OR': np.round(np.exp(model.params[predictors]), 3),
            '95% CI Lower': np.round(np.exp(model.conf_int().loc[predictors, 0]), 3),
            '95% CI Upper': np.round(np.exp(model.conf_int().loc[predictors, 1]), 3),
            'p-value': np.round(model.pvalues[predictors], 3)
        })
        print("\nFormatted multivariate results:")
        print(multi_or.to_string(index=False))
    except Exception as e:
        print(f"\nMultivariate analysis failed: {str(e)}")
