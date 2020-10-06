models_dict = {'lasso_cv': 'Lasso',
               'ridge_cv': 'Ridge',
               'xgb': 'GBRT', 'dt':
               'DT', 'infinite_dt':
               'Infinite DT',
               'rf': 'RF',
               'infinite_rf': 'Infinite RF',
               'spp_reg_less_bins': 'SPP Few Bins',
               'spp_reg_more_bins': 'SPP Many Bins'}

legend_dict = {'best_test_score': 'Test', 'best_train_score': 'Train'}

dataset_names_dict = {'auto_prices': 'Auto Prices',
                      'black_friday': 'Black Friday',
                      'NYC_taxis': 'NYC Taxis',
                      'la_crimes': 'L.A. Crimes'}

n_features_dict = {'auto_prices': 15,
                   'black_friday': 9,
                   'NYC_taxis': 14,
                   'la_crimes': 25}

n_samples_dict = {'auto_prices': 1000000,
                  'black_friday': 166821,
                  'NYC_taxis': 581835,
                  'la_crimes': 1468825}
