import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# Loss function MAE (Mean Absolute Error)
def custom_mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


'''
The number of components required to capture the give variance
Input:
train_scaled: scaled training data set (dataframe) for example with Standard Scaler
test_scaled:  scaled testing data set (dataframe) for example with Standard Scaler
'''
def pca_calc(train_scaled, test_scaled, variance):
    pca = PCA(variance)
    pca.fit(train_scaled)
    print(f'The number of components is: {pca.n_components_}')

    train_pca = pca.transform(train_scaled)
    train_pca_reconstructed = pca.inverse_transform(train_pca)
    train_reconstruction_variance = np.var(train_scaled - pca.inverse_transform(train_pca))
    print(f'The training reonstruction variance is: {train_reconstruction_variance}')

    test_pca = pca.transform(test_scaled)
    test_pca_reconstructed = pca.inverse_transform(test_pca)
    test_reconstruction_variance = np.var(test_scaled - pca.inverse_transform(test_pca))
    print(f'The testing reonstruction variance is: {test_reconstruction_variance}')

    train_loss = custom_mae(train_scaled, train_pca_reconstructed)
    print(f'MAE loss on train data using pca: {train_loss}')

    test_loss = custom_mae(test_scaled, test_pca_reconstructed)
    print(f'MAE loss on test data using pca: {test_loss}')

    return train_pca_reconstructed, test_pca_reconstructed