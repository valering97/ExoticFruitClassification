from pandas import DataFrame, concat
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

def feature_scaling(df: DataFrame, method: str = 'Standard') -> DataFrame:
    """
    Scales the features of the dataset using the specified method.
    """
    try:
        X = df.drop(columns='Frutto')
        y = df['Frutto']
        feature_names = X.columns

        if method == 'Standard':
            preprocessing = ColumnTransformer(
                transformers=[
                    ('standard_scaling', StandardScaler(), feature_names)
                ],
                remainder='passthrough'
            )
        elif method == 'MinMax':
            preprocessing = ColumnTransformer(
                transformers=[
                    ('min_max_scaling', MinMaxScaler(), feature_names)
                ],
                remainder='passthrough'
            )
        else:
            raise ValueError('Choose MinMax or Standard')

        X_scaled = preprocessing.fit_transform(X)
        X_scaled = DataFrame(X_scaled, columns=feature_names)

        df_transformed = concat([X_scaled, y], axis=1)

        return df_transformed

    except Exception as e:
        print(f"Error in feature scaling: {e}")
        return df


def data_split(df: DataFrame, target: str, test_size: float = 0.2) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Splits a DataFrame into training and testing sets.
    """
    try:
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test

    except KeyError:
        raise ValueError(f"The target column '{target}' was not found in the DataFrame.")
    except ValueError as e:
        raise ValueError(f"Invalid value for parameters: {e}")