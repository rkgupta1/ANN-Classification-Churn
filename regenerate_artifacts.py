import pickle

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def main():
    data = pd.read_csv("Churn_Modelling.csv")
    data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    label_encoder_gender = LabelEncoder()
    data["Gender"] = label_encoder_gender.fit_transform(data["Gender"])

    onehot_encoder_geo = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    geo_encoded = onehot_encoder_geo.fit_transform(data[["Geography"]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
        index=data.index,
    )

    data = pd.concat([data.drop("Geography", axis=1), geo_encoded_df], axis=1)

    X = data.drop("Exited", axis=1)
    y = data["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        callbacks=[early_stopping],
        verbose=0,
    )

    model.save("model.h5")

    with open("label_encoder_gender.pkl", "wb") as file:
        pickle.dump(label_encoder_gender, file)

    with open("onehot_encoder_geo.pkl", "wb") as file:
        pickle.dump(onehot_encoder_geo, file)

    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    print("Saved model.h5, scaler.pkl, label_encoder_gender.pkl, onehot_encoder_geo.pkl")
    print("Feature order:", list(scaler.feature_names_in_))


if __name__ == "__main__":
    main()
