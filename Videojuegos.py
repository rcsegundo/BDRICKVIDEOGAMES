import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.write("""# Predicción del éxito de un videojuego""")
st.write("Esta aplicación utiliza un árbol de decisión entrenado con un dataset sintético de videojuegos.")

st.header("Datos de evaluación")

def user_input_features():
    Platform = st.selectbox(
        "Plataforma:",
        options=["PC", "PlayStation", "Xbox", "Nintendo", "Mobile"]
    )

    Genre = st.selectbox(
        "Género:",
        options=["Acción", "Aventura", "Deportes", "RPG", "Estrategia", "Simulación"]
    )

    Year = st.number_input(
        "Año de lanzamiento:",
        min_value=2010, max_value=2026, value=2024, step=1
    )

    Price = st.number_input(
        "Precio (USD):",
        min_value=0.0, max_value=100.0, value=39.99, step=0.01
    )

    Reviews = st.number_input(
        "Número de reseñas:",
        min_value=0, max_value=20000, value=1000, step=100
    )

    Hours_Played = st.number_input(
        "Horas promedio jugadas:",
        min_value=0.0, max_value=200.0, value=25.0, step=1.0
    )

    Multiplayer = st.selectbox(
        "¿Tiene multijugador?",
        options=["No", "Sí"]
    )

    DLC = st.selectbox(
        "¿Tiene contenido descargable (DLC)?",
        options=["No", "Sí"]
    )

    Rating = st.number_input(
        "Calificación promedio:",
        min_value=1.0, max_value=10.0, value=7.5, step=0.1
    )

    platform_map = {
        "PC": 0, "PlayStation": 1, "Xbox": 2, "Nintendo": 3, "Mobile": 4
    }

    genre_map = {
        "Acción": 0, "Aventura": 1, "Deportes": 2,
        "RPG": 3, "Estrategia": 4, "Simulación": 5
    }

    user_input_data = {
        "Platform": platform_map[Platform],
        "Genre": genre_map[Genre],
        "Year": Year,
        "Price": Price,
        "Reviews": Reviews,
        "Hours_Played": Hours_Played,
        "Multiplayer": 1 if Multiplayer == "Sí" else 0,
        "DLC": 1 if DLC == "Sí" else 0,
        "Rating": Rating
    }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

videojuegos = pd.read_csv("Videojuegos.csv", encoding="utf-8")

X = videojuegos.drop(columns="Successful")
Y = videojuegos["Successful"]

classifier = DecisionTreeClassifier(
    max_depth=8,
    criterion="entropy",
    min_samples_leaf=10,
    max_features=7,
    random_state=0
)

classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader("Predicción")

if prediction[0] == 0:
    st.write("❌ El videojuego probablemente NO será exitoso.")
elif prediction[0] == 1:
    st.write("🎮 El videojuego probablemente será EXITOSO.")
else:
    st.write("Sin predicción")
