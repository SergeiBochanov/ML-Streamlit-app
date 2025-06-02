import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.title("Визуализация")

df = pd.read_csv("data/asteroids_processed.csv")

st.subheader("1. Тепловая карта")
fig = plt.figure(figsize=(5, 3))
sns.heatmap(df.corr(), annot = True)
st.pyplot(fig, use_container_width=False)

st.subheader("2. Countplot")
fig = plt.figure(figsize=(5, 3))
sns.countplot(df, x='hazardous')
st.pyplot(fig, use_container_width=False)

st.subheader("3. Pairplot")
fig = sns.pairplot(df, hue='hazardous')
st.pyplot(fig, use_container_width=False)

st.subheader("4. Boxplot")
for col in ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance']:
    fig = plt.figure(figsize=(5, 3))
    sns.boxplot(data=df, x='hazardous', y=col)
    st.pyplot(fig, use_container_width=False)