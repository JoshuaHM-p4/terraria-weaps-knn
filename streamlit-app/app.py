import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('knn_model.pkl')
model = load_model()

# Load dataframe
@st.cache_data
def load_dataframe():
    return pd.read_csv('cleaned_df.csv')
df = pd.read_csv('cleaned_df.csv')

# Get Features
X = df.drop(['NAME', 'CLASS'], axis=1)

# Standardize features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X)

# Streamlit UI Title
st.title("Terraria Weapon KNN Recommender")

# Scatter
fig = px.scatter(
    df,
    x = "DPS (SINGLE TARGET)",
    y = "DPS (MULTI TARGET)",
    color = "GAME PROGRESSION",
    hover_name = "NAME",
    title="Weapon DPS Comparison",
    log_y=True,
    log_x=True,
)
st.plotly_chart(fig)

# st.write(df[['NAME', 'CLASS']])

# Input
user_input = st.text_input("Search for a weapon: ")

# Reset highlights
df['HIGHLIGHT'] = 'Other'

if user_input:
    # Filter with case-insensitive partial match
    entry = df[df['NAME'].str.contains(user_input, case=False, na=False)]
    entry_list = entry.index.tolist()

    if not entry.empty and entry_list:

        entry_selections = entry['NAME'].tolist()
        entry_selections.insert(0, "All")
        selection = st.pills("🔍 Matching Searches:", entry_selections, selection_mode="multi")

        if "All" in selection:
            selection = entry_selections[1:]
        if not selection:
            selection = entry_selections[1:]
        entry_list = [entry.index[entry['NAME'] == i][0] for i in selection]

        st.write("### Selected Weapons:")
        for idx, i in enumerate(entry_list):
            df.at[i, 'HIGHLIGHT'] = 'Main'

            # Get neighbors
            distances, indices = model.kneighbors(df_scaled[i].reshape(1, -1), n_neighbors=5)
            neighbor_indices = indices[0][1:]

            # Mark neighbors in the DataFrame
            for neighbor_idx in neighbor_indices:
                df.at[neighbor_idx, 'HIGHLIGHT'] = 'Neighbor'

            # Display weapon info
            with st.columns(3)[1]:
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                st.write("#", df.iloc[i]['NAME'])
                st.image(f'icons/{df.iloc[i]["NAME"]}.png', width=100)
                st.write("**DPS (SINGLE TARGET)**:", df.iloc[i]['DPS (SINGLE TARGET)'])
                st.write("**DPS (MULTI TARGET)**:", df.iloc[i]['DPS (MULTI TARGET)'])
                st.write("**CLASS**:", df.iloc[i]['CLASS'])
                st.link_button("View on Wiki", f"https://terraria.wiki.gg/wiki/{df.iloc[i]['NAME'].replace(' ', '_')}")
                st.markdown("</div>", unsafe_allow_html=True)

            # Display neighbors
            st.write("### Similar Weapons:")
            columns = st.columns(2)
            st.write('---')

            neighbor_list = df.iloc[neighbor_indices]
            for loop_idx, (jdx, neighbor) in enumerate(neighbor_list.iterrows()):
                col_idx = loop_idx % 2
                with columns[col_idx]:
                    with st.container(border=True):
                        st.write(f"###", neighbor['NAME'])
                        st.image(f'icons/{neighbor["NAME"]}.png', width=50)
                        st.write("**DPS (SINGLE TARGET)**:", neighbor['DPS (SINGLE TARGET)'])
                        st.write("**DPS (MULTI TARGET)**:", neighbor['DPS (MULTI TARGET)'])
                        st.write("**CLASS**:", neighbor['CLASS'])
                        st.link_button("View on Wiki", f"https://terraria.wiki.gg/wiki/{neighbor['NAME'].replace(' ', '_')}")
    else:
        st.warning("No matching weapons found.")

st.write("### Dataframe Overview")

# Scatter Plot (Now placed AFTER HIGHLIGHT is set)
fig = px.scatter(
    df,
    x="DPS (SINGLE TARGET)",
    y="DPS (MULTI TARGET)",
    color="HIGHLIGHT",
    hover_name="NAME",
    title="Weapon DPS Plot",
    log_y=True,
    log_x=True
)
st.plotly_chart(fig)

dps_columns = [
    'DPS (SINGLE TARGET)',
    'DPS (MULTI TARGET)',
    'DPS (SINGLE TARGET + PROJECTILE ONLY)',
    'DPS (MULTI TARGET + PROJECTILE ONLY)'
]

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))
# Create the boxplot
sns.boxplot(data=df, x='GAME PROGRESSION', y='DPS (SINGLE TARGET)', ax=ax)
# Set log scale and title
ax.set_yscale('log')
ax.set_title('DPS (SINGLE TARGET) Across Game Progression Levels (LOG)')
# Render in Streamlit
st.pyplot(fig)

st.write("#### Boxplots of DPS across different classes")
for i in range(0, len(dps_columns), 2):
    col1, col2 = st.columns(2)
    for col, dps_col in zip([col1, col2], dps_columns[i:i+2]):
        with col:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='CLASS', y=dps_col, ax=ax)
            ax.set_yscale('log')
            ax.set_title(f'{dps_col} Across Classes')
            st.pyplot(fig)

st.write('---')

# Footer
st.write("- Made with ❤️ by [JoshuaHM-p4](https://github.com/JoshuaHM-p4/)")
st.write("- Terraria DPS Data (Game Version: 1.4.4.9) from [Kaggle by Andres Coronel](https://www.kaggle.com/datasets/acr1209/all-terraria-weapons-dps-v-1449)")
st.write("- KNN Model trained using scikit-learn through [Kaggle Notebooks](https://www.kaggle.com/code/joshuamistal/terraria-weapon-recommender-using-knn).")
st.write("- Icons from [Terraria Wiki](https://terraria.wiki.gg/wiki/).")

