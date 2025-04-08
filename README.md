# Terraria Weapon KNN Recommender

[![View Website](https://img.shields.io/badge/View-Website-brightgreen)](https://terraria-weapons-knn.streamlit.app/)

## Overview
The **Terraria Weapon KNN Recommender** is a Streamlit-based web application that helps players of the game Terraria find similar weapons based on their stats. Using a K-Nearest Neighbors (KNN) model, the app recommends weapons that are statistically similar to the selected weapon, providing insights into their performance and progression in the game.

## Features
- **Weapon Search**: Search for weapons by name using a case-insensitive partial match.
- **Weapon Recommendations**: View similar weapons based on the KNN model.
- **Weapon Stats Visualization**:
  - Scatter plots comparing single-target and multi-target DPS.
  - Boxplots showing DPS distribution across game progression levels and weapon classes.
- **Interactive UI**: Explore weapon stats and recommendations with an intuitive interface.
- **Links to Wiki**: Direct links to the Terraria Wiki for detailed weapon information.

## Data Sources
- **Weapon DPS Data**: [Kaggle Dataset by Andres Coronel](https://www.kaggle.com/datasets/acr1209/all-terraria-weapons-dps-v-1449) (Game Version: 1.4.4.9).
- **Icons**: [Terraria Wiki](https://terraria.wiki.gg/wiki/).

## How to Run the App
1. Clone the repository:
   ```bash
   git clone https://github.com/JoshuaHM-p4/terraria-weapon-knn-recommender.git
   cd terraria-weapon-knn-recommender
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run streamlit-app/app.py
   ```
4. Open the app in your browser at `http://localhost:8501`.

## Usage Instructions
1. **Search for a Weapon**:
   - Enter the name of a weapon in the search bar.
   - Select one or more matching weapons from the search results.
2. **View Recommendations**:
   - The app highlights the selected weapon and its nearest neighbors in the scatter plot.
   - Detailed stats and images of the selected weapon and its neighbors are displayed.
3. **Explore Visualizations**:
   - Use the scatter plots and boxplots to analyze weapon performance across different categories.

## Model Details
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Features Used**: Weapon DPS (Single Target, Multi Target, etc.), Game Progression, and other numerical attributes.
- **Preprocessing**: StandardScaler for feature standardization.

## Acknowledgments
- Made with ❤️ by [JoshuaHM-p4](https://github.com/JoshuaHM-p4/).
- Data and inspiration from the Terraria community.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
