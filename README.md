# 🚢 Titanic Dataset Visualization Web App (Flask)

This is a Flask-based web application that generates and displays **10+ visualizations** using the Titanic dataset. The goal is to explore the dataset interactively and demonstrate various data visualization techniques.

---

## 📊 Visualizations Included

This app uses **Matplotlib** and **Seaborn** to create the following visualizations:

1. **Survival Count (Bar Plot)**  
   - Displays the number of passengers who survived vs those who didn't.

2. **Age Distribution (Histogram with KDE)**  
   - Shows the distribution of passenger ages with a smooth density curve.

3. **Class Distribution (Pie Chart)**  
   - Percentage distribution of passengers across First, Second, and Third class.

4. **Box Plot: Age vs Class**  
   - Compares age distributions for different passenger classes.

5. **Violin Plot: Age vs Survival**  
   - Distribution of passenger ages for survivors and non-survivors.

6. **Correlation Heatmap**  
   - Shows correlations between numeric variables like age, fare, and survival.

7. **Fare vs Age (Scatter Plot)**  
   - Scatter plot showing fare paid vs passenger age, colored by survival.

8. **Embarked vs Class (Count Plot)**  
   - Number of passengers from each embarkation point grouped by class.

9. **Swarm Plot: Age by Sex and Survival**  
   - Detailed point-wise visualization of age distribution by sex and survival status.

10. **Boxen Plot: Fare by Embarkation Point**  
    - Detailed distribution of fare values grouped by the port of embarkation.

---

## 🛠️ Technologies Used

- Python 3  
- Flask  
- HTML5  
- CSS3  
- Bootstrap 5

---

## 🔗 To Create Virtual Environment
To create virtual env --> **Python -m venv Name_of_Virtual_env** (for e.g. --> python -m venv myenv)  
To activate --> **Name_of_virtual_env/Scripts/activate** (for e.g. --> myenv/Scripts/activate)

---

## 🖥️ To Install Dependencies
pip install -r requirements.txt  

---

## 🚨 To Run App Using Flask 
python app.py

---

## 🌐 Live Demo
Check out the live deployed project here: [View Project on Render](https://celebal-assignment-2.onrender.com)
