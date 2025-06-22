from flask import Flask, render_template, redirect, url_for
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

app = Flask(__name__)
df = sns.load_dataset('titanic')
PLOT_DIR = 'static/images'

def save_plot(name, plot_func):
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    filepath = os.path.join(PLOT_DIR, name)
    if not os.path.exists(filepath):
        plot_func()
        plt.savefig(filepath, bbox_inches='tight')
        plt.clf()

# --- Plotting functions ---
def plot_survival(): sns.countplot(data=df, x='survived'); plt.title("Survival Count")
def plot_age_hist(): sns.histplot(df['age'].dropna(), kde=True); plt.title("Age Distribution")
def plot_class_pie(): df['class'].value_counts().plot.pie(autopct='%1.1f%%'); plt.title("Class Distribution"); plt.ylabel("")
def plot_box(): sns.boxplot(data=df, x='class', y='age'); plt.title("Age by Class")
def plot_violin(): sns.violinplot(data=df, x='survived', y='age'); plt.title("Violin: Age by Survival")
def plot_heatmap(): sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm'); plt.title("Correlation Heatmap")
def plot_scatter(): sns.scatterplot(data=df, x='age', y='fare', hue='survived'); plt.title("Fare vs Age")
def plot_embarked(): sns.countplot(data=df, x='embarked', hue='class'); plt.title("Embarkation vs Class")
def plot_swarm(): sns.swarmplot(data=df, x='sex', y='age', hue='survived'); plt.title("Swarm: Age, Sex, Survival")
def plot_boxen(): sns.boxenplot(data=df, x='embarked', y='fare'); plt.title("Boxen: Fare by Embarked")

# Map plot names to functions
plot_map = {
    'survival_count': plot_survival,
    'age_distribution': plot_age_hist,
    'class_pie': plot_class_pie,
    'boxplot_age_class': plot_box,
    'violin_age_survival': plot_violin,
    'heatmap_corr': plot_heatmap,
    'scatter_fare_age': plot_scatter,
    'embarked_class': plot_embarked,
    'swarm_age_sex_survival': plot_swarm,
    'boxen_fare_embarked': plot_boxen
}

@app.route('/')
def index():
    plots = list(plot_map.keys())
    return render_template('index.html', plots=plots)

@app.route('/plot/<plot_name>')
def show_plot(plot_name):
    if plot_name not in plot_map:
        return redirect(url_for('index'))
    filename = f"{plot_name}.png"
    save_plot(filename, plot_map[plot_name])
    return render_template('show_plot.html', image_file=filename, title=plot_name.replace("_", " ").title())

if __name__ == '__main__':
    app.run(debug=True)
