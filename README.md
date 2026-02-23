# 🚢 Titanic Survival Prediction - Machine Learning Project

Proyecto de Machine Learning utilizando el dataset Titanic de Kaggle.

Este proyecto implementa un pipeline completo de procesamiento de datos y entrenamiento de modelo, incluyendo optimización de hiperparámetros mediante GridSearch y validación cruzada.

---

## 📊 Dataset

Dataset obtenido desde Kaggle:  
Titanic - Machine Learning from Disaster

Archivo utilizado:
- train.csv

---

## 🧠 Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## ⚙️ Técnicas aplicadas

✔ Análisis Exploratorio de Datos (EDA)  
✔ Feature Engineering (FamilySize)  
✔ Pipeline con ColumnTransformer  
✔ Imputación de valores faltantes  
✔ Escalado de variables numéricas  
✔ OneHotEncoding para variables categóricas  
✔ RandomForestClassifier  
✔ GridSearchCV  
✔ Cross Validation (cv=5)  
✔ Evaluación con Accuracy, Precision, Recall y F1-score  
✔ Exportación del modelo a archivo .pkl  

---

## 📂 Estructura del proyecto

```
proyecto_titanic_ml/
│
├── data/
│   └── train.csv
│
├── notebook/
│   └── titanic_pipeline.ipynb
│
├── modelo/
│   └── modelo_titanic.pkl
│
├── requirements.txt
└── README.md
```

---

## 🚀 Cómo ejecutar el proyecto

1️⃣ Clonar el repositorio

```
git clone https://github.com/Laugalin/titanic-ml-pipeline.git
```

2️⃣ Crear entorno virtual

```
python -m venv .venv
```

3️⃣ Activar entorno

En Windows:
```
.venv\Scripts\activate
```

4️⃣ Instalar dependencias

```
pip install -r requirements.txt
```

5️⃣ Ejecutar el notebook

```
jupyter notebook
```

---

## 📈 Modelo Final

Modelo utilizado: RandomForestClassifier  
Optimización realizada con GridSearchCV  
Validación cruzada con 5 folds  

El modelo final fue exportado como:

```
modelo/modelo_titanic.pkl
```

Este archivo contiene:
- Preprocesamiento completo
- Feature Engineering
- Modelo entrenado
- Mejores hiperparámetros

---

## 🎯 Objetivo del proyecto

Aplicar los conocimientos de:

- Pipelines
- ColumnTransformer
- GridSearch
- Cross Validation
- Feature Engineering
- Exportación de modelos para producción

---

## 👩‍💻 Autor

Laura Galindo  
Ingeniería en Software