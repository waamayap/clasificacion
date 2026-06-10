# Modelo de Clasificación — Predicción de Acuerdos de Cobranza

## Descripción

Notebook de ciencia de datos para predecir si un cliente llegará a un **acuerdo de pago** (`con_acuerdo = 1`) durante un proceso de cobranza. Compara tres modelos de clasificación optimizados con Optuna.

---

## Estructura del Notebook

### 1. Preprocesamiento
- Carga del dataset (`prueba_analist_de_datos_crecere.csv`)
- Limpieza de nombres de columnas (tildes, espacios)
- Detección y eliminación de duplicados y valores nulos
- Eliminación de outliers (variable `TiempoActividad`)
- Ingeniería de fechas (`fecha_de_aplicacion` → mes/año)

### 2. Análisis Exploratorio (EDA)
- Distribución de la variable objetivo (`con_acuerdo`)
- Análisis de variables categóricas vs. target (barras apiladas %)
- Boxplots de variables numéricas por clase
- Correlación biserial puntual con el target
- Detección de multicolinealidad (correlación > 0.6)
- **Leakage detectado:** `deuda_total` eliminada por alta correlación espuria

### 3. Preparación de Features
- **Variables numéricas:** `gasto_mensual`, `ingresos_mensuales`, `cupo_solicitado`, `cupo_aprobado`, `edad`, `mes_aplicacion` → `MinMaxScaler`
- **Variables categóricas:** `origen_deudor`, `estado_civil`, `producto_origen_deuda`, `mujer`, `region_bogota`, `estrategia_agresiva`, `tiene_propiedades`, `beneficiario_subsidios` → `OneHotEncoder`
- Split: **60% train / 20% val / 20% test** (estratificado)

### 4. Modelos Entrenados

| Modelo | Optimización |
|---|---|
| Logistic Regression | Optuna (C, solver, penalty) |
| XGBoost | Optuna (n_estimators, max_depth, lr, etc.) |
| Random Forest | Optuna (n_estimators, max_depth, etc.) |

Cada modelo usa `Pipeline` con preprocesador integrado y maneja el **desbalanceo de clases** (`class_weight='balanced'` / `scale_pos_weight`).

### 5. Evaluación
- `classification_report` (precision, recall, F1)
- Curva ROC + AUC-ROC
- Matriz de confusión con porcentajes
- Importancia de variables (top 15)

---

## Requisitos

```bash
pip install pandas numpy scikit-learn xgboost optuna imbalanced-learn shap matplotlib seaborn plotly statsmodels joblib
```

---

## Variables más relevantes (según correlación biserial)

| Variable | Dirección | Nota |
|---|---|---|
| `estrategia_agresiva` | ↑ positiva | Mayor impacto |
| `propuesta_cuotas` | ↑ positiva | |
| `ingreso` | ↑ positiva | |
| `mujer` | ↓ negativa | Leve |
| `beneficiario_subsidios` | ↓ negativa | Leve |

---

## Notas
- `deuda_total` fue eliminada por ser posible **data leakage**.
- La métrica principal de comparación es **AUC-ROC**.
- Los pipelines están listos para serialización con `joblib`.
