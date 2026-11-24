# Reporte Técnico: Sistema Integrado de Transporte Inteligente (RN-T3)

**Autores:** *(Completar nombres de las 2 parejas)*  
**Curso / Cohorte:** *(Completar)*  
**Repositorio:** *(Enlace al repositorio Git)*  
**Video demostrativo:** *(Enlace al video)*  

---

## 1. Resumen Ejecutivo y Arquitectura del Sistema

*(Presentar el problema integral de movilidad, el valor de negocio de integrar pronóstico de demanda, seguridad vial y recomendación de destinos, y un diagrama de la arquitectura MLOps reproducible y 100% offline).*

- **Módulo 1 (M1):** Pronóstico probabilístico de demanda y planificación de flota/conductores.
- **Módulo 2 (M2):** Clasificación de conducción distractiva con explicabilidad Grad-CAM.
- **Módulo 3 (M3):** Recomendación personalizada de destinos y optimización de campañas.
- **Interfaz unificada:** Aplicación web en Streamlit (`app/main.py`) consumiendo pesos preentrenados en `models/`.

---

## 2. Fundamentación Matemática

### 2.1 Pronóstico Cuantílico de Series de Tiempo (M1)
- Formulación de ventanas deslizantes y modelos secuenciales recurrentes (GRU/LSTM).
- Función de pérdida Pinball Loss para cuantiles $\tau \in \{0.10, 0.50, 0.90\}$:
  $$L_{\tau}(y, \hat{y}) = \max\big(\tau(y - \hat{y}), (\tau - 1)(y - \hat{y})\big)$$
- Modelo operativo de dimensionamiento de flota, flota de reserva y restricción de jornada laboral.

### 2.2 Redes Convolucionales, Transfer Learning y Grad-CAM (M2)
- Convolución 2D, entropía cruzada con *label smoothing* y transferencia de características.
- Formulación de Grad-CAM mediante ponderación por gradiente promedio global:
  $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}, \qquad L_{\text{Grad-CAM}}^c = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$
- Prevención de fuga de datos mediante deduplicación perceptual (pHash) y partición por conductor.

### 2.3 Aprendizaje de Representaciones y Recomendación Neuronal (M3)
- Factorización matricial neuronal, Neural Collaborative Filtering (NCF) y arquitectura Two-Towers con similitud coseno escalada por temperatura $\tau$.
- Métricas de ranking Top-K ($\text{HitRate@K}$, $\text{Precision@K}$, $\text{Recall@K}$, $\text{NDCG@K}$) y re-ranking de diversidad.
- Formulación de optimización de ROI bajo restricción de presupuesto de campaña.

---

## 3. Protocolo Experimental y Reproducibilidad (MLOps)

- Control estricto de semillas (`config/base.yaml`, `src/utils/seed.py`).
- Configuración declarativa y overrides CLI (`src/utils/config.py`, `run.py`).
- Protocolo de presupuesto justo y evaluación estadística en **30 semillas** (`--mode eval30`: media, desviación estándar, mínimo y máximo).
- Garantía *offline-first* (`data/processed/` y `models/`).

---

## 4. Módulo 1: Pronóstico de Demanda y Planificación de Flota

### 4.1 Datos, Preprocesamiento y EDA
*(Incluir figuras de `reports/figures/m1/` y análisis de estacionalidad/anomalías).*

### 4.2 Baselines vs. Arquitectura Neuronal (30 Semillas)
*(Tabla comparativa con media ± std, min y max de MAE, RMSE, Pinball Loss y cobertura P10-P90).*

### 4.3 Impacto de Negocio (Flota y Conductores)
*(Resultados de `business_fleet.py`: buses requeridos, turnos, costos evitados y análisis de sensibilidad).*

---

## 5. Módulo 2: Clasificación de Conducción Distractiva (Seguridad Vial)

### 5.1 Deduplicación, Split por Conductor y EDA
*(Reporte de duplicados removidos, prevención de data leakage y figuras de `reports/figures/m2/`).*

### 5.2 Baselines vs. CNN / Transfer Learning (30 Semillas)
*(Tabla comparativa con media ± std, min y max de Accuracy, Macro-F1, Recall en clases críticas y matriz de confusión).*

### 5.3 Explicabilidad Visual (Grad-CAM) y Calibración de Umbral de Alerta
*(Análisis cualitativo de mapas Grad-CAM, detección de atajos espurios y trade-off precisión vs. falsas alarmas).*

---

## 6. Módulo 3: Recomendación de Destinos y Optimización de Campañas

### 6.1 Datos, Dispersión, Cold-Start y EDA
*(Análisis de sparsity, long-tail y figuras de `reports/figures/m3/`).*

### 6.2 Baselines vs. NCF / Two-Towers (30 Semillas)
*(Tabla comparativa de HitRate@K, NDCG@K, cobertura y diversidad; visualización UMAP de embeddings).*

### 6.3 Impacto de Negocio (ROI de Campaña)
*(Resultados de `business_campaign.py` comparando asignación por modelo vs. popularidad/aleatoria).*

---

## 7. Aplicación Web Integrada (Streamlit)

*(Capturas de pantalla, flujo de usuario en las 3 páginas y validación del contrato de inferencia offline sin reentrenamiento).*

---

## 8. Ética, Privacidad, Sesgos y Riesgos Operativos

- **Vigilancia y privacidad en cabina (M2):** anonimización, procesamiento en el borde (*edge*), retención mínima de imágenes y consentimiento informado de conductores.
- **Sesgos demográficos y de iluminación (M2) / equidad de cobertura geográfica (M1 y M3):** evaluación por subgrupos y mitigación de *popularity bias*.
- **Impacto laboral y seguridad humana:** rol de apoyo a la decisión (*human-in-the-loop*), evitando automatización punitiva ante falsos positivos.

---

## 9. Declaración de Uso de Inteligencia Artificial Generativa

*(Describir qué herramientas de IA se utilizaron, para qué tareas —arquitectura, depuración, documentación— cómo se validó humanamente cada salida y remitir al registro detallado de fallos en [hallucinations.md](hallucinations.md)).*

---

## 10. Bibliografía (Formato APA 7)

*(Completar y verificar todas las referencias reales utilizadas; no incluir citas no verificadas).*

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.
- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural collaborative filtering. *Proceedings of the 26th International Conference on World Wide Web*, 173–182. https://doi.org/10.1145/3038912.3052569
- Koenker, R., & Bassett, G., Jr. (1978). Regression quantiles. *Econometrica*, 46(1), 33–50. https://doi.org/10.2307/1913643
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618–626. https://doi.org/10.1109/ICCV.2017.74
