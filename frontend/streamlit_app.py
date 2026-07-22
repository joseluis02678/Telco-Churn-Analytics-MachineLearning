import os

import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "https://telco-churn-analytics-machinelearning.onrender.com"
)


st.set_page_config(
    page_title="Predicción de abandono",
    page_icon="📊",
    layout="wide"
)

st.write("API usada:", API_URL)

def check_backend() -> bool:
    """Comprueba si el backend FastAPI está disponible."""
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )
        return response.status_code == 200

    except requests.exceptions.RequestException:
        return False


def request_prediction(customer_data: dict) -> dict:
    """Envía los datos del cliente al endpoint de predicción."""
    response = requests.post(
        f"{API_URL}/predict",
        json=customer_data,
        timeout=15
    )

    response.raise_for_status()
    return response.json()


def show_prediction(result: dict) -> None:
    """Muestra visualmente la respuesta de FastAPI."""
    churn_prediction = result.get("churn_prediction", "No disponible")
    probability = float(result.get("churn_probability", 0))
    risk_level = result.get("risk_level", "No disponible")
    model_version = result.get("model_version", "No disponible")

    percentage = probability * 100

    st.divider()
    st.subheader("Resultado de la predicción")

    metric_column_1, metric_column_2 = st.columns(2)

    with metric_column_1:
        st.metric(
            "Probabilidad de abandono",
            f"{percentage:.2f}%"
        )

    with metric_column_2:
        st.metric(
            "Nivel de riesgo",
            risk_level
        )

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    if churn_prediction == "Yes":
        st.error(
            "⚠️ El cliente presenta riesgo de abandonar el servicio."
        )

        st.info(
            "Se recomienda contactar al cliente y evaluar descuentos, "
            "promociones o mejoras en la atención."
        )

    else:
        st.success(
            "✅ El cliente probablemente permanecerá en el servicio."
        )

    st.caption(f"Modelo utilizado: {model_version}")

    with st.expander("Ver respuesta completa de la API"):
        st.json(result)


st.title("📊 Predicción de abandono de clientes")

st.write(
    "Completa la información del cliente para estimar la probabilidad "
    "de que abandone el servicio."
)


if check_backend():
    st.success(f"Backend conectado correctamente: {API_URL}")
else:
    st.error(f"No se pudo conectar con FastAPI: {API_URL}")


with st.form("customer_form"):

    st.subheader("Información general")

    column_1, column_2, column_3 = st.columns(3)

    with column_1:
        tenure = st.number_input(
            "Meses de permanencia",
            min_value=0.0,
            value=12.0,
            step=1.0
        )

    with column_2:
        monthly_charges = st.number_input(
            "Cargo mensual en USD",
            min_value=0.0,
            value=70.35,
            step=1.0
        )

    with column_3:
        total_charges = st.number_input(
            "Cargo total acumulado en USD",
            min_value=0.0,
            value=845.50,
            step=1.0
        )

    st.subheader("Contrato y servicios")

    column_1, column_2 = st.columns(2)

    with column_1:

        contract = st.selectbox(
            "Tipo de contrato",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        internet_service = st.selectbox(
            "Servicio de internet",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        payment_method = st.selectbox(
            "Método de pago",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        online_security = st.selectbox(
            "Seguridad en línea",
            ["Yes", "No"]
        )

        tech_support = st.selectbox(
            "Soporte técnico",
            ["Yes", "No"]
        )

    with column_2:

        online_backup = st.selectbox(
            "Respaldo en línea",
            ["Yes", "No"]
        )

        device_protection = st.selectbox(
            "Protección de dispositivos",
            ["Yes", "No"]
        )

        streaming_movies = st.selectbox(
            "Películas en streaming",
            ["Yes", "No"]
        )

        streaming_tv = st.selectbox(
            "TV en streaming",
            ["Yes", "No"]
        )

        paperless_billing = st.selectbox(
            "Facturación electrónica",
            ["Yes", "No"]
        )

    submitted = st.form_submit_button(
        "Realizar predicción",
        use_container_width=True
    )


if submitted:

    customer_data = {
        "tenure": float(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "Contract": contract,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "StreamingMovies": streaming_movies,
        "StreamingTV": streaming_tv,
        "PaperlessBilling": paperless_billing
    }

    try:
        with st.spinner("Analizando los datos del cliente..."):
            result = request_prediction(customer_data)

        show_prediction(result)

    except requests.exceptions.HTTPError as error:

        status_code = error.response.status_code

        if status_code == 422:
            st.error(
                "Los datos enviados no coinciden con CustomerData."
            )
        elif status_code == 500:
            st.error(
                "El backend tuvo un error al realizar la predicción."
            )
        else:
            st.error(
                f"FastAPI devolvió el error HTTP {status_code}."
            )

        try:
            st.json(error.response.json())
        except ValueError:
            st.write(error.response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "No se pudo conectar con FastAPI."
        )

    except requests.exceptions.Timeout:
        st.error(
            "La API tardó demasiado tiempo en responder."
        )

    except requests.exceptions.RequestException as error:
        st.error(
            f"Error de comunicación con FastAPI: {error}"
        )

    except (TypeError, ValueError) as error:
        st.error(
            f"La respuesta de la API tiene un formato incorrecto: {error}"
        )