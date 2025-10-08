import argparse
from azureml.core import Run, Model
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, required=True)
    parser.add_argument("--eval_metrics", type=str, required=False)
    args = parser.parse_args()

    # 1️⃣ Obtener el contexto actual del pipeline (ya autenticado)
    run = Run.get_context()
    ws = run.experiment.workspace

    print(f"📦 Conectado automáticamente al workspace: {ws.name}")

    # 2️⃣ Datos del modelo
    model_name = "modelo_abandono_estudiantil"
    description = (
        "Modelo predictivo de abandono estudiantil basado en variables académicas y socioeconómicas."
    )

    # 3️⃣ Registrar el modelo directamente en el workspace
    registered_model = Model.register(
        workspace=ws,
        model_name=model_name,
        model_path=args.model_input,  # Carpeta del modelo entrenado
        description=description,
        tags={
            "autor": "Jose Iriarte",
            "pipeline": "dropout_pipeline",
            "evaluacion": args.eval_metrics if args.eval_metrics else "N/A",
        },
    )

    # 4️⃣ Mostrar confirmación
    print("✅ Modelo registrado correctamente.")
    print(f"   • Nombre: {registered_model.name}")
    print(f"   • Versión: {registered_model.version}")
    print(f"   • ID: {registered_model.id}")

if __name__ == "__main__":
    main()
