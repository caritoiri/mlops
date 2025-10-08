import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep=";")

    # Normalizar nombres de columnas
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("\t", "")
    )

    # Selección de columnas relevantes
    selected_cols = [
        "Marital_status",
        "Application_mode",
        "Application_order",
        "Course",
        "Daytime/evening_attendance",
        "Previous_qualification",
        "Previous_qualification_grade",
        "Nacionality",
        "Mothers_qualification",
        "Fathers_qualification",
        "Admission_grade",
        "Debtor",
        "Tuition_fees_up_to_date",
        "Gender",
        "Scholarship_holder",
        "Age_at_enrollment",
        "International",
        "Unemployment_rate",
        "Inflation_rate",
        "GDP",
        "Target",
    ]

    # Filtrar columnas disponibles
    df = df[[c for c in selected_cols if c in df.columns]]

    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
