import pandas as pd

def preprocess_for_eda(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = raw_data.copy()

    df['Days in Shelter'] = pd.to_numeric(df['Days in Shelter'], errors='coerce')
    df['Days in Shelter'] = df['Days in Shelter'].astype('Int64')

    df['is_adopted'] = (
        df['Outcome Status']
        .str.startswith('Adopt')
        .astype(int)
    )

    print("\n--- Wstępna Weryfikacja Zmiennej Docelowej ---")
    print(f"Liczba wszystkich próbek: {len(df)}")
    print("Rozkład Adopcja (1) / Inny Wynik (0):")
    print(df['is_adopted'].value_counts(normalize=True))
    print("-" * 35)

    return df

