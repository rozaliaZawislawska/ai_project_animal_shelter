import pandas as pd
import unicodedata

def clean_text(text):
    if isinstance(text, str):
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        # Remove diacritics
        text = ''.join(c for c in text if not unicodedata.combining(c))
        # Replace remaining non-ASCII characters
        text = text.encode('ascii', 'replace').decode('ascii')
    return text

def preprocess_for_eda(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = raw_data.copy()

    df['Days in Shelter'] = pd.to_numeric(df['Days in Shelter'], errors='coerce')
    df['Days in Shelter'] = df['Days in Shelter'].astype('Int64')

    df['is_adopted'] = (
        df['Outcome Status']
        .str.startswith('Adopt')
        .astype(int)
    )

    # Clean text data in all string columns
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].apply(clean_text)

    print("\n--- Wstępna Weryfikacja Zmiennej Docelowej ---")
    print(f"Liczba wszystkich próbek: {len(df)}")
    print("Rozkład Adopcja (1) / Inny Wynik (0):")
    print(df['is_adopted'].value_counts(normalize=True))
    print("-" * 35)

    return df
