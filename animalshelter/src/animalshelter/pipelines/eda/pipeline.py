from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_for_eda

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_for_eda,       # Użyj funkcji z pliku nodes.py
                inputs="raw_data",             # Wejście: klucz z catalog.yml (nasze surowe dane)
                outputs="processed_data",      # Wyjście: nowa nazwa (klucz do zapisania przetworzonych danych)
                name="preprocess_data_for_eda_node"
            ),
        ]
    )