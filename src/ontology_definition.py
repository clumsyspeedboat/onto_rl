# src/ontology_definition.py

from owlready2 import *
import os
import logging


def create_ontology():
    """
    Create and save the Ping Pong ontology.
    """
    try:
        ontology = get_ontology("http://example.org/pingpong.owl")

        with ontology:
            class Ball(Thing):
                pass

            class Circle(Ball):
                pass

            class Square(Ball):
                pass

            class has_shape(DataProperty, FunctionalProperty):
                domain = [Ball]
                range = [str]

            class has_speed(DataProperty, FunctionalProperty):
                domain = [Ball]
                range = [float]

            # Assign properties to Ball instances with distinct names
            circle_obj = Circle("circle_obj")
            circle_obj.has_shape = "Circle"
            circle_obj.has_speed = 1.0

            square_obj = Square("square_obj")
            square_obj.has_shape = "Square"
            square_obj.has_speed = 1.5

        # Define the path to save the ontology
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ontology_dir = os.path.join(script_dir, '..', 'ontology')
        os.makedirs(ontology_dir, exist_ok=True)
        ontology_path = os.path.join(ontology_dir, 'pingpong.owl')

        # Save the ontology to a file
        ontology.save(file=ontology_path, format="rdfxml")
        logging.info(f"Ontology successfully created and saved to {ontology_path}")

    except Exception as e:
        logging.error(f"Failed to create ontology: {e}")


if __name__ == "__main__":
    # Setup basic logging for the ontology creation script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("ontology/ontology_creation.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    create_ontology()
