# ontology_definition.py

from owlready2 import *

# Create and load ontology
onto = get_ontology("http://example.org/pingpong.owl")

with onto:
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

# Save the ontology to a file
onto.save(file="pingpong.owl", format="rdfxml")
