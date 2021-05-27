import faker
import numpy as np


def generate_labels(unique_values, number):
    items = list(range(unique_values))
    for _ in range(number - unique_values):
        items.append(np.random.randint(unique_values))
    np.random.shuffle(items)
    return items


def generate_texts(number):
    fake = faker.Faker()
    return [fake.text() for _ in range(number)]
