
def generate_data(number_of_samples: int) -> list:
    # names = generate_medication_names(number_of_samples)
    return list(range(1, number_of_samples + 1))

def generate_medication_names(number_of_samples: int) -> list:
    return [f"Medication{i}" for i in range(1, number_of_samples + 1)]