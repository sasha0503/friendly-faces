from test_inference import detect

import numpy as np

import os


embeddings_database = {}

data_path = 'data/people'
for person in os.listdir(data_path):
    print(f'Processing {person}')
    person_embeddings = []
    person_path = os.path.join(data_path, person)
    if not os.path.isdir(person_path):
        continue
    for image in os.listdir(person_path):
        image_path = os.path.join(person_path, image)
        try:
            img_probs, img_embedding = detect(image_path)
            if img_embedding.shape[1] != 8631:
                print(f'Error processing: {image_path}')
                continue
        except:
            print(f'Error processing: {image_path}')
            continue
        person_embeddings.append(img_embedding)

    embeddings_database[person] = person_embeddings

# Save embeddings database
np.save('embeddings_database_new.npy', embeddings_database)
