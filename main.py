import face_recognition
known_image = face_recognition.load_image_file("2024-08-03-211238.jpg")
unknown_image = face_recognition.load_image_file("imgs/2_n.jpg")

# Detect face locations
face_locations = face_recognition.face_locations(known_image)
face_locations2 = face_recognition.face_locations(unknown_image)

print(face_locations == face_locations2)
# Check if any faces are found
if len(face_locations) == 0 or len(face_locations2)==0:
    print("No faces were found.")
    raise("No faces found in the image.")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
print(biden_encoding == unknown_encoding)

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
print(results)